import copy
import os
import pickle
import shutil
from functools import partial
import json

import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.converter.utils import contains_explicit_return, process_memory
from scenariomax.unified_to_tfrecord.build_tfexample import build_tfexample
from scenariomax.unified_to_gpudrive.build_gpudrive_example import build_gpudrive_example


logger = get_logger(__name__)


def single_worker_preprocess(x, worker_index):
    """
    All scenarios passed to write_to_directory_single_worker will be preprocessed. The input is expected to be a list.
    The output should be a list too. The element in the second list will be processed by convertors. By default, you
    don't need to provide this processor. We override it for waymo convertor to release the memory in time.
    :param x: input
    :param worker_index: worker_index, useful for logging
    :return: input
    """
    return x


def write_to_directory(
    convert_func,
    write_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    num_workers=8,
    write_pickle=False,
    preprocess=None,
    write_json=False,
    **kwargs,
):
    if preprocess is None:
        preprocess = single_worker_preprocess

    # make sure dir not exist
    kwargs_for_workers = [{} for _ in range(num_workers)]
    for key, value in kwargs.items():
        for i in range(num_workers):
            kwargs_for_workers[i][key] = value[i]

    save_path = copy.deepcopy(output_path)
    if os.path.exists(output_path):
        logger.info(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(save_path, exist_ok=False)
    logger.info(f"Created output directory: {save_path}")

    basename = os.path.basename(output_path)

    for i in range(num_workers):
        subdir = os.path.join(output_path, f"{basename}_{i!s}")
        os.makedirs(subdir, exist_ok=False)
        logger.debug(f"Created worker directory: {subdir}")

    # get arguments for workers
    num_files = len(scenarios)
    if num_files < num_workers:
        # single process
        logger.info(f"Using single worker as number of scenarios ({num_files}) < number of workers ({num_workers})")
        num_workers = 1

    argument_list = []
    output_pathes = []
    num_files_each_worker = int(num_files // num_workers)

    for i in range(num_workers):
        end_idx = num_files if i == num_workers - 1 else (i + 1) * num_files_each_worker

        subdir = os.path.join(output_path, f"{basename}_{i!s}")
        output_pathes.append(subdir)
        argument_list.append([scenarios[i * num_files_each_worker : end_idx], kwargs_for_workers[i], i, subdir])
        logger.debug(f"Worker {i} will process {len(scenarios[i * num_files_each_worker : end_idx])} scenarios")

    # prefill arguments
    func = partial(
        writing_to_directory_wrapper,
        convert_func=convert_func,
        write_func=write_func,
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        write_pickle=write_pickle,
    )

    logger.info(f"Starting processing {num_files} scenarios with {num_workers} workers")

    with Parallel(n_jobs=num_workers) as parallel:
        done = parallel(delayed(func)(arg) for arg in argument_list)

    for i, d in enumerate(done):
        if not d:
            err_msg = f"Worker {i} failed!"
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Merge tfrecord files into one
    if write_json:
        merge_json_files(save_path)
    elif not write_pickle:
        merge_and_clean_tfrecord_files(save_path, f"{dataset_name}.tfrecord")


def merge_and_clean_tfrecord_files(output_dir: str, merged_filename: str) -> None:
    # List all TFRecord files in the output directory
    tfrecord_files = []

    for out_dir in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, out_dir)):
            list_dir = os.listdir(os.path.join(output_dir, out_dir))
            tfrecord_files += [os.path.join(output_dir, out_dir, f) for f in list_dir if f.endswith(".tfrecord")]

    logger.info(f"Found {len(tfrecord_files)} TFRecord files to merge")

    # Define the path for the merged TFRecord file
    merged_file_path = os.path.join(output_dir, merged_filename)
    logger.info(f"Merging files into: {merged_file_path}")

    total_records = 0
    with tf.io.TFRecordWriter(merged_file_path) as writer:
        for tfrecord_file in tqdm(tfrecord_files, desc="Merging TFRecord files"):
            # Read the current TFRecord file
            dataset = tf.data.TFRecordDataset(tfrecord_file)

            file_records = 0
            for record in dataset:
                writer.write(record.numpy())
                file_records += 1

            total_records += file_records
            logger.debug(f"Merged {file_records} records from {tfrecord_file}")

            # Delete the current directory
            dir_to_remove = os.path.dirname(tfrecord_file)
            shutil.rmtree(dir_to_remove)
            logger.debug(f"Removed directory: {dir_to_remove}")

    # Shuffle the merged TFRecord file
    logger.info(f"Shuffling merged file with {total_records} records")
    shuffle_tfrecord_file(merged_file_path)

    logger.info(f"Successfully merged and shuffled TFRecord file at {merged_file_path}")


def shuffle_tfrecord_file(tfrecord_file: str, buffer_size: int = 10000) -> None:
    # Create a temporary file to store shuffled records
    temp_file = tfrecord_file + ".shuffled"
    logger.debug(f"Creating temporary shuffled file: {temp_file}")

    # Read the original TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.shuffle(buffer_size)

    # Write the shuffled records to the temporary file
    record_count = 0
    with tf.io.TFRecordWriter(temp_file) as writer:
        for record in dataset:
            writer.write(record.numpy())
            record_count += 1

    logger.debug(f"Wrote {record_count} shuffled records to temporary file")

    # Replace the original file with the shuffled file
    os.replace(temp_file, tfrecord_file)
    logger.debug("Replaced original file with shuffled file")

def merge_json_files(output_dir: str):
    json_files = []

    for out_dir in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, out_dir)):
            list_dir = os.listdir(os.path.join(output_dir, out_dir))
            json_files += [os.path.join(output_dir, out_dir, f) for f in list_dir if f.endswith(".json")]
    
    logger.info(f"Found {len(json_files)} JSON files to merge")
    logger.info(f"Merging files into: {output_dir}")
    for file in json_files:
        shutil.move(file, output_dir)
    
    for out_dir in os.listdir(output_dir):
        shutil.rmtree(os.path.join(output_dir, out_dir))

    logger.info(f"All json files moved to {output_dir} and subdirs deleted.")

def writing_to_directory_wrapper(
    args,
    convert_func,
    write_func,
    dataset_version,
    dataset_name,
    write_pickle,
    preprocess=single_worker_preprocess,
):
    return write_to_directory_single_worker(
        convert_func=convert_func,
        write_func=write_func,
        scenarios=args[0],
        output_path=args[3],
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        write_pickle=write_pickle,
        worker_index=args[2],
        **args[1],
    )


def write_to_directory_single_worker(
    convert_func,
    write_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    worker_index=0,
    write_pickle=False,
    preprocess=single_worker_preprocess,
    **kwargs,
):
    """
    Convert a batch of scenarios.
    """
    logger.debug(f"Worker {worker_index} starting with {len(scenarios)} scenarios")

    if not contains_explicit_return(convert_func):
        err_msg = "The convert function should return a metadata dict"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("The specified version in kwargs is replaced by argument: 'dataset_version'")

    memory_before = process_memory()
    logger.debug(f"Worker {worker_index} memory before preprocessing: {memory_before:.2f} MB")

    scenarios = preprocess(scenarios, worker_index)

    memory_after = process_memory()
    logger.debug(
        f"Worker {worker_index} memory after preprocessing: {memory_after:.2f} MB (delta: {memory_after - memory_before:.2f} MB)",
    )

    pbar = tqdm(desc=f"Worker {worker_index} processing scenarios")
    processed_count = 0
    error_count = 0

    if write_pickle:
        for scenario in scenarios:
            try:
                sd_scenario, export_file_name = process_scenario(
                    scenario,
                    convert_func,
                    dataset_version,
                    dataset_name,
                    **kwargs,
                )

                with open(os.path.join(output_path, export_file_name), "wb") as f:
                    pickle.dump(sd_scenario, f)

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count, "errors": error_count})
            except Exception as e:
                error_count += 1
                logger.error(f"Worker {worker_index} failed to process scenario: {e}")
    else:
        write_func(output_path, worker_index, scenarios, convert_func, dataset_version, dataset_name, pbar, **kwargs)

    memory_final = process_memory()
    logger.info(
        f"Worker {worker_index} finished! Processed {processed_count} scenarios with {error_count} errors. "
        f"Memory usage: {memory_final:.2f} MB, Files saved at: {output_path}",
    )

    return True


def process_scenario(scenario, convert_func, dataset_version, dataset_name, **kwargs):
    """Process a single scenario and convert it to the desired format.

    Args:
        scenario: The scenario to be processed.
        convert_func: The function used to convert the scenario.
        dataset_name: The name of the dataset.
        **kwargs: Additional arguments for the conversion function.

    Returns:
        sd_scenario: The converted scenario.
        export_file_name: The name of the file where the scenario will be saved.
    """
    sd_scenario = convert_func(scenario, dataset_version, **kwargs)
    scenario_id = sd_scenario[SD.ID]
    export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

    sd_scenario = sd_scenario.to_dict()
    SD.sanity_check(sd_scenario, check_self_type=True)

    return sd_scenario, export_file_name

def write_tf_record(output_path, worker_index, scenarios, convert_func, dataset_version, dataset_name, pbar, **kwargs):
    processed_count = 0
    error_count = 0
    tf_record_file = os.path.join(output_path, "training.tfrecord")
    logger.info(f"Worker {worker_index} writing to TFRecord: {tf_record_file}")

    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for scenario in scenarios:
            try:
                sd_scenario, _ = process_scenario(
                    scenario,
                    convert_func,
                    dataset_version,
                    dataset_name,
                    **kwargs,
                )

                dict_to_convert = build_tfexample(sd_scenario)
                example = tf.train.Example(features=tf.train.Features(feature=dict_to_convert))

                if example is not None:
                    writer.write(example.SerializeToString())
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_count, "errors": error_count})
            except Exception as e:
                error_count += 1
                logger.error(f"Worker {worker_index} failed to process scenario: {str(e)}")

def write_gpudrive_json(output_path, worker_index, scenarios, convert_func, dataset_version, dataset_name, pbar, **kwargs):
    logger.info(f"Worker {worker_index} writing to JSON: {output_path}")
    processed_count = 0
    error_count = 0
    for scenario in scenarios:
        try:
            sd_scenario, export_file_name = process_scenario(
                scenario, 
                convert_func, 
                dataset_version, 
                dataset_name, 
                **kwargs
            )
            gpudrive_json_file = os.path.join(output_path, f"{dataset_name}_{sd_scenario[SD.ID]}.json")
            example = build_gpudrive_example(os.path.basename(export_file_name), sd_scenario)
            if example is not None:
                with open(gpudrive_json_file, "w") as f:
                    json.dump(example, f)
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count, "errors": error_count})
        except Exception as e:
            error_count += 1
            logger.error(f"Worker {worker_index} failed to process scenario: {str(e)}")
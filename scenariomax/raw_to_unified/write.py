import copy
import json
import os
import pickle
import shutil
from collections.abc import Callable, Generator, Iterable
from functools import partial
from typing import Any

import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.utils import contains_explicit_return, process_memory
from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.unified_to_gpudrive.build_gpudrive_example import build_gpudrive_example
from scenariomax.unified_to_tfexample.build_tfexample import build_tfexample
from scenariomax.unified_to_tfexample.exceptions import NotEnoughValidObjectsException, OverpassException


logger = get_logger(__name__)


def single_worker_preprocess(
    x: list[Any] | Generator[Any, None, None],
    worker_index: int,
) -> list[Any] | Generator[Any, None, None]:
    """
    All scenarios passed to write_to_directory_single_worker will be preprocessed. The input can be a list or generator.
    The output should be a list or generator too. The element in the output will be processed by convertors.
    By default, you don't need to provide this processor.
    We override it for waymo convertor to release the memory in time.

    Args:
        x: Input scenarios (list or generator)
        worker_index: Worker index, useful for logging

    Returns:
        Processed input scenarios (list or generator)
    """
    return x


def write_to_directory(
    convert_func: Callable,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    output_path: str,
    dataset_version: str,
    dataset_name: str,
    num_workers: int = 8,
    preprocess: Callable | None = None,
    postprocess_func: Callable | None = None,
    **kwargs,
) -> None:
    """
    Write scenarios to directory using multiple workers.

    Args:
        convert_func: Function to convert scenarios
        postprocess_func: Optional post-processing function
        scenarios: List or generator of scenarios to process
        output_path: Path to output directory
        dataset_version: Dataset version
        dataset_name: Dataset name
        num_workers: Number of parallel workers
        preprocess: Optional preprocessing function
        **kwargs: Additional keyword arguments for workers
    """
    if preprocess is None:
        preprocess = single_worker_preprocess

    kwargs_for_workers = [{} for _ in range(num_workers)]
    for key, value in kwargs.items():
        for i in range(num_workers):
            kwargs_for_workers[i][key] = value

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

    # Get arguments for workers
    num_files = len(scenarios)
    if num_files < num_workers:
        logger.info(f"Using single worker as number of scenarios ({num_files}) < number of workers ({num_workers})")
        num_workers = 1

    argument_list = []
    output_paths = []
    num_files_each_worker = num_files // num_workers

    for i in range(num_workers):
        start_idx = i * num_files_each_worker
        end_idx = num_files if i == num_workers - 1 else start_idx + num_files_each_worker

        subdir = os.path.join(output_path, f"{basename}_{i!s}")
        output_paths.append(subdir)
        scenario_chunk = scenarios[start_idx:end_idx]
        argument_list.append([scenario_chunk, kwargs_for_workers[i], i, subdir])

        logger.info(f"Worker {i} will process {len(scenario_chunk)} items (indices {start_idx}:{end_idx})")

    func = partial(
        write_to_directory_single_worker,
        convert_func=convert_func,
        postprocess_func=postprocess_func,
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
    )

    logger.info(f"Starting processing {num_files} scenarios with {num_workers} workers")

    with Parallel(n_jobs=num_workers) as parallel:
        done = parallel(
            delayed(func)(list_scenarios=arg[0], worker_kwargs=arg[1], worker_index=arg[2], output_path=arg[3])
            for arg in argument_list
        )

    for i, d in enumerate(done):
        if not d:
            err_msg = f"Worker {i} failed!"
            logger.error(err_msg)
            raise ValueError(err_msg)

    if postprocess_func == postprocess_gpudrive:
        merge_json_files(save_path)
    elif postprocess_func == postprocess_tfexample:
        merge_tfrecord_files(save_path, f"{dataset_name}.tfrecord")


def write_to_directory_single_worker(
    convert_func: Callable,
    postprocess_func: Callable | None,
    list_scenarios: list[Any],
    output_path: str,
    dataset_version: str,
    dataset_name: str,
    worker_index: int = 0,
    preprocess: Callable = single_worker_preprocess,
    worker_kwargs: dict | None = None,
    **kwargs,
) -> bool:
    """
    Convert a batch of scenarios using a single worker.

    Args:
        convert_func: Function to convert scenarios
        postprocess_func: Optional post-processing function
        scenarios: List or generator of scenarios to process
        output_path: Path to output directory
        dataset_version: Dataset version
        dataset_name: Dataset name
        worker_index: Worker index for parallel processing
        preprocess: Function for preprocessing scenarios
        worker_kwargs: Worker-specific keyword arguments
        **kwargs: Additional keyword arguments

    Returns:
        True if processing was successful, False otherwise
    """
    if worker_kwargs is None:
        worker_kwargs = {}

    kwargs.update(worker_kwargs)

    if not contains_explicit_return(convert_func):
        err_msg = "The convert function should return a metadata dict"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("The specified version in kwargs is replaced by argument: 'dataset_version'")

    memory_before = process_memory()
    logger.info(f"Worker {worker_index} starting - Memory usage: {memory_before:.2f} MB")

    # Apply preprocessing
    scenarios = preprocess(list_scenarios, worker_index)

    if dataset_name == "womd":
        from scenariomax.raw_to_unified.converter.waymo.load import count_waymo_scenarios

        num_scenarios = count_waymo_scenarios(list_scenarios)
    else:
        num_scenarios = len(scenarios)

    memory_after = process_memory()
    logger.debug(
        f"Worker {worker_index} preprocessing complete - Memory: {memory_after:.2f} MB "
        f"(delta: {memory_after - memory_before:.2f} MB)",
    )

    # Set up progress tracking
    logger.info(f"Worker {worker_index} processing {num_scenarios} scenarios")

    pbar = tqdm(desc=f"Worker {worker_index}", unit=" scenario", total=num_scenarios)

    processed_count = 0
    error_count = 0

    try:
        if not postprocess_func:
            logger.info(f"Worker {worker_index} starting scenario processing (standard mode)")

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
                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_count, "errors": error_count})
        else:
            logger.info(f"Worker {worker_index} starting scenario processing (postprocess mode)")
            postprocess_func(
                output_path,
                worker_index,
                scenarios,
                convert_func,
                dataset_version,
                dataset_name,
                pbar,
                **kwargs,
            )

        memory_final = process_memory()

        # Detailed completion logging
        logger.info(f"Worker {worker_index} COMPLETED:")
        logger.info(f"  ✅ Processed: {processed_count} scenarios")
        logger.info(f"  ❌ Errors: {error_count} scenarios")
        logger.info(f"  📊 Memory: {memory_final:.2f} MB")
        logger.info(f"  📁 Output: {output_path}")

        return True
    except Exception as e:
        logger.error(f"Worker {worker_index} encountered an error: {e!s}")
        return False
    finally:
        pbar.close()


def process_scenario(
    scenario: Any,
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    **kwargs,
) -> tuple:
    """
    Process a single scenario and convert it to the desired format.

    Args:
        scenario: The scenario to be processed.
        convert_func: The function used to convert the scenario.
        dataset_version: The version of the dataset.
        dataset_name: The name of the dataset.
        **kwargs: Additional arguments for the conversion function.

    Returns:
        tuple: (sd_scenario, export_file_name)
            - sd_scenario: The converted scenario.
            - export_file_name: The name of the file where the scenario will be saved.
    """
    sd_scenario = convert_func(scenario, dataset_version, **kwargs)
    scenario_id = sd_scenario[SD.ID]
    export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

    sd_scenario = sd_scenario.to_dict()
    SD.sanity_check(sd_scenario, check_self_type=True)

    return sd_scenario, export_file_name


def postprocess_tfexample(
    output_path: str,
    worker_index: int,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    pbar: tqdm,
    **kwargs,
) -> None:
    """
    Process scenarios and write them as TFExample records.

    Args:
        output_path: Path to output directory
        worker_index: Worker index for parallel processing
        scenarios: List or generator of scenarios to process
        convert_func: Function to convert scenarios
        dataset_version: Dataset version
        dataset_name: Dataset name
        pbar: Progress bar instance
        **kwargs: Additional keyword arguments
    """
    processed_count = 0
    error_count = 0
    filtered_count = 0
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
                    pbar.set_postfix({"processed": processed_count, "filtered": filtered_count, "errors": error_count})
            except (OverpassException, NotEnoughValidObjectsException):
                filtered_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count, "filtered": filtered_count, "errors": error_count})
            except Exception as e:
                error_count += 1
                logger.error(f"Worker {worker_index} failed to process scenario: {e!s}")
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count, "filtered": filtered_count, "errors": error_count})


def postprocess_gpudrive(
    output_path: str,
    worker_index: int,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    pbar: tqdm,
    **kwargs,
) -> None:
    """
    Process scenarios and write them as GPU Drive JSON files.

    Args:
        output_path: Path to output directory
        worker_index: Worker index for parallel processing
        scenarios: List or generator of scenarios to process
        convert_func: Function to convert scenarios
        dataset_version: Dataset version
        dataset_name: Dataset name
        pbar: Progress bar instance
        **kwargs: Additional keyword arguments
    """
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
                **kwargs,
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
            logger.exception(f"Worker {worker_index} failed to process scenario")
            pbar.update(1)
            pbar.set_postfix({"processed": processed_count, "errors": error_count})


def merge_tfrecord_files(output_dir: str, merged_filename: str) -> None:
    """
    Merge TFRecord files from multiple directories into a single file and clean up.

    Args:
        output_dir: Directory containing subdirectories with TFRecord files
        merged_filename: Name for the output merged file
    """
    tfrecord_files = []

    try:
        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                list_dir = os.listdir(dir_path)
                tfrecord_files += [os.path.join(dir_path, f) for f in list_dir if f.endswith(".tfrecord")]

        logger.info(f"Found {len(tfrecord_files)} TFRecord files to merge")

        # Define the path for the merged TFRecord file
        merged_file_path = os.path.join(output_dir, merged_filename)
        logger.info(f"Merging files into: {merged_file_path}")

        total_records = 0
        with tf.io.TFRecordWriter(merged_file_path) as writer:
            for tfrecord_file in tqdm(tfrecord_files, desc="Merging TFRecord files"):
                try:
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
                except Exception as e:
                    logger.error(f"Error processing file {tfrecord_file}: {e!s}")

        logger.info(f"Shuffling merged file with {total_records} records")
        shuffle_tfrecord_file(merged_file_path)

        logger.info(f"Successfully merged and shuffled TFRecord file at {merged_file_path}")
    except Exception as e:
        logger.error(f"Error during TFRecord merging: {e!s}")
        raise


def merge_json_files(output_dir: str) -> None:
    """
    Merge JSON files from subdirectories into the main output directory.

    Args:
        output_dir: Directory containing subdirectories with JSON files
    """
    json_files = []

    try:
        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                list_dir = os.listdir(dir_path)
                json_files += [os.path.join(dir_path, f) for f in list_dir if f.endswith(".json")]

        logger.info(f"Found {len(json_files)} JSON files to merge")
        logger.info(f"Merging files into: {output_dir}")

        for file in json_files:
            shutil.move(file, output_dir)

        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

        logger.info(f"All json files moved to {output_dir} and subdirs deleted.")
    except Exception as e:
        logger.error(f"Error during JSON file merging: {e!s}")
        raise


def shuffle_tfrecord_file(tfrecord_file: str, buffer_size: int = 10000) -> None:
    """
    Shuffle a TFRecord file to improve training data randomization.

    Args:
        tfrecord_file: Path to the TFRecord file
        buffer_size: Buffer size for shuffling
    """
    # Create a temporary file to store shuffled records
    temp_file = tfrecord_file + ".shuffled"
    logger.debug(f"Creating temporary shuffled file: {temp_file}")

    try:
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
    except Exception as e:
        logger.error(f"Error during shuffling: {e!s}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

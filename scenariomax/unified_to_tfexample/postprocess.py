import os
from collections.abc import Callable, Generator, Iterable
from typing import Any

from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.tf_utils import get_tensorflow
from scenariomax.unified_to_tfexample import convert_to_tfexample, exceptions


logger = logger_utils.get_logger(__name__)


def postprocess_tfexample(
    output_path: str,
    worker_index: int,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    pbar: tqdm,
    process_scenario_func: Callable,
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
        process_scenario_func: Function to process individual scenarios
        **kwargs: Additional keyword arguments
    """
    processed_count = 0
    filtered_count = 0
    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()
    tf_record_file = os.path.join(output_path, "training.tfrecord")
    logger.debug(f"Worker {worker_index} writing to TFRecord: {tf_record_file}")

    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for scenario in scenarios:
            try:
                unified_scenario = process_scenario_func(
                    scenario,
                    convert_func,
                    dataset_version,
                    dataset_name,
                    **kwargs,
                )

                dict_to_convert = convert_to_tfexample.convert(unified_scenario)
                example = tf.train.Example(features=tf.train.Features(feature=dict_to_convert))

                if example is not None:
                    writer.write(example.SerializeToString())
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_count, "filtered": filtered_count})
            except (exceptions.OverpassException, exceptions.NotEnoughValidObjectsException):
                filtered_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count, "filtered": filtered_count})
            except Exception as e:
                logger.error(f"Worker {worker_index} failed to process scenario: {e!s}")
                pbar.close()
                raise e


def merge_dataset_workers(dataset_dir: str, dataset_name: str) -> str:
    """
    Merge TFRecord files from worker subdirectories into a single dataset file.

    Args:
        dataset_dir: Directory containing worker subdirectories with TFRecord files
        dataset_name: Name of the dataset for output filename

    Returns:
        Path to the merged dataset TFRecord file
    """
    import shutil

    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()
    tfrecord_files = []

    # Look for worker subdirectories and their TFRecord files
    basename = os.path.basename(dataset_dir)
    logger.info(f"Merging {dataset_name} workers from: {dataset_dir}")

    for item in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, item)
        # Look for directories that match the worker pattern: {basename}_{worker_index}
        if os.path.isdir(dir_path) and item.startswith(f"{basename}_"):
            list_dir = os.listdir(dir_path)
            worker_tfrecord_files = [os.path.join(dir_path, f) for f in list_dir if f.endswith(".tfrecord")]
            tfrecord_files.extend(worker_tfrecord_files)
            logger.debug(f"Found {len(worker_tfrecord_files)} TFRecord files in worker dir {dir_path}")

    logger.info(f"Found {len(tfrecord_files)} worker TFRecord files for {dataset_name}")

    if not tfrecord_files:
        raise RuntimeError(f"No TFRecord files found for dataset {dataset_name} in {dataset_dir}")

    # Create dataset-specific merged file in parent directory
    parent_dir = os.path.dirname(dataset_dir)
    dataset_tfrecord_path = os.path.join(parent_dir, f"{dataset_name}.tfrecord")
    logger.info(f"Merging {dataset_name} workers into: {dataset_tfrecord_path}")

    total_records = 0
    dirs_to_remove = set()

    with tf.io.TFRecordWriter(dataset_tfrecord_path) as writer:
        for tfrecord_file in tqdm(tfrecord_files, desc=f"Merging {dataset_name} workers"):
            try:
                # Read the current TFRecord file
                dataset = tf.data.TFRecordDataset(tfrecord_file)

                file_records = 0
                for record in dataset:
                    writer.write(record.numpy())
                    file_records += 1

                total_records += file_records
                logger.debug(f"Merged {file_records} records from {tfrecord_file}")

                # Track directory for later removal
                dirs_to_remove.add(os.path.dirname(tfrecord_file))
            except Exception as e:
                logger.error(f"Error processing file {tfrecord_file}: {e!s}")
                raise

    # Remove worker directories after all files are processed
    for dir_to_remove in dirs_to_remove:
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)
            logger.debug(f"Removed worker directory: {dir_to_remove}")

    # Remove the empty dataset directory
    if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
        shutil.rmtree(dataset_dir)
        logger.debug(f"Removed empty dataset directory: {dataset_dir}")

    logger.info(f"Successfully merged {dataset_name}: {total_records} records → {dataset_tfrecord_path}")
    return dataset_tfrecord_path


def merge_multiple_datasets(output_dir: str, merged_filename: str) -> None:
    """
    Merge multiple dataset TFRecord files into a single final file and shuffle.

    Args:
        output_dir: Directory containing dataset-specific TFRecord files
        merged_filename: Name for the final merged file
    """
    import shutil

    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()

    # Look for dataset TFRecord files (not in subdirectories)
    dataset_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".tfrecord") and os.path.isfile(os.path.join(output_dir, f))
    ]

    logger.info(f"Found {len(dataset_files)} dataset files to merge: {[os.path.basename(f) for f in dataset_files]}")

    if not dataset_files:
        logger.warning("No dataset TFRecord files found to merge")
        return

    if len(dataset_files) == 1:
        # Single dataset - just rename it
        single_file = dataset_files[0]
        final_path = os.path.join(output_dir, merged_filename)
        shutil.move(single_file, final_path)
        logger.debug(f"Renamed single dataset file to: {final_path}")
        shuffle_tfrecord_file(final_path)
        return

    # Multiple datasets - merge and shuffle
    final_path = os.path.join(output_dir, merged_filename)
    logger.info(f"Merging {len(dataset_files)} datasets into: {final_path}")

    total_records = 0
    with tf.io.TFRecordWriter(final_path) as writer:
        for dataset_file in tqdm(dataset_files, desc="Merging datasets"):
            try:
                dataset = tf.data.TFRecordDataset(dataset_file)
                file_records = 0
                for record in dataset:
                    writer.write(record.numpy())
                    file_records += 1

                total_records += file_records
                logger.debug(f"Merged {file_records} records from {os.path.basename(dataset_file)}")

                # Remove the individual dataset file
                os.remove(dataset_file)

            except Exception as e:
                logger.error(f"Error processing dataset file {dataset_file}: {e!s}")
                raise

    logger.info(f"Shuffling final merged file with {total_records} records")
    shuffle_tfrecord_file(final_path)
    logger.info(f"Successfully created final dataset: {final_path}")


def merge_tfrecord_files(output_dir: str, merged_filename: str) -> None:
    """
    Merge TFRecord files from multiple directories into a single file and clean up.

    Args:
        output_dir: Directory containing subdirectories with TFRecord files
        merged_filename: Name for the output merged file
    """
    import shutil

    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()
    tfrecord_files = []

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


def shuffle_tfrecord_file(tfrecord_file: str, buffer_size: int = 10000) -> None:
    """
    Shuffle a TFRecord file to improve training data randomization.

    Args:
        tfrecord_file: Path to the TFRecord file
        buffer_size: Buffer size for shuffling
    """
    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()

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

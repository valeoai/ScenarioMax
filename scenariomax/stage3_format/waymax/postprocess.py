import os

from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.tf_utils import get_tensorflow


logger = logger_utils.get_logger(__name__)


def merge_tfrecord_files(tfrecord_files: list, merged_file_path: str) -> None:
    """
    Merge TFRecord files from multiple directories into a single file and clean up.

    Optimized approach:
    - Streams records from each file (no memory loading)
    - Deletes individual files immediately after reading (reduces disk usage)
    - Single-pass merge (no redundant I/O)

    Args:
        tfrecord_files: List of TFRecord file paths to merge
        merged_file_path: Path for the output merged file
    """
    import shutil

    # Get TensorFlow with optimized configuration
    tf = get_tensorflow()

    logger.info(f"Found {len(tfrecord_files)} TFRecord files to merge")

    # Define the path for the merged TFRecord file
    logger.info(f"Merging files into: {merged_file_path}")

    total_records = 0
    dirs_to_remove = set()

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

                # Track directory for later removal
                dir_to_remove = os.path.dirname(tfrecord_file)
                dirs_to_remove.add(dir_to_remove)

                # Delete the individual file immediately after reading to free disk space
                try:
                    os.remove(tfrecord_file)
                    logger.debug(f"Deleted merged file: {tfrecord_file}")
                except OSError as e:
                    logger.warning(f"Could not delete file {tfrecord_file}: {e!s}")
            except Exception as e:
                logger.error(f"Error processing file {tfrecord_file}: {e!s}")

    # Remove empty directories after all files are processed
    for dir_to_remove in dirs_to_remove:
        if os.path.exists(dir_to_remove):
            try:
                # Only remove if directory is empty (all files were successfully deleted)
                if not os.listdir(dir_to_remove):
                    shutil.rmtree(dir_to_remove)
                    logger.debug(f"Removed empty directory: {dir_to_remove}")
                else:
                    logger.warning(f"Directory not empty, skipping removal: {dir_to_remove}")
            except Exception as e:
                logger.warning(f"Could not remove directory {dir_to_remove}: {e!s}")

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

import math
import os

import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.logger_utils import get_logger


logger = get_logger(__name__)


def process_shard(shard_index: int, src_path: str, format_str: str, num_shards: int):
    """Process and write a single shard from the source TFRecord file.

    Args:
        shard_index: Index of the shard to process
        src_path: Path to the source TFRecord file
        format_str: Format string for output shard filenames
        num_shards: Total number of shards
    """
    file_path = format_str % (shard_index, num_shards)
    raw_dataset = tf.data.TFRecordDataset(src_path)
    shard_dataset = raw_dataset.shard(num_shards, shard_index).prefetch(tf.data.experimental.AUTOTUNE)

    # Write the shard to a file
    with tf.io.TFRecordWriter(file_path) as writer:
        for raw_record in shard_dataset:
            writer.write(raw_record.numpy())


def shard_tfrecord(src: str, filename: str, num_threads: int, num_shards: int = 1000, delete_original: bool = True):
    """Shard a TFRecord file into multiple smaller files for parallel processing.

    Args:
        src: Source directory containing the TFRecord file
        filename: Name of the TFRecord file (without extension)
        num_threads: Number of parallel threads to use
        num_shards: Number of output shards to create
        delete_original: Whether to delete the original file after sharding
    """
    logger.info(f"Sharding TFRecord {filename} into {num_shards} shards")

    src_path = os.path.join(src, f"{filename}.tfrecord")

    # Check if the file exists
    if not os.path.exists(src_path):
        logger.error(f"Source file {src_path} does not exist")
        return

    # Calculate the width needed for shard numbering
    shard_width = max(5, int(math.log10(num_shards) + 1))
    format_str = src_path + "-%0" + str(shard_width) + "d-of-%05d"

    # Process shards in parallel using thread-based parallelism for I/O operations
    logger.info(f"Starting parallel sharding with {num_threads} threads")
    with Parallel(n_jobs=num_threads, prefer="threads", verbose=5) as parallel:
        parallel(
            delayed(process_shard)(i, src_path, format_str, num_shards)
            for i in tqdm(range(num_shards), desc="Sharding TFRecord")
        )

    logger.info(f"Sharding completed. Shards saved to {src_path}-00000-of-{num_shards:05d}")

    if delete_original:
        try:
            os.remove(src_path)
            logger.info(f"Deleted original TFRecord file: {src_path}")
        except Exception as e:
            logger.error(f"Failed to delete original file: {e!s}")


if __name__ == "__main__":
    shard_tfrecord(
        src="output",
        filename="training",
        num_threads=50,
        num_shards=10000,
        delete_original=True,
    )

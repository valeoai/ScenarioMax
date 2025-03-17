import math
import os

import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.logger_utils import get_logger


logger = get_logger(__name__)


def process_shard(shard_index: int, src_path: str, format_str: str, num_shards: int):
    file_path = format_str % (shard_index, num_shards)
    raw_dataset = tf.data.TFRecordDataset(src_path)
    shard_dataset = raw_dataset.shard(num_shards, shard_index).prefetch(tf.data.experimental.AUTOTUNE)

    # Write the shard to a file
    with tf.io.TFRecordWriter(file_path) as writer:
        for raw_record in shard_dataset:
            writer.write(raw_record.numpy())


def shard_tfrecord(src: str, filename: str, num_threads: int, num_shards: int = 1000, delete_original: bool = True):
    os.makedirs(src + "/shard", exist_ok=True)

    src_path = os.path.join(src, f"{filename}.tfrecord")
    output_path = os.path.join(src, f"shard/{filename}.tfrecord")

    shard_width = max(5, int(math.log10(num_shards) + 1))
    format_str = output_path + "-%0" + str(shard_width) + "d-of-%05d"

    with Parallel(n_jobs=num_threads) as parallel:
        parallel(
            delayed(process_shard)(i, src_path, format_str, num_shards)
            for i in tqdm(range(num_shards), desc="Sharding TFRecord")
        )  # noqa: E501

    if delete_original:
        os.remove(src_path)


if __name__ == "__main__":
    shard_tfrecord(
        src="output",
        filename="training",
        num_threads=50,
        num_shards=10000,
        delete_original=True,
    )

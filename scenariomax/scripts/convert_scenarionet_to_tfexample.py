import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.unified_to_tfexample.build_tfexample import build_tfexample
from scenariomax.unified_to_tfexample.utils import list_files


def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Scenarionet data to TFRecord")
    parser.add_argument(
        "-d",
        type=str,
        default="nuplan",
        help="Choose the data to convert to TFRecord",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to convert",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
        help="Number of parallel jobs",
    )
    return parser.parse_args()


def process_scenario(path_to_scenario: str, debug: bool = False) -> tf.train.Example | None:
    try:
        scenario_to_convert = pd.read_pickle(path_to_scenario)
    except Exception as e:
        print(f"Error reading pickle file {path_to_scenario}: {e}")
        return None

    try:
        dict_to_convert = build_tfexample(scenario_to_convert, debug)
    except Exception as e:
        print(f"Error converting scenario to TFExample: {e}")
        return None

    if dict_to_convert is None:
        print(f"Error in scenario: {path_to_scenario}")
        return None

    return tf.train.Example(features=tf.train.Features(feature=dict_to_convert))


def process_and_save_scenario_chunk(pkle_files_chunk: list[str], output_dir: str, chunk_index: int):
    tf_record_file = os.path.join(output_dir, f"training_{chunk_index}.tfrecord")

    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for path_to_scenario in tqdm(pkle_files_chunk, desc=f"Chunk {chunk_index}"):
            example = process_scenario(path_to_scenario)
            if example is not None:
                writer.write(example.SerializeToString())

    print(f"Chunk {chunk_index} saved to {tf_record_file}")


def write_tfrecord_from_scenariomax_files(
    pkle_files: list[str], data_choice: str, num_files: int = None, n_jobs: int = -1
) -> None:
    output_dir = f"/data/tfrecord/{data_choice}"
    os.makedirs(output_dir, exist_ok=True)

    # Remove the first two files (dataset_summary.pkl and dataset_mapping.pkl)
    pkle_files = [f for f in pkle_files if not f.endswith(("dataset_summary.pkl", "dataset_mapping.pkl"))]

    if num_files:
        pkle_files = pkle_files[:num_files]

    num_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    # Divide the list into chunks
    pkle_files_chunks = np.array_split(pkle_files, num_workers)

    print(f"Converting {len(pkle_files)} files to TFRecord format using {num_workers} workers")
    print(f"Size chunks: {[len(chunk) for chunk in pkle_files_chunks]}")
    print(f"Output directory: {output_dir}")

    with Parallel(n_jobs=num_workers) as parallel:
        parallel(
            delayed(process_and_save_scenario_chunk)(pkle_files_chunk, output_dir, i)
            for i, pkle_files_chunk in enumerate(pkle_files_chunks)
        )

    print(f"TFRecord files created in {output_dir}")

    merge_and_clean_tfrecord_files(output_dir, "training.tfrecord")


def merge_and_clean_tfrecord_files(output_dir: str, merged_filename: str) -> None:
    # List all TFRecord files in the output directory
    tfrecord_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".tfrecord")]

    # Define the path for the merged TFRecord file
    merged_file_path = os.path.join(output_dir, merged_filename)

    with tf.io.TFRecordWriter(merged_file_path) as writer:
        for tfrecord_file in tqdm(tfrecord_files, desc="Merging TFRecord files"):
            # Read the current TFRecord file
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            for record in dataset:
                writer.write(record.numpy())

            # Delete the current TFRecord file
            os.remove(tfrecord_file)

    print(f"Merged TFRecord file created at {merged_file_path} and individual files deleted.")


if __name__ == "__main__":
    args = arg_parser()

    data_choice = args.d
    data_location = "/data/out"
    pkle_files = list_files(data_location)

    write_tfrecord_from_scenariomax_files(pkle_files, data_choice, args.num_files, args.n_jobs)

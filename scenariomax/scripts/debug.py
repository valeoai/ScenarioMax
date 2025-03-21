import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from scenariomax.unified_to_tfexample.build_tfexample import build_tfexample
from scenariomax.unified_to_tfexample.utils import list_files


def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Scenarionet data to tfrecord")
    parser.add_argument(
        "--src",
        type=str,
        default="",
        help="Choose the data to convert to tfrecord",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="",
        help="The directory to save the converted data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="The dataset to convert",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=100,
        help="Number of files to convert",
    )

    return parser.parse_args()


def process_scenario(path_to_scenario: str, debug: bool = False) -> tf.train.Example | None:
    try:
        scenario_to_convert = pd.read_pickle(path_to_scenario)
    except Exception as e:
        print(f"Error reading pickle file {path_to_scenario}: {e}")
        return None

    try:
        dict_to_convert = build_tfexample(scenario_to_convert, False, debug)
    except Exception as e:
        print(f"Error building TF Example for {path_to_scenario}: {e}")
        return None

    return tf.train.Example(features=tf.train.Features(feature=dict_to_convert))


def write_tfrecord_from_scenariomax_files(
    pkle_files: list[str],
    output_dir: str,
    dataset: str,
    num_files: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tf_record_file = f"{output_dir}/training.tfrecord"

    # Remove the first two files (dataset_summary.pkl and dataset_mapping.pkl)
    pkle_files = [f for f in pkle_files if not f.endswith(("dataset_summary.pkl", "dataset_mapping.pkl"))]

    if num_files:
        pkle_files = pkle_files[:num_files]

    total_files = len(pkle_files)

    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for i, path_to_scenario in enumerate(pkle_files):
            idx = i + 1

            print(f"\nProcessing scenario {idx} / {total_files}")

            example = process_scenario(path_to_scenario, debug=True)

            ax = plt.gca()
            fig = plt.gcf()

            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()
            size_x = x_ticks[-1] - x_ticks[0]
            size_y = y_ticks[-1] - y_ticks[0]
            print("size_x: ", size_x)
            print("size_y: ", size_y)

            fig.set_figwidth(size_x / 10)
            fig.set_figheight(size_y / 10)

            os.makedirs(f"debug/{dataset}/png", exist_ok=True)
            if example is not None:
                writer.write(example.SerializeToString())
                plt.savefig(f"debug/{dataset}/png/scenario_{idx}.png")
            # else:
            # plt.savefig(f"debug/{data_choice}/png/scenario_{idx}_overpass.png")

            plt.close()

    print(f"TFRecord file created at {tf_record_file}")


if __name__ == "__main__":
    args = arg_parser()

    pkle_files = list_files(args.src)
    num_files = args.num_files if args.num_files > 0 else None

    print(f"Found {len(pkle_files)} files in {args.src}")

    write_tfrecord_from_scenariomax_files(pkle_files, args.dst, args.dataset, args.num_files)

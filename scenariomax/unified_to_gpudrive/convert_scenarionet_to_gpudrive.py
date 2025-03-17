import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax.unified_to_gpudrive import utils
from scenariomax.unified_to_gpudrive.build_gpudrive_example import build_gpudrive_example


def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Scenarionet data to TFRecord")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Root directory of data to convert",
    )
    parser.add_argument("--dataset", type=str, help="Name of dataset to convert i.e. Nuplan etc.")
    parser.add_argument("--split", type=str, help="Split of dataset to convert, i.e. train, test, mini etc.")
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
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to write files. Will write to output_dir/dataset/split/",
    )
    return parser.parse_args()


def process_scenario(path_to_scenario: str, debug: bool = False) -> dict | None:
    try:
        scenario_to_convert = pd.read_pickle(path_to_scenario)
    except Exception as e:
        print(f"Error reading pickle file {path_to_scenario}: {e}")
        return None

    try:
        dict_to_convert = build_gpudrive_example(path_to_scenario.split("/")[-1], scenario_to_convert, debug)
    except Exception as e:
        print(f"Error converting scenario to GPUDrive format: {e}")
        return None

    if dict_to_convert is None:
        print(f"Error in scenario: {path_to_scenario}")
        return None

    return dict_to_convert


def process_and_save_scenario_chunk(pkle_files_chunk: list[str], output_dir: str, file_prefix: str, chunk_index: int):
    for path_to_scenario in tqdm(pkle_files_chunk, desc=f"Chunk {chunk_index}"):
        example = process_scenario(path_to_scenario)
        gpudrive_json_file = os.path.join(output_dir, f"{file_prefix}_{example['scenario_id']}.json")
        if example is not None:
            with open(gpudrive_json_file, "w") as f:
                json.dump(utils.convert_numpy(example), f)

    print(f"Chunk {chunk_index} saved to {gpudrive_json_file}")


def write_gpudrive_from_scenariomax_files(
    pkle_files: list[str],
    output_dir: str,
    file_prefix: str,
    num_files: int = None,
    n_jobs: int = -1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Remove the first two files (dataset_summary.pkl and dataset_mapping.pkl)
    pkle_files = [f for f in pkle_files if not f.endswith(("dataset_summary.pkl", "dataset_mapping.pkl"))]

    if num_files:
        pkle_files = pkle_files[:num_files]

    num_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    # Divide the list into chunks
    pkle_files_chunks = np.array_split(pkle_files, num_workers)

    print(f"Converting {len(pkle_files)} files to GPUDrive format using {num_workers} workers")
    print(f"Size chunks: {[len(chunk) for chunk in pkle_files_chunks]}")
    print(f"Output directory: {output_dir}")

    with Parallel(n_jobs=num_workers) as parallel:
        parallel(
            delayed(process_and_save_scenario_chunk)(pkle_files_chunk, output_dir, file_prefix, i)
            for i, pkle_files_chunk in enumerate(pkle_files_chunks)
        )

    print(f"GPUDrive files created in {output_dir}")


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(".pkl"):
                r.append(os.path.join(root, name))
    return r


if __name__ == "__main__":
    args = arg_parser()

    input_dir = args.input_dir
    dataset = args.dataset
    split = args.split
    output_dir = os.path.join(args.output_dir, dataset, split)
    file_prefix = f"{dataset}_{split}"
    pkle_files = list_files(input_dir)

    write_gpudrive_from_scenariomax_files(pkle_files, output_dir, file_prefix, args.num_files, args.n_jobs)

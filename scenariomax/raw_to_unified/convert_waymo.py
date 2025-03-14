import argparse
import os

from scenariomax.raw_to_unified.converter.waymo import (
    convert_waymo_scenario,
    get_waymo_scenarios,
    preprocess_waymo_scenarios,
)
from scenariomax.raw_to_unified.converter.write import write_to_directory


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser(description="Build database from Waymo scenarios")
    parser.add_argument(
        "--src",
        default="",
        help="The directory stores all waymo tfrecord",
    )
    parser.add_argument(
        "--dst",
        default="",
        help="A directory, the path to place the converted data",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers to use",
    )
    parser.add_argument(
        "--num-files",
        default=None,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]. Default: None, will read all files.",
    )
    parser.add_argument(
        "--start_file_index",
        default=0,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]. Default: 0.",
    )
    parser.add_argument(
        "--write-pickle",
        action="store_true",
        help="Write the converted data to pickle file",
    )
    args = parser.parse_args()

    output_path = args.dst

    files = get_waymo_scenarios(args.src, args.start_file_index, args.num_files)

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=files,
        output_path=output_path,
        dataset_name="womd",
        dataset_version="v1.3",
        num_workers=args.num_workers,
        preprocess=preprocess_waymo_scenarios,
        write_pickle=args.write_pickle,
    )

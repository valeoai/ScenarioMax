import argparse
import logging
import os
import shutil

from scenariomax.raw_to_unified.converter.argoverse2.utils import (
    convert_av2_scenario,
    get_av2_scenarios,
    preprocess_av2_scenarios,
)
from scenariomax.raw_to_unified.converter.write import write_to_directory


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Build database from Argoverse scenarios")
    parser.add_argument(
        "--src",
        type=str,
        default="",
        help="the place store .db files",
    )
    parser.add_argument(
        "--dst",
        default="",
        help="A directory, the path to place the data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers to use",
    )
    parser.add_argument(
        "--num_files",
        default=1000,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]",
    )
    parser.add_argument(
        "--start_file_index",
        default=0,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]",
    )
    parser.add_argument(
        "--write-pickle",
        action="store_true",
        help="Write the converted data to pickle file",
    )
    args = parser.parse_args()

    output_path = args.dst

    save_path = output_path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    av2_data_directory = args.src

    scenarios = get_av2_scenarios(av2_data_directory, args.start_file_index, args.num_files)

    write_to_directory(
        convert_func=convert_av2_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version="v2",
        dataset_name="argoverse",
        num_workers=args.num_workers,
        preprocess=preprocess_av2_scenarios,
        write_pickle=args.write_pickle,
    )

import argparse
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from scenariomax.raw_to_unified.converter.nuplan import convert_nuplan_scenario, get_nuplan_scenarios
from scenariomax.raw_to_unified.converter.write import write_to_directory
from scenariomax.raw_to_unified.utils import setup_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build database from nuPlan scenarios")
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
        "--num-workers",
        type=int,
        default=8,
        help="number of workers to use",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="for test use only. convert one log",
    )
    args = parser.parse_args()

    setup_logging()

    output_path = args.dst

    data_root = args.src
    map_root = os.getenv("NUPLAN_MAPS_ROOT")
    if args.test:
        scenarios = get_nuplan_scenarios(data_root, map_root, logs=["2021.07.16.20.45.29_veh-35_01095_01486"])
    else:
        scenarios = get_nuplan_scenarios(data_root, map_root)

    write_to_directory(
        convert_func=convert_nuplan_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version="v1.1",
        dataset_name="nuplan",
        num_workers=args.num_workers,
    )

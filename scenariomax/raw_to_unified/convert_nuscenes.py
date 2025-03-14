import argparse

from scenariomax.raw_to_unified.converter.nuscenes import (
    convert_nuscenes_scenario,
    get_nuscenes_prediction_split,
    get_nuscenes_scenarios,
)
from scenariomax.raw_to_unified.converter.write import write_to_directory
from scenariomax.unified_to_tfrecord.constants import NUM_TS_FUTURE, NUM_TS_PAST


if __name__ == "__main__":
    prediction_split = ["mini_train", "mini_val", "train", "train_val", "val"]
    scene_split = ["v1.0-mini", "v1.0-trainval", "v1.0-test"]

    parser = argparse.ArgumentParser(description="Build database from nuScenes/Lyft scenarios")
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
        "--split",
        default="v1.0-trainval",
        choices=scene_split + prediction_split,
        help=f"Which splits of nuScenes data should be sued. If set to {scene_split}, it will convert the full log into scenarios"
        f" with 20 second episode length. If set to {prediction_split}, it will convert segments used for nuScenes prediction"
        " challenge to scenarios, resulting in more converted scenarios. Generally, you should choose this "
        f" parameter from {scene_split} to get complete scenarios for planning unless you want to use the converted scenario "
        " files for prediction task.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers to use",
    )
    parser.add_argument(
        "--write-pickle",
        action="store_true",
        help="Write the converted data to pickle file",
    )
    args = parser.parse_args()

    map_radius = 300
    past = NUM_TS_PAST / 10
    future = NUM_TS_FUTURE / 10

    output_path = args.dst
    version = args.split

    if version in scene_split:
        scenarios, nuscs = get_nuscenes_scenarios(args.src, version, args.num_workers)
    else:
        scenarios, nuscs = get_nuscenes_prediction_split(
            args.src,
            version,
            args.past,
            args.future,
            args.num_workers,
        )
    write_to_directory(
        convert_func=convert_nuscenes_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name="nuscenes",
        num_workers=args.num_workers,
        nuscenes=nuscs,
        prediction=[version in prediction_split for _ in range(args.num_workers)],
        write_pickle=args.write_pickle,
    )

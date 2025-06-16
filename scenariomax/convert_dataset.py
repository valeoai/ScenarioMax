import argparse
import logging
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from scenariomax.logger_utils import get_logger, setup_logger
from scenariomax.raw_to_unified.write import (
    merge_tfrecord_files,
    postprocess_gpudrive,
    postprocess_tfexample,
    write_to_directory,
)
from scenariomax.unified_to_tfexample.shard_tfexample import shard_tfrecord


logger = get_logger(__name__)


def convert_dataset(args, dataset):
    logger.info(f"Starting conversion for dataset: {dataset}")

    if args.target_format == "tfexample":
        postprocess_func = postprocess_tfexample
    elif args.target_format == "gpudrive":
        postprocess_func = postprocess_gpudrive
    elif args.target_format == "pickle":
        postprocess_func = None
    else:
        raise ValueError(f"Unsupported target format: {args.target_format}")

    if dataset == "waymo":
        from scenariomax.raw_to_unified.converter import waymo

        logger.info(f"Loading Waymo scenarios from {args.waymo_src}")
        scenarios = waymo.get_waymo_scenarios(args.waymo_src, num=args.num_files)

        preprocess_func = waymo.preprocess_waymo_scenarios
        convert_func = waymo.convert_waymo_scenario

        dataset_name = "womd"
        dataset_version = "v1.3"
        additional_args = {}
    elif dataset == "nuscenes":
        from scenariomax.raw_to_unified.converter import nuscenes

        logger.info(f"Loading nuScenes scenarios from {args.nuscenes_src} (split: {args.split})")
        scenarios, nuscs = nuscenes.get_nuscenes_scenarios(args.nuscenes_src, args.split, args.num_workers)

        preprocess_func = None
        convert_func = nuscenes.convert_nuscenes_scenario

        dataset_name = "nuscenes"
        dataset_version = "v1.0"
        additional_args = {"nuscs": nuscs}
    elif dataset == "argoverse2":
        from scenariomax.raw_to_unified.converter.argoverse2.utils import (
            convert_av2_scenario,
            get_av2_scenarios,
            preprocess_av2_scenarios,
        )

        logger.info(f"Loading Argoverse2 scenarios from {args.argoverse2_src}")
        scenarios = get_av2_scenarios(args.argoverse2_src)

        postprocess_func = preprocess_av2_scenarios
        convert_func = convert_av2_scenario

        dataset_name = "argoverse"
        dataset_version = "v2.0"
        additional_args = {}
    elif dataset == "nuplan":
        from scenariomax.raw_to_unified.converter import nuplan

        logger.info(f"Loading NuPlan scenarios from {args.nuplan_src} (maps: {os.getenv('NUPLAN_MAPS_ROOT', 'N/A')})")
        scenarios = nuplan.get_nuplan_scenarios(
            args.nuplan_src,
            os.getenv("NUPLAN_MAPS_ROOT"),
            num_files=args.num_files,
        )

        preprocess_func = None
        convert_func = nuplan.convert_nuplan_scenario

        dataset_name = "nuplan"
        dataset_version = "v1.1"
        additional_args = {}
    elif dataset == "openscenes":
        from scenariomax.raw_to_unified.converter import openscenes
        logger.info(f"Loading OpenScenes scenarios from {args.openscenes_metadata_src} (maps: {os.getenv('NUPLAN_MAPS_ROOT', 'N/A')})")
        scenarios = openscenes.get_nuplan_scenarios(
            os.getenv("NUPLAN_DATA_ROOT"),
            args.nuplan_src,
            os.getenv("NUPLAN_MAPS_ROOT"),
            args.openscenes_metadata_src,
            num_files=args.num_files,
        )

        preprocess_func = None
        convert_func = openscenes.convert_nuplan_scenario

        dataset_name = "openscenes"
        dataset_version = "v1.1"
        additional_args = {}
    else:
        err_msg = f"Unsupported dataset: {dataset}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"Loaded {len(scenarios)} scenarios from {dataset}")

    output_path = args.dst + f"/{dataset}"
    logger.info(f"Writing converted data to {output_path}")

    write_to_directory(
        convert_func=convert_func,
        postprocess_func=postprocess_func,
        scenarios=scenarios,
        output_path=output_path,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        num_workers=args.num_workers,
        preprocess=preprocess_func,
        **additional_args,
    )

    logger.info(f"Finished processing {dataset}")


if __name__ == "__main__":
    # Set up logging first thing
    setup_logger()
    logger.info("Starting dataset conversion process")

    parser = argparse.ArgumentParser(description="Build database from various scenarios")
    parser.add_argument(
        "--waymo_src",
        type=str,
        default=None,
        help="The directory storing the raw Waymo data",
    )
    parser.add_argument(
        "--nuplan_src",
        type=str,
        default=None,
        help="The directory storing the raw NuPlan data",
    )
    parser.add_argument(
        "--nuscenes_src",
        type=str,
        default=None,
        help="The directory storing the raw nuScenes data",
    )
    parser.add_argument(
        "--argoverse2_src",
        type=str,
        default=None,
        help="The directory storing the raw Argoverse data",
    )
    parser.add_argument(
        "--openscenes_metadata_src",
        type=str,
        default=None,
        help="The directory storing the openscenes metadata"
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="A directory, the path to place the converted data",
    )
    parser.add_argument(
        "--target_format",
        type=str,
        default="pickle",
        help="The target format for conversion (pickle, tfexample or gpudrive).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=1,
        help="Number of shards to split the output into",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use",
    )
    parser.add_argument(
        "--tfrecord_name",
        type=str,
        default="training",
        help="Name of the output tfrecord file",
    )
    parser.add_argument(
        "--split",
        default="v1.0-trainval",
        help="Which splits of nuScenes data should be used. Only applicable for nuScenes dataset.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Optional number of files to convert",
    )
    args = parser.parse_args()

    # Update log level and file if specified
    if args.log_level or args.log_file:
        log_level = getattr(logging, args.log_level) if args.log_level else None
        setup_logger(log_level=log_level, log_file=args.log_file)
        logger.info(f"Log level set to {args.log_level}")
        if args.log_file:
            logger.info(f"Logging to file: {args.log_file}")

    logger.info(f"Output directory: {args.dst}")
    logger.info(f"Using {args.num_workers} workers")

    datasets_to_process = []

    if args.waymo_src is not None:
        datasets_to_process.append("waymo")
    if args.nuplan_src is not None and args.openscenes_metadata_src is not None:
        datasets_to_process.append("openscenes")
    elif args.nuplan_src is not None:
        datasets_to_process.append("nuplan")
    if args.nuscenes_src is not None:
        datasets_to_process.append("nuscenes")
    if args.argoverse2_src is not None:
        datasets_to_process.append("argoverse2")

    logger.info(f"Will process the following datasets: {', '.join(datasets_to_process)}")

    for dataset in datasets_to_process:
        convert_dataset(args, dataset)

    if args.target_format == "tfexample":
        logger.info("Merging final TFRecord files")
        merge_tfrecord_files(args.dst, f"{args.tfrecord_name}.tfrecord")

        if args.shard > 1:
            logger.info(f"Sharding output into {args.shard} shards")
            shard_tfrecord(
                src=args.dst,
                filename=args.tfrecord_name,
                num_threads=args.num_workers,
                num_shards=args.shard,
            )

    logger.info("Dataset conversion completed successfully")

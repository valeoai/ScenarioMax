import scenariomax.tf_suppress  # noqa: F401, I001
import argparse
import logging
import os
import time
from pathlib import Path

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
    """Convert a dataset with enhanced progress tracking and timing."""
    start_time = time.time()
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting conversion for dataset: {dataset}")
    logger.info(f"Target format: {args.target_format}")
    logger.info(f"Workers: {args.num_workers}")

    if args.target_format == "tfexample":
        postprocess_func = postprocess_tfexample
    elif args.target_format == "gpudrive":
        postprocess_func = postprocess_gpudrive
    elif args.target_format == "pickle":
        postprocess_func = None
    else:
        raise ValueError(f"Unsupported target format: {args.target_format}")

    # Dataset-specific loading with timing
    load_start = time.time()

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
        if args.nuplan_direct_from_logs:
            scenarios = nuplan.get_nuplan_scenarios_by_scene(
                args.nuplan_src,
                os.getenv("NUPLAN_MAPS_ROOT"),
                num_files=args.num_files,                
            )
        else:
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

        logger.info(
            f"Loading OpenScenes scenarios from {args.openscenes_metadata_src} (maps: {os.getenv('NUPLAN_MAPS_ROOT', 'N/A')})",  # noqa: E501
        )
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

    load_time = time.time() - load_start
    logger.info(f"✓ Loaded {len(scenarios)} scenarios from {dataset} in {load_time:.2f}s")

    output_path = args.dst + f"/{dataset}"
    logger.info(f"📁 Writing converted data to {output_path}")

    # Main processing with detailed timing
    processing_start = time.time()

    try:
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

        processing_time = time.time() - processing_start
        total_time = time.time() - start_time

        logger.info(f"✅ Successfully processed {dataset}")
        logger.info("📈 Processing stats:")
        logger.info(f"   • Total scenarios: {len(scenarios)}")
        logger.info(f"   • Processing time: {processing_time:.2f}s")
        logger.info(f"   • Total time (including load): {total_time:.2f}s")
        logger.info(f"   • Average per scenario: {processing_time / len(scenarios) * 1000:.1f}ms")

        # Check output directory size
        if Path(output_path).exists():
            output_size = sum(f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file())
            logger.info(f"   • Output size: {output_size / (1024**3):.2f} GB")

    except Exception as e:
        processing_time = time.time() - processing_start
        logger.error(f"❌ Failed to process {dataset} after {processing_time:.2f}s: {e}")
        raise

    logger.info(f"{'=' * 60}")


def main():
    """Main function with enhanced pipeline tracking."""
    # Set up logging first thing
    setup_logger()

    pipeline_start = time.time()
    logger.info("🚀 Starting ScenarioMax dataset conversion pipeline")
    logger.info(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
        "--nuplan_direct_from_logs",
        type=bool,
        default=False,
        help="If true, does not use the nuplan devkit to create scenarios and instead parses scenes directly from logs",
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
        help="The directory storing the openscenes metadata",
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

    # Log configuration summary
    logger.info("⚙️  Pipeline Configuration:")
    logger.info(f"   • Output directory: {args.dst}")
    logger.info(f"   • Target format: {args.target_format}")
    logger.info(f"   • Workers: {args.num_workers}")
    logger.info(f"   • Log level: {args.log_level}")
    if args.log_file:
        logger.info(f"   • Log file: {args.log_file}")
    if args.num_files:
        logger.info(f"   • File limit: {args.num_files}")

    # Update log level and file if specified
    if args.log_level or args.log_file:
        print(f"Setting up logger with level: {args.log_level} and file: {args.log_file}")
        log_level = getattr(logging, args.log_level) if args.log_level else None
        setup_logger(log_level=log_level, log_file=args.log_file)
        logger.info(f"Log level set to {args.log_level}")
        if args.log_file:
            logger.info(f"Logging to file: {args.log_file}")

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

    if not datasets_to_process:
        logger.warning("⚠️  No datasets specified for processing")
        return

    logger.info(f"📋 Processing pipeline: {' → '.join(datasets_to_process)}")

    # Track overall pipeline progress
    pipeline_stats = {
        "total_datasets": len(datasets_to_process),
        "completed_datasets": 0,
        "failed_datasets": 0,
        "total_scenarios": 0,
        "total_processing_time": 0,
    }

    for i, dataset in enumerate(datasets_to_process, 1):
        logger.info(f"📊 Pipeline Progress: {i}/{len(datasets_to_process)} datasets")

        try:
            dataset_start = time.time()
            convert_dataset(args, dataset)
            dataset_time = time.time() - dataset_start

            pipeline_stats["completed_datasets"] += 1
            pipeline_stats["total_processing_time"] += dataset_time

        except Exception as e:
            pipeline_stats["failed_datasets"] += 1
            logger.error(f"❌ Dataset {dataset} failed: {e}")
            continue

    # Post-processing for TFExample format
    if args.target_format == "tfexample":
        logger.info("🔄 Starting post-processing for TFExample format...")
        postprocess_start = time.time()

        try:
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

            postprocess_time = time.time() - postprocess_start
            pipeline_stats["total_processing_time"] += postprocess_time
            logger.info(f"✅ Post-processing completed in {postprocess_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Post-processing failed: {e}")

    # Final pipeline summary
    total_pipeline_time = time.time() - pipeline_start

    logger.info("🏁 Pipeline Completion Summary:")
    logger.info(f"   • Total time: {total_pipeline_time:.2f}s ({total_pipeline_time / 60:.1f} minutes)")
    logger.info(f"   • Datasets processed: {pipeline_stats['completed_datasets']}/{pipeline_stats['total_datasets']}")
    if pipeline_stats["failed_datasets"] > 0:
        logger.warning(f"   • Failed datasets: {pipeline_stats['failed_datasets']}")

    if pipeline_stats["completed_datasets"] > 0:
        logger.info("✅ Dataset conversion pipeline completed successfully")
    else:
        logger.error("❌ Pipeline failed - no datasets processed successfully")


if __name__ == "__main__":
    main()

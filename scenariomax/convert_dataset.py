import argparse
import logging
import os
import warnings


# Suppress TensorFlow logs before any potential imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorboard.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*tensorflow.*")

# Set TensorFlow logging levels
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)

from scenariomax import logger_utils  # noqa: E402
from scenariomax.core import pipeline  # noqa: E402


logger = logger_utils.get_logger(__name__)


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert autonomous driving datasets to unified formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Use Case 1: Raw → Pickle
    python convert_dataset.py --waymo_src /data/waymo --dst /output --target_format pickle

    # Use Case 2: Raw → Enhanced Pickle
    python convert_dataset.py --waymo_src /data/waymo --dst /output --target_format pickle --enable_enhancement

    # Use Case 3: Pickle → TFRecord
    python convert_dataset.py --pickle_src /data/pickle --dst /output --target_format tfexample

    # Use Case 4: Pickle → Enhanced → TFRecord
    python convert_dataset.py --pickle_src /data/pickle --dst /output --target_format tfexample --enable_enhancement

    # Use Case 5: Raw → TFRecord (streaming)
    python convert_dataset.py --waymo_src /data/waymo --dst /output --target_format tfexample

    # Use Case 6: Raw → Enhanced → TFRecord (streaming)
    python convert_dataset.py --waymo_src /data/waymo --dst /output --target_format tfexample --enable_enhancement

    # Multi-dataset conversion
    python convert_dataset.py --waymo_src /data/waymo --nuscenes_src /data/nuscenes --dst /output --target_format tfexample
    """,  # noqa: E501
    )

    # Dataset sources
    parser.add_argument("--waymo_src", help="Waymo dataset directory")
    parser.add_argument("--nuplan_src", help="NuPlan dataset directory")
    parser.add_argument("--nuscenes_src", help="nuScenes dataset directory")
    parser.add_argument("--argoverse2_src", help="Argoverse2 dataset directory")
    parser.add_argument("--openscenes_metadata_src", help="OpenScenes metadata directory")
    parser.add_argument("--pickle_src", help="Pickle dataset directory")

    # Output configuration
    parser.add_argument("--dst", required=True, help="Output directory")
    parser.add_argument(
        "--target_format",
        default="pickle",
        choices=["pickle", "tfexample", "gpudrive"],
        help="Target format (default: pickle)",
    )

    # Processing options
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--shard", type=int, default=1, help="Number of output shards")
    parser.add_argument("--num_files", type=int, help="Limit number of files to process")
    parser.add_argument(
        "--enable_enhancement",
        action="store_true",
        help="Enable scenario enhancement step (placeholder functionality)",
    )

    # Format-specific options
    parser.add_argument("--tfrecord_name", default="training", help="TFRecord filename")
    parser.add_argument("--split", default="v1.0-trainval", help="nuScenes split (default: v1.0-trainval)")
    parser.add_argument(
        "--nuplan_direct_from_logs",
        action="store_true",
        help="Parse nuPlan scenes directly from logs (alternative method)",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--log_file", help="Log file path")

    return parser


def main():
    """Main entry point for dataset conversion."""
    logger_utils.setup_logger()

    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    if args.log_level or args.log_file:
        log_level = getattr(logging, args.log_level) if args.log_level else None
        logger_utils.setup_logger(log_level=log_level, log_file=args.log_file)

    # Convert args to unified pipeline format
    source_paths = {}
    source_type = "raw"

    if args.pickle_src:
        source_type = "pickle"
        source_paths["pickle"] = args.pickle_src
    else:
        if args.waymo_src:
            source_paths["waymo"] = args.waymo_src
        if args.nuplan_src:
            if args.openscenes_metadata_src:
                source_paths["openscenes"] = args.nuplan_src
            else:
                source_paths["nuplan"] = args.nuplan_src
        if args.nuscenes_src:
            source_paths["nuscenes"] = args.nuscenes_src
        if args.argoverse2_src:
            source_paths["argoverse2"] = args.argoverse2_src

    if not source_paths:
        logger.error("❌ No datasets specified")
        parser.print_help()
        exit(1)

    # Determine stream mode (avoid disk I/O for raw->target conversions)
    stream_mode = source_type == "raw" and args.target_format != "pickle"

    # Run unified conversion pipeline
    stats = pipeline.convert_dataset(
        source_type=source_type,
        source_paths=source_paths,
        target_format=args.target_format,
        output_path=args.dst,
        enhancement=args.enable_enhancement,
        stream_mode=stream_mode,
        num_workers=args.num_workers,
        # Additional arguments
        num_files=args.num_files,
        split=args.split,
        tfrecord_name=args.tfrecord_name,
        shard=args.shard,
        openscenes_metadata_src=args.openscenes_metadata_src,
        nuplan_direct_from_logs=args.nuplan_direct_from_logs,
    )
    if stats:
        logger.info(f"✅ Conversion completed successfully: {stats}")
    else:
        logger.error("❌ Conversion failed or no scenarios processed")
    logger.info("✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()

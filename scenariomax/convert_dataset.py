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
    """Create and configure argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="ScenarioMax: Convert AV datasets through a 3-stage pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
3-Stage Pipeline Architecture:
  1. convert: Raw dataset(s) → Unified pickles
  2. process: Unified pickles → Processed pickles (optional)
  3. format:  Unified pickles → Target format (tfrecord/json)

  pipeline: Run all 3 stages together
  viz: Visualize unified scenarios in Bird's Eye View (BEV)

Examples:
  # Stage 1: Convert raw Waymo to unified format
  scenariomax convert --waymo_src /data/waymo --dst /output/unified --num_workers 16

  # Stage 1: Convert multiple datasets
  scenariomax convert --waymo_src /data/waymo --nuplan_src /data/nuplan --dst /output/unified

  # Stage 2: Process unified scenarios
  scenariomax process --src /output/unified --dst /output/processed --traffic-lights

  # Stage 3: Convert to TFRecord with sharding
  scenariomax format --src /output/processed --dst /output/tfrecord --format tfexample --shard 10

  # Stage 3: Convert to JSON
  scenariomax format --src /output/processed --dst /output/json --format json

  # Stage 3: Convert to Puffer
  scenariomax format --src /output/processed --dst /output/puffer --format puffer

  # Visualize unified scenarios (BEV PNG images at first timestep)
  scenariomax viz --src /output/unified --dst /output/viz --format png --max-scenarios 100

  # Generate animated videos
  scenariomax viz --src /output/unified --dst /output/videos --format video --fps 10 --max-scenarios 10

  # Full pipeline: All 3 stages at once
  scenariomax pipeline --waymo_src /data/waymo --dst /output --format tfexample --process --shard 10

  # Full pipeline: Multiple datasets
  scenariomax pipeline --waymo_src /data/waymo --nuplan_src /data/nuplan --dst /output --format json
        """,
    )

    # Logging arguments (shared across all subcommands)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--log_file", help="Log file path")

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage", required=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Subcommand: convert (Stage 1: Raw → Unified)
    # ═══════════════════════════════════════════════════════════════════════
    convert_parser = subparsers.add_parser(
        "convert",
        help="Stage 1: Convert raw dataset(s) to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset sources
    convert_parser.add_argument("--waymo_src", help="Waymo dataset directory")
    convert_parser.add_argument("--nuplan_src", help="NuPlan dataset directory")
    convert_parser.add_argument("--nuscenes_src", help="nuScenes dataset directory")
    convert_parser.add_argument("--argoverse2_src", help="Argoverse2 dataset directory")
    convert_parser.add_argument("--openscenes_metadata_src", help="OpenScenes metadata directory")

    # Output
    convert_parser.add_argument("--dst", required=True, help="Output directory for unified pickles")

    # Processing options
    convert_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers (default: 8)")
    convert_parser.add_argument("--num_files", type=int, help="Limit number of files to process")

    # Dataset-specific options
    convert_parser.add_argument("--split", default="v1.0-trainval", help="nuScenes split (default: v1.0-trainval)")
    convert_parser.add_argument(
        "--nuplan_direct_from_logs",
        action="store_true",
        help="Parse nuPlan scenes directly from logs",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Subcommand: process (Stage 2: Unified → Processed)
    # ═══════════════════════════════════════════════════════════════════════
    process_parser = subparsers.add_parser(
        "process",
        help="Stage 2: Process unified scenarios with transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    process_parser.add_argument("--src", required=True, help="Input directory with unified pickles")
    process_parser.add_argument("--dst", required=True, help="Output directory for processed pickles")
    process_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers (default: 8)")

    # Processor options
    process_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run soft validation (structure checks)",
    )
    process_parser.add_argument(
        "--validate-strict",
        action="store_true",
        help="Run strict validation (physics checks)",
    )
    process_parser.add_argument(
        "--traffic-lights",
        action="store_true",
        help="Add traffic light data",
    )
    process_parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving output (validation-only mode)",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Subcommand: format (Stage 3: Unified → Target Format)
    # ═══════════════════════════════════════════════════════════════════════
    format_parser = subparsers.add_parser(
        "format",
        help="Stage 3: Convert unified pickles to target format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    format_parser.add_argument("--src", required=True, help="Input directory with unified pickles")
    format_parser.add_argument("--dst", required=True, help="Output directory for target format")
    format_parser.add_argument(
        "--format",
        required=True,
        choices=["tfexample", "json", "puffer"],
        help="Target format",
    )
    format_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers (default: 8)")

    # Format-specific options
    format_parser.add_argument("--shard", type=int, default=1, help="Number of output shards (tfexample only)")
    format_parser.add_argument(
        "--tfrecord_name",
        default="training",
        help="TFRecord filename (default: training)",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Subcommand: viz (Visualize unified scenarios)
    # ═══════════════════════════════════════════════════════════════════════
    viz_parser = subparsers.add_parser(
        "viz",
        help="Visualize unified scenarios in Bird's Eye View (BEV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    viz_parser.add_argument("--src", required=True, help="Input directory with unified pickle files")
    viz_parser.add_argument("--dst", required=True, help="Output directory for PNG/MP4 visualizations")
    viz_parser.add_argument(
        "--format",
        choices=["png", "video"],
        default="png",
        help="Output format: png (first timestep) or video (animated) (default: png)",
    )
    viz_parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video output (default: 10)",
    )
    viz_parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Maximum number of scenarios to visualize (default: all)",
    )
    viz_parser.add_argument(
        "--no-log-trajectory",
        action="store_true",
        help="Don't show trajectory history",
    )
    viz_parser.add_argument(
        "--scatter-map",
        action="store_true",
        help="Render road map as scattered points instead of lines",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Subcommand: pipeline (Full 3-stage pipeline)
    # ═══════════════════════════════════════════════════════════════════════
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run all 3 stages together (Raw → Unified → Processed → Target)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset sources
    pipeline_parser.add_argument("--waymo_src", help="Waymo dataset directory")
    pipeline_parser.add_argument("--nuplan_src", help="NuPlan dataset directory")
    pipeline_parser.add_argument("--nuscenes_src", help="nuScenes dataset directory")
    pipeline_parser.add_argument("--argoverse2_src", help="Argoverse2 dataset directory")
    pipeline_parser.add_argument("--openscenes_metadata_src", help="OpenScenes metadata directory")

    # Output
    pipeline_parser.add_argument("--dst", required=True, help="Base output directory")

    # Target format
    pipeline_parser.add_argument(
        "--format",
        required=True,
        choices=["tfexample", "json", "puffer"],
        help="Target format",
    )

    # Processing options
    pipeline_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers (default: 8)")
    pipeline_parser.add_argument("--num_files", type=int, help="Limit number of files to process")

    # Processing stage
    pipeline_parser.add_argument(
        "--process",
        action="store_true",
        help="Enable processing stage (Stage 2)",
    )
    pipeline_parser.add_argument(
        "--traffic-lights",
        action="store_true",
        help="Add traffic light data (implies --process)",
    )

    # Pipeline mode
    pipeline_parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate pickles to disk (default: in-memory streaming)",
    )
    pipeline_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for in-memory processing (default: 100 scenarios)",
    )

    # Format-specific options
    pipeline_parser.add_argument("--shard", type=int, default=1, help="Number of output shards (tfexample only)")
    pipeline_parser.add_argument(
        "--tfrecord_name",
        default="training",
        help="TFRecord filename (default: training)",
    )

    # Dataset-specific options
    pipeline_parser.add_argument("--split", default="v1.0-trainval", help="nuScenes split (default: v1.0-trainval)")
    pipeline_parser.add_argument(
        "--nuplan_direct_from_logs",
        action="store_true",
        help="Parse nuPlan scenes directly from logs",
    )

    return parser


def handle_convert_command(args):
    """Handle Stage 1: Raw → Unified."""
    # Build datasets dict
    datasets = {}
    if args.waymo_src:
        datasets["waymo"] = args.waymo_src
    if args.nuplan_src:
        if args.openscenes_metadata_src:
            datasets["openscenes"] = args.nuplan_src
        else:
            datasets["nuplan"] = args.nuplan_src
    if args.nuscenes_src:
        datasets["nuscenes"] = args.nuscenes_src
    if args.argoverse2_src:
        datasets["argoverse2"] = args.argoverse2_src

    if not datasets:
        logger.error("❌ No datasets specified. Use --waymo_src, --nuplan_src, etc.")
        return 1

    # Run Stage 1
    stats = pipeline.convert_raw_to_unified(
        datasets=datasets,
        output_path=args.dst,
        num_workers=args.num_workers,
        num_files=args.num_files,
        split=args.split,
        openscenes_metadata_src=args.openscenes_metadata_src,
        nuplan_direct_from_logs=args.nuplan_direct_from_logs,
    )

    logger.info(f"✅ Stage 1 completed: {stats}")
    return 0


def handle_process_command(args):
    """Handle Stage 2: Unified → Processed."""
    # Build processors list based on flags
    processors = []

    # Validation (runs first if specified)
    if args.validate or args.validate_strict:
        from scenariomax.stage2_process import validate_scenario

        # Create validation processor with appropriate strictness
        def validation_processor(scenario):
            return validate_scenario(scenario, True)  #  strict=args.validate_strict)

        processors.append(validation_processor)

    # Traffic lights (legacy/default)
    if args.traffic_lights:
        from scenariomax.stage2_process import enhance_scenarios

        processors.append(enhance_scenarios)

    # Default behavior if no processors specified
    if not processors:
        logger.warning("⚠️  No processors specified. Available options:")
        logger.warning("   --validate / --validate-strict: Validate scenarios")
        logger.warning("")
        logger.warning("   Proceeding with default processor (enhance_scenarios)")

        from scenariomax.stage2_process import enhance_scenarios

        processors = [enhance_scenarios]

    # Determine if output should be saved
    save_output = not args.no_output

    # Run Stage 2
    stats = pipeline.process_unified_scenarios(
        input_path=args.src,
        output_path=args.dst,
        processors=processors,
        num_workers=args.num_workers,
        save_output=save_output,
    )

    logger.info(f"✅ Stage 2 completed: {stats}")
    return 0


def handle_format_command(args):
    """Handle Stage 3: Unified → Target Format."""
    # Run Stage 3
    stats = pipeline.format_unified_to_target(
        input_path=args.src,
        output_path=args.dst,
        format=args.format,
        num_workers=args.num_workers,
        shard=args.shard,
        tfrecord_name=args.tfrecord_name,
    )

    logger.info(f"✅ Stage 3 completed: {stats}")
    return 0


def handle_viz_command(args):
    """Handle visualization: Unified pickles → BEV PNG images or MP4 videos."""
    from scenariomax.visualization import visualize_scenarios

    # Run visualization
    stats = visualize_scenarios(
        input_path=args.src,
        output_path=args.dst,
        max_scenarios=args.max_scenarios,
        show_trajectory=not args.no_log_trajectory,
        output_format=args.format,
        fps=args.fps,
        scatter_map=args.scatter_map,
    )

    logger.info(f"✅ Visualization completed: {stats}")
    return 0


def handle_pipeline_command(args):
    """Handle full pipeline: Raw → Unified → Processed → Target."""
    # Build datasets dict
    datasets = {}
    if args.waymo_src:
        datasets["waymo"] = args.waymo_src
    if args.nuplan_src:
        if args.openscenes_metadata_src:
            datasets["openscenes"] = args.nuplan_src
        else:
            datasets["nuplan"] = args.nuplan_src
    if args.nuscenes_src:
        datasets["nuscenes"] = args.nuscenes_src
    if args.argoverse2_src:
        datasets["argoverse2"] = args.argoverse2_src

    if not datasets:
        logger.error("❌ No datasets specified. Use --waymo_src, --nuplan_src, etc.")
        return 1

    # Build processors list
    processors = None
    if args.process or args.traffic_lights:
        from scenariomax.stage2_process import enhance_scenarios

        processors = [enhance_scenarios]

    # Run full pipeline
    stats = pipeline.process_scenarios(
        datasets=datasets,
        output_path=args.dst,
        format=args.format,
        processors=processors,
        num_workers=args.num_workers,
        save_intermediate=args.save_intermediate,
        batch_size=args.batch_size,
        num_files=args.num_files,
        split=args.split,
        shard=args.shard,
        tfrecord_name=args.tfrecord_name,
        openscenes_metadata_src=args.openscenes_metadata_src,
        nuplan_direct_from_logs=args.nuplan_direct_from_logs,
    )

    logger.info(f"✅ Full pipeline completed: {stats}")
    return 0


def main():
    """Main entry point for dataset conversion."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging once with user-specified options
    log_level = getattr(logging, args.log_level) if args.log_level else logging.INFO
    logger_utils.setup_logger(log_level=log_level, log_file=args.log_file)

    # Route to appropriate command handler
    if args.command == "convert":
        return handle_convert_command(args)
    elif args.command == "process":
        return handle_process_command(args)
    elif args.command == "format":
        return handle_format_command(args)
    elif args.command == "viz":
        return handle_viz_command(args)
    elif args.command == "pipeline":
        return handle_pipeline_command(args)
    else:
        logger.error(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())

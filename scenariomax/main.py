"""
ScenarioMax main entry point with Hydra configuration.

This module replaces the argparse-based CLI with Hydra configuration management.
All settings are loaded from scenariomax/config.yaml and can be overridden from CLI.

Usage:
    # Use default config
    scenariomax

    # Override specific values
    scenariomax command=convert datasets.waymo=/data/waymo

    # Full pipeline
    scenariomax command=pipeline datasets.waymo=/data/waymo format=tfexample

    # Multiple processors
    scenariomax command=process processors=[validation,traffic_lights]
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


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

# Get config directory dynamically for Hydra
_CONFIG_DIR = str(Path(__file__).parent)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration before execution."""
    command = cfg.command

    if command in ["convert", "pipeline"]:
        # Check that at least one dataset is specified
        datasets_specified = any(
            [
                cfg.datasets.waymo,
                cfg.datasets.nuplan,
                cfg.datasets.nuscenes,
                cfg.datasets.argoverse2,
                cfg.datasets.openscenes,
            ],
        )
        if not datasets_specified:
            raise ValueError(
                "No datasets specified. Set at least one dataset path:\n"
                "  datasets.waymo=/path/to/waymo\n"
                "  datasets.nuplan=/path/to/nuplan\n"
                "  etc.",
            )

    if command in ["process", "format"] and not cfg.output.dst:
        # These commands need source path in output.dst
        raise ValueError(f"Command '{command}' requires output.dst to be set")


def build_datasets_dict(cfg: DictConfig) -> dict[str, str]:
    """Build datasets dictionary from config."""
    datasets = {}

    if cfg.datasets.waymo:
        datasets["waymo"] = cfg.datasets.waymo
    if cfg.datasets.nuplan:
        if cfg.dataset_options.openscenes_metadata_src:
            datasets["openscenes"] = cfg.datasets.nuplan
        else:
            datasets["nuplan"] = cfg.datasets.nuplan
    if cfg.datasets.nuscenes:
        datasets["nuscenes"] = cfg.datasets.nuscenes
    if cfg.datasets.argoverse2:
        datasets["argoverse2"] = cfg.datasets.argoverse2
    if cfg.datasets.openscenes:
        datasets["openscenes"] = cfg.datasets.openscenes

    return datasets


def handle_convert_command(cfg: DictConfig):
    """Handle Stage 1: Raw → Unified."""
    logger.info("🚀 Executing command: convert (Stage 1)")

    datasets = build_datasets_dict(cfg)

    stats = pipeline.convert_raw_to_unified(
        datasets=datasets,
        output_path=cfg.output.dst,
        num_workers=cfg.execution.num_workers,
        batch_size=cfg.execution.batch_size,
        num_files=cfg.dataset_options.num_files,
        split=cfg.dataset_options.split,
        openscenes_metadata_src=cfg.dataset_options.openscenes_metadata_src,
        nuplan_direct_from_logs=cfg.dataset_options.nuplan_direct_from_logs,
    )

    logger.info(f"✅ Stage 1 completed: {stats}")
    return 0


def handle_process_command(cfg: DictConfig):
    """Handle Stage 2: Unified → Processed."""
    logger.info("🚀 Executing command: process (Stage 2)")

    # Get processors and configs from config file
    processors = cfg.processing.processors if cfg.processing.processors else None
    processor_configs = OmegaConf.to_container(cfg.processing.processor_configs, resolve=True)

    # If no processors specified, warn user
    if not processors or len(processors) == 0:
        from scenariomax.stage2_process import list_processors

        available = list(list_processors().keys())
        logger.warning("⚠️  No processors specified in config. Available processors:")
        for name in available:
            logger.warning(f"   - {name}")
        logger.warning("")
        logger.warning("   Set 'processing.processors=[validation]' in config or override from CLI")
        logger.warning("   Example: scenariomax command=process processors=[validation]")
        logger.warning("")
        logger.warning("   Proceeding with no-op processor")

        # Use no-op for backward compatibility
        from scenariomax.stage2_process import enhance_scenarios

        processors = [enhance_scenarios]
        processor_configs = None

    # For 'process' command, src is the input and dst is the output
    # We use output.dst for both, but user should override
    input_path = cfg.get("input_path", cfg.output.dst)
    output_path = cfg.output.dst

    stats = pipeline.process_unified_scenarios(
        input_path=input_path,
        output_path=output_path,
        processors=processors,
        processor_configs=processor_configs,
        num_workers=cfg.execution.num_workers,
        save_output=cfg.processing.save_output,
    )

    logger.info(f"✅ Stage 2 completed: {stats}")
    return 0


def handle_format_command(cfg: DictConfig):
    """Handle Stage 3: Unified → Target Format."""
    logger.info("🚀 Executing command: format (Stage 3)")

    # For 'format' command, src is the input
    input_path = cfg.get("input_path", cfg.output.dst)
    output_path = cfg.output.dst

    # Get optional processors from format_options
    processors = cfg.format_options.get("processors", None)
    processor_configs = cfg.format_options.get("processor_configs", None)

    # Convert OmegaConf to regular Python types
    if processor_configs:
        processor_configs = OmegaConf.to_container(processor_configs, resolve=True)

    stats = pipeline.format_unified_to_target(
        input_path=input_path,
        output_path=output_path,
        format=cfg.output.format,
        num_workers=cfg.execution.num_workers,
        processors=processors,
        processor_configs=processor_configs,
        shard=cfg.format_options.shard,
        tfrecord_name=cfg.format_options.tfrecord_name,
    )

    logger.info(f"✅ Stage 3 completed: {stats}")
    return 0


def handle_viz_command(cfg: DictConfig):
    """Handle visualization: Unified pickles → BEV images/videos."""
    logger.info("🚀 Executing command: viz (Visualization)")

    from scenariomax.visualization import visualize_scenarios

    # For 'viz' command, src is the input
    input_path = cfg.get("input_path", cfg.output.dst)
    output_path = cfg.output.dst

    stats = visualize_scenarios(
        input_path=input_path,
        output_path=output_path,
        max_scenarios=cfg.visualization.max_scenarios,
        show_trajectory=cfg.visualization.show_trajectory,
        output_format=cfg.visualization.output_format,
        fps=cfg.visualization.fps,
        scatter_map=cfg.visualization.scatter_map,
    )

    logger.info(f"✅ Visualization completed: {stats}")
    return 0


def handle_pipeline_command(cfg: DictConfig):
    """Handle full pipeline: Raw → Unified → Processed → Target."""
    logger.info("🚀 Executing command: pipeline (Full 3-stage)")

    datasets = build_datasets_dict(cfg)

    # Get processors and configs from config file
    processors = cfg.processing.processors if cfg.processing.processors else None
    processor_configs = OmegaConf.to_container(cfg.processing.processor_configs, resolve=True)

    stats = pipeline.process_scenarios(
        datasets=datasets,
        output_path=cfg.output.dst,
        format=cfg.output.format,
        processors=processors,
        processor_configs=processor_configs,
        num_workers=cfg.execution.num_workers,
        batch_size=cfg.execution.batch_size,
        save_intermediate=cfg.pipeline.save_intermediate,
        num_files=cfg.dataset_options.num_files,
        split=cfg.dataset_options.split,
        shard=cfg.format_options.shard,
        tfrecord_name=cfg.format_options.tfrecord_name,
        openscenes_metadata_src=cfg.dataset_options.openscenes_metadata_src,
        nuplan_direct_from_logs=cfg.dataset_options.nuplan_direct_from_logs,
    )

    logger.info(f"✅ Full pipeline completed: {stats}")
    return 0


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for ScenarioMax with Hydra configuration.

    Args:
        cfg: Hydra configuration object loaded from config.yaml

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Configure logging
    log_level = getattr(logging, cfg.logging.level) if cfg.logging.level else logging.INFO
    logger_utils.setup_logger(log_level=log_level, log_file=cfg.logging.log_file)

    # Print configuration
    logger.info("=" * 80)

    # Validate configuration
    try:
        validate_config(cfg)
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return 1

    # Route to appropriate command handler
    command = cfg.command
    try:
        if command == "convert":
            return handle_convert_command(cfg)
        elif command == "process":
            return handle_process_command(cfg)
        elif command == "format":
            return handle_format_command(cfg)
        elif command == "viz":
            return handle_viz_command(cfg)
        elif command == "pipeline":
            return handle_pipeline_command(cfg)
        else:
            logger.error(f"❌ Unknown command: {command}")
            logger.error("   Available commands: convert, process, format, viz, pipeline")
            return 1
    except Exception as e:
        logger.error(f"❌ Command failed: {e}", exc_info=True)
        return 1


def cli_entry_point():
    """CLI entry point wrapper for setuptools."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())

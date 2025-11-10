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
    scenariomax command=pipeline datasets.waymo=/data/waymo format=waymax

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
        # Check that at least one dataset has path specified
        datasets_specified = any(
            [
                cfg.datasets.waymo.path,
                cfg.datasets.nuplan.path,
                cfg.datasets.nuscenes.path,
                cfg.datasets.openscenes.path,
            ],
        )
        if not datasets_specified:
            raise ValueError(
                "No datasets specified. Set at least one dataset path:\n"
                "  datasets.waymo.path=/path/to/waymo\n"
                "  datasets.nuplan.path=/path/to/nuplan\n"
                "  etc.",
            )

    if command in ["process", "format", "viz"]:
        # These commands need input path
        if not cfg.paths.input_dir and not cfg.paths.output_dir:
            raise ValueError(
                f"Command '{command}' requires paths.input_dir or paths.output_dir to be set",
            )

    # OpenScenes requires metadata_dir
    if cfg.datasets.openscenes.path and not cfg.datasets.openscenes.metadata_dir:
        raise ValueError(
            "OpenScenes dataset requires metadata_dir to be specified:\n"
            "  datasets.openscenes.metadata_dir=/path/to/metadata",
        )


def build_datasets_dict(cfg: DictConfig) -> dict[str, dict]:
    """Build datasets dictionary from config, filtering those with path set."""
    datasets = {}

    # Only include datasets that have path specified
    if cfg.datasets.waymo.path:
        datasets["waymo"] = OmegaConf.to_container(cfg.datasets.waymo, resolve=True)
    if cfg.datasets.nuplan.path:
        datasets["nuplan"] = OmegaConf.to_container(cfg.datasets.nuplan, resolve=True)
    if cfg.datasets.nuscenes.path:
        datasets["nuscenes"] = OmegaConf.to_container(cfg.datasets.nuscenes, resolve=True)
    if cfg.datasets.openscenes.path:
        datasets["openscenes"] = OmegaConf.to_container(cfg.datasets.openscenes, resolve=True)

    return datasets


def handle_convert_command(cfg: DictConfig):
    """Handle Stage 1: Raw → Unified."""
    logger.info("🚀 Executing command: convert (Stage 1)")

    datasets = build_datasets_dict(cfg)

    stats = pipeline.convert_raw_to_unified(
        datasets=datasets,
        output_path=cfg.paths.output_dir,
        num_workers=cfg.execution.num_workers,
        batch_size=cfg.execution.batch_size,
    )

    logger.info(f"✅ Stage 1 completed: {stats}")
    return 0


def handle_process_command(cfg: DictConfig):
    """Handle Stage 2: Unified → Processed."""
    logger.info("🚀 Executing command: process (Stage 2)")

    # Get processors list
    processors = cfg.processing.processors if cfg.processing.processors else None

    # Extract processor configs (everything in processing except 'processors' key)
    processing_dict = OmegaConf.to_container(cfg.processing, resolve=True)
    processor_configs = {k: v for k, v in processing_dict.items() if k != "processors"}

    # If no processors specified, warn user
    if not processors or len(processors) == 0:
        logger.warning("⚠️  No processors specified. Scenarios will be copied without modifications.")
        logger.warning("   Available processors: validation, traffic_lights, polyline_interpolation")
        logger.warning("")
        logger.warning("   Set 'processing.processors=[validation]' in config or override from CLI")
        logger.warning("   Example: scenariomax command=process processing.processors=[validation]")
        logger.warning("")
        logger.warning("   Proceeding with identity copy (no processing)")

        # Set empty list - will be handled by pipeline
        processors = []
        processor_configs = None

    # Determine input and output paths
    input_path = cfg.paths.input_dir if cfg.paths.input_dir else f"{cfg.paths.output_dir}/unified"
    output_path = cfg.paths.output_dir

    stats = pipeline.process_unified_scenarios(
        input_path=input_path,
        output_path=output_path,
        processors=processors,
        processor_configs=processor_configs,
        num_workers=cfg.execution.num_workers,
    )

    logger.info(f"✅ Stage 2 completed: {stats}")
    return 0


def handle_format_command(cfg: DictConfig):
    """Handle Stage 3: Unified → Target Format."""
    logger.info("🚀 Executing command: format (Stage 3)")

    # Determine input and output paths
    input_path = cfg.paths.input_dir if cfg.paths.input_dir else f"{cfg.paths.output_dir}/unified"
    output_path = cfg.paths.output_dir

    # Handle optional processors during formatting
    processors = None
    processor_configs = None
    if cfg.formatting.apply_processors:
        processors = cfg.processing.processors if cfg.processing.processors else None
        if processors:
            processing_dict = OmegaConf.to_container(cfg.processing, resolve=True)
            processor_configs = {k: v for k, v in processing_dict.items() if k != "processors"}

    # Get format-specific configs
    target_format = cfg.formatting.target_format
    format_config = OmegaConf.to_container(cfg.formatting.get(target_format, {}), resolve=True)

    stats = pipeline.format_unified_to_target(
        input_path=input_path,
        output_path=output_path,
        format=target_format,
        num_workers=cfg.execution.num_workers,
        processors=processors,
        processor_configs=processor_configs,
        format_config=format_config,
    )

    logger.info(f"✅ Stage 3 completed: {stats}")
    return 0


def handle_viz_command(cfg: DictConfig):
    """Handle visualization: Unified pickles → BEV images/videos."""
    logger.info("🚀 Executing command: viz (Visualization)")

    from scenariomax.visualization import visualize_scenarios

    # Determine input and output paths
    input_path = cfg.paths.input_dir if cfg.paths.input_dir else f"{cfg.paths.output_dir}/unified"
    output_path = cfg.paths.output_dir

    stats = visualize_scenarios(
        input_path=input_path,
        output_path=output_path,
        max_scenarios=cfg.visualization.max_scenarios,
        show_trajectory=cfg.visualization.show_trajectory,
        output_format=cfg.visualization.format,
        fps=cfg.visualization.fps,
        scatter_map=cfg.visualization.scatter_map,
        follow_ego=cfg.visualization.get("follow_ego", False),
        field_radius=cfg.visualization.get("field_radius", None),
    )

    logger.info(f"✅ Visualization completed: {stats}")
    return 0


def handle_pipeline_command(cfg: DictConfig):
    """Handle full pipeline: Raw → Unified → Processed → Target."""
    logger.info("🚀 Executing command: pipeline (Full 3-stage)")

    datasets = build_datasets_dict(cfg)

    # Get processors and configs from config file
    processors = cfg.processing.processors if cfg.processing.processors else None
    processing_dict = OmegaConf.to_container(cfg.processing, resolve=True)
    processor_configs = {k: v for k, v in processing_dict.items() if k != "processors"}

    # Get format-specific configs
    target_format = cfg.formatting.target_format
    format_config = OmegaConf.to_container(cfg.formatting.get(target_format, {}), resolve=True)

    stats = pipeline.run_all_pipeline(
        datasets=datasets,
        output_path=cfg.paths.output_dir,
        format=target_format,
        processors=processors,
        processor_configs=processor_configs,
        num_workers=cfg.execution.num_workers,
        batch_size=cfg.execution.batch_size,
        format_config=format_config,
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

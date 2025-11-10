"""
Simplified 3-stage pipeline for dataset conversion.

Core concept: One universal function that processes a single scenario/file through
any combination of stages (convert, process, format). Parallelization handled by joblib.
"""

import glob
import os
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax import dataset_registry, logger_utils
from scenariomax.core.types import FORMAT_GPUDRIVE, FORMAT_PUFFERDRIVE, FORMAT_WAYMAX, SUPPORTED_FORMATS
from scenariomax.core.utils import (
    NumpyEncoder,
    clean_and_create_output_directory,
    get_format_function,
    load_pickle,
    save_pickle,
)
from scenariomax.stage2_process.overpass_filtering.processor import OverpassDetectedException


logger = logger_utils.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Processing Function - Handles File Batches
# ═══════════════════════════════════════════════════════════════════════════


def worker_scenario_func(
    input_data: list[Any],
    convert_func: Callable | None = None,
    process_func: Callable | None = None,
    format_func: Callable | None = None,
    output_path: str | None = None,
    target_format: str | None = None,
    dataset_config: Any = None,
) -> dict[str, Any]:
    """
    Process batch of scenarios through the pipeline.

    Unified batch processing for all stages:
    - Stage 1 alone: List of file paths/metadata (with dataset_config, no process/format funcs)
    - Stage 2 alone: List of pickle paths (no dataset_config)
    - Stage 3 alone: List of pickle paths (no dataset_config)
    - Full pipeline: List of file paths/metadata (with dataset_config + process/format funcs)

    With dataset_config (Stage 1 and Full Pipeline):
    - Handles dataset-specific loading (Waymo TFRecords, nuPlan metadata)
    - Reuses DB connections for nuPlan/OpenScenes across batch
    - Applies dataset_config.convert_func automatically
    - Also applies process_func and format_func if provided (full pipeline)

    Without dataset_config (Stages 2/3):
    - Loads from pickle paths or uses raw scenario objects
    - Applies convert/process/format functions as specified

    Args:
        input_data: List of inputs (file paths, pickle paths, metadata dicts, or scenario objects)
        convert_func: Optional converter (raw → unified) - NOT used with dataset_config
        process_func: Optional processor (unified → unified)
        format_func: Optional formatter (unified → target)
        output_path: Optional path to save results
        target_format: Optional target format string (waymax, gpudrive, pufferdrive)
        dataset_config: Dataset configuration (enables dataset-specific optimizations)

    Returns:
        Dict with keys: 'successes', 'failures'
    """
    successes = 0
    failures = 0
    filtered = 0

    preprocess_func = getattr(dataset_config, "preprocess_func", None) if dataset_config else None

    # Preprocess input data (e.g., batch-read TFRecords for Waymo)
    try:
        list_scenarios = preprocess_func(input_data) if preprocess_func else input_data
    except Exception as e:
        logger.error(f"Preprocessing failed for batch: {e}")
        # Return early with all failures
        return {"successes": 0, "filtered": 0, "failures": len(input_data) if isinstance(input_data, list) else 1}

    for scenario in tqdm(list_scenarios, desc=" Processing scenarios", unit=" scenario", leave=False, position=1):
        try:
            if convert_func:
                scenario = convert_func(scenario)

            if process_func:
                scenario = process_func(scenario)

            # Validate scenario has required ID field
            if not isinstance(scenario, dict):
                logger.error(f"Scenario is not a dict, got type: {type(scenario).__name__}")
                raise ValueError(f"Invalid scenario type: {type(scenario).__name__}, expected dict")

            scenario_id = scenario.get("id")
            if not scenario_id:
                logger.error(f"Scenario missing 'id' field after processing. Scenario keys: {list(scenario.keys())}")
                raise ValueError("Invalid scenario: missing 'id' field")

            if format_func:
                scenario = format_func(scenario)

            if output_path:
                _save_result(scenario, scenario_id, output_path, format_func, target_format)

            successes += 1
        except OverpassDetectedException:
            filtered += 1
        except Exception as e:
            # Log error with scenario ID if available, otherwise log type info
            scenario_info = (
                scenario.get("id", f"<unknown, type: {type(scenario).__name__}>")
                if isinstance(scenario, dict)
                else f"<invalid type: {type(scenario).__name__}>"
            )
            logger.exception("Failed to process scenario %s: %s", scenario_info, str(e))
            failures += 1

    return {"successes": successes, "filtered": filtered, "failures": failures}


def _save_result(
    scenario: Any,
    scenario_id: str,
    output_path: str,
    format_func: Callable | None,
    target_format: str | None,
) -> None:
    """Save result based on format."""
    os.makedirs(output_path, exist_ok=True)

    # If no format function, save as pickle
    if format_func is None:
        save_pickle(scenario, os.path.join(output_path, f"{scenario_id}.pkl"))
        return

    # For Waymax, we need special handling (write to TFRecord)
    if target_format == FORMAT_WAYMAX:
        # This is a serialized TFExample
        tfrecord_file = os.path.join(output_path, f"{scenario_id}.tfrecord")
        from scenariomax.tf_utils import get_tensorflow

        tf = get_tensorflow()
        scenario = tf.train.Example(features=tf.train.Features(feature=scenario))

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            writer.write(scenario.SerializeToString())
    elif target_format == FORMAT_GPUDRIVE:
        # This is JSON format
        import json

        json_file = os.path.join(output_path, f"{scenario_id}.json")
        with open(json_file, "w") as f:
            json.dump(scenario, f, indent=2, cls=NumpyEncoder)
    elif target_format == FORMAT_PUFFERDRIVE:
        # Convert puffer dict to binary format
        from scenariomax.stage3_format.pufferdrive.binary_converter import puffer_dict_to_binary

        binary_data = puffer_dict_to_binary(scenario)
        binary_file = os.path.join(output_path, f"{scenario_id}.bin")
        with open(binary_file, "wb") as f:
            f.write(binary_data)
    else:
        # Fallback to pickle
        save_pickle(scenario, os.path.join(output_path, f"{scenario_id}.pkl"))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Raw → Unified
# ═══════════════════════════════════════════════════════════════════════════


def convert_raw_to_unified(
    datasets: dict[str, dict] | str,
    output_path: str,
    num_workers: int = 8,
    batch_size: int = 10,
) -> dict[str, Any]:
    """
    Stage 1: Convert raw dataset(s) to unified pickle format.

    Args:
        datasets: Dict mapping dataset names to config dicts (with 'path' and options)
                  OR single path string (auto-detected dataset)
        output_path: Output directory for unified pickles
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch

    Returns:
        Statistics dict
    """
    start_time = time.time()

    # Normalize datasets to dict
    if isinstance(datasets, str):
        datasets = {"auto": {"path": datasets}}

    logger.info(f"🚀 Stage 1: Converting {len(datasets)} dataset(s) → Unified")
    logger.info(f"   • Workers: {num_workers}, Batch size: {batch_size}")

    clean_and_create_output_directory(output_path)

    total_scenarios = 0
    total_filtered = 0
    total_errors = 0

    # Process each dataset
    for dataset_name, dataset_config in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset path and options from config
        dataset_path = dataset_config["path"]
        dataset_options = {k: v for k, v in dataset_config.items() if k != "path"}

        # Get dataset config from registry
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load file paths/metadata
        file_list = config.load_func(data_path=dataset_path, **dataset_options)
        logger.info(f"   • Found {len(file_list)} files")

        # Create batches
        file_batches = [file_list[i : i + batch_size] for i in range(0, len(file_list), batch_size)]
        logger.info(f"   • Created {len(file_batches)} batches")

        # Setup output path for this dataset
        dataset_output = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        # Process batches in parallel
        results = Parallel(n_jobs=num_workers)(
            delayed(worker_scenario_func)(
                input_data=batch,
                convert_func=config.convert_func,
                output_path=dataset_output,
                dataset_config=config,
            )
            for batch in tqdm(file_batches, desc="Processing batches", unit=" batch")
        )

        # Aggregate statistics
        successes = sum(r["successes"] for r in results)
        filtered = sum(r["filtered"] for r in results)
        failures = sum(r["failures"] for r in results)

        total_scenarios += successes
        total_filtered += filtered
        total_errors += failures

        logger.info(f"   ✅ Processed: {successes}, 🚫 Filtered: {filtered}, ❌ Errors: {failures}")

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 1 completed in {elapsed_time:.2f}s")

    return {
        "stage": "raw_to_unified",
        "datasets_processed": len(datasets),
        "total_scenarios": total_scenarios,
        "errors": total_errors,
        "filtered": total_filtered,
        "elapsed_time": elapsed_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Process Unified → Unified
# ═══════════════════════════════════════════════════════════════════════════


def process_unified_scenarios(
    input_path: str,
    output_path: str,
    processors: list[Callable] | list[str] | None = None,
    processor_configs: dict[str, dict] | None = None,
    num_workers: int = 8,
    batch_size: int = 10,
) -> dict[str, Any]:
    """
    Stage 2: Process unified scenarios (apply transformations).

    Args:
        input_path: Directory containing unified pickles
        output_path: Output directory
        processors: List of processor functions or names
        processor_configs: Processor configurations
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch (default: 10)

    Returns:
        Statistics dict
    """
    start_time = time.time()

    logger.info("🚀 Stage 2: Processing Unified Scenarios")
    logger.info(f"   • Processors: {len(processors) if processors else 0}")
    logger.info(f"   • Workers: {num_workers}, Batch size: {batch_size}")

    # Resolve processor names
    if processors:
        from scenariomax.stage2_process import apply_processors

        _apply_processors = partial(apply_processors, processor_names=processors, configs=processor_configs)
    else:
        _apply_processors = None

    # Get all pickle files
    pickle_files = sorted(glob.glob(os.path.join(input_path, "**/*.pkl"), recursive=True))

    logger.info(f"   • Found {len(pickle_files)} pickle files")

    clean_and_create_output_directory(output_path)

    # Create batches
    file_batches = [pickle_files[i : i + batch_size] for i in range(0, len(pickle_files), batch_size)]
    logger.info(f"   • Created {len(file_batches)} batches")

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(worker_scenario_func)(
            input_data=batch,
            convert_func=load_pickle,
            process_func=_apply_processors,
            output_path=output_path,
        )
        for batch in tqdm(file_batches, desc="Processing batches", unit=" batch")
    )

    # Aggregate statistics
    successes = sum(r["successes"] for r in results)
    filtered = sum(r["filtedred"] for r in results)
    failures = sum(r["failures"] for r in results)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 2 completed in {elapsed_time:.2f}s")
    logger.info(f"   ✅ Processed: {successes}, 🚫 Filtered: {filtered}, ❌ Errors: {failures}")

    return {
        "stage": "process_unified",
        "scenarios_processed": successes,
        "errors": failures,
        "filtered": filtered,
        "elapsed_time": elapsed_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Format Unified → Target Format
# ═══════════════════════════════════════════════════════════════════════════


def format_unified_to_target(
    input_path: str,
    output_path: str,
    format: str,
    num_workers: int = 8,
    batch_size: int = 10,
    processors: list[Callable] | list[str] | None = None,
    processor_configs: dict[str, dict] | None = None,
    format_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Stage 3: Format unified scenarios to target format.

    Args:
        input_path: Directory containing unified pickles
        output_path: Output directory
        format: Target format (waymax, gpudrive, pufferdrive)
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch (default: 10)
        processors: Optional processors to apply before formatting
        processor_configs: Processor configurations
        format_config: Format-specific configuration dict

    Returns:
        Statistics dict
    """
    start_time = time.time()

    # Validate format
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use {', '.join(sorted(SUPPORTED_FORMATS))}")

    logger.info(f"🚀 Stage 3: Formatting Unified → {format.upper()}")
    logger.info(f"   • Processors: {len(processors) if processors else 0}")
    logger.info(f"   • Workers: {num_workers}, Batch size: {batch_size}")

    if format_config is None:
        format_config = {}

    # Resolve processor names
    if processors:
        from scenariomax.stage2_process import apply_processors

        _apply_processors = partial(apply_processors, processor_names=processors, configs=processor_configs)
    else:
        _apply_processors = None

    # Create format function
    _format_func = get_format_function(format)

    # Wrap format function with format_config for pufferdrive
    if format == FORMAT_PUFFERDRIVE:
        _format_func = partial(
            _format_func,
            min_route_valid_points=format_config.get("min_route_valid_points", 0),
            route_check_timestep=format_config.get("route_check_timestep", 0),
        )

    # Get all pickle files
    pickle_files = sorted(glob.glob(os.path.join(input_path, "**/*.pkl"), recursive=True))

    logger.info(f"   • Found {len(pickle_files)} pickle files")

    clean_and_create_output_directory(output_path)

    # Create batches
    file_batches = [pickle_files[i : i + batch_size] for i in range(0, len(pickle_files), batch_size)]
    logger.info(f"   • Created {len(file_batches)} batches")

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(worker_scenario_func)(
            input_data=batch,
            convert_func=load_pickle,
            process_func=_apply_processors,
            format_func=_format_func,
            output_path=output_path,
            target_format=format,
        )
        for batch in tqdm(file_batches, desc="Formatting batches", unit=" batch")
    )

    # Aggregate statistics
    successes = sum(r["successes"] for r in results)
    filtered = sum(r["filtedred"] for r in results)
    failures = sum(r["failures"] for r in results)

    # Postprocess if needed (merge workers, shuffle, shard)
    if format == FORMAT_WAYMAX:
        _postprocess_waymax(output_path, format_config)
    elif format == FORMAT_GPUDRIVE:
        logger.info("✅ JSON files ready")
    elif format == FORMAT_PUFFERDRIVE:
        _postprocess_pufferdrive(output_path)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 3 completed in {elapsed_time:.2f}s")
    logger.info(f"   ✅ Processed: {successes}, 🚫 Filtered: {filtered}, ❌ Errors: {failures}")

    return {
        "stage": "unified_to_target",
        "format": format,
        "scenarios_processed": successes,
        "errors": failures,
        "filtered": filtered,
        "elapsed_time": elapsed_time,
    }


def _postprocess_waymax(output_path: str, format_config: dict) -> None:
    """Merge TFRecord files, shuffle, and optionally shard."""
    from scenariomax.stage3_format.waymax import postprocess

    logger.info("🔄 Merging TFRecord files")

    # Merge all .tfrecord files into one
    base_filename = format_config.get("base_filename", "training")

    # # Collect all subdirectories
    subdirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]

    logger.info(f"Found {len(subdirs)} dataset subdirectories: {subdirs}")

    # Collect all TFRecord files from subdirectories
    all_tfrecord_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(output_path, subdir)
        tfrecord_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".tfrecord")]
        all_tfrecord_files.extend(tfrecord_files)
        logger.info(f"  {subdir}: {len(tfrecord_files)} TFRecord files")

    if all_tfrecord_files:
        # Merge all files from all datasets
        merged_file = os.path.join(output_path, f"{base_filename}.tfrecord")
        logger.info(f"Merging {len(all_tfrecord_files)} files into {merged_file}")
        postprocess.merge_tfrecord_files(all_tfrecord_files, merged_file)
        postprocess.shuffle_tfrecord_file(merged_file)

        # Shard if requested
        num_shards = format_config.get("num_shards", 1)
        if num_shards > 1:
            from scenariomax.stage3_format.waymax import shard

            logger.info(f"Sharding into {num_shards} shards")
            shard.shard_tfrecord(
                src=output_path,
                filename=base_filename,
                num_threads=format_config.get("num_workers", 8),
                num_shards=num_shards,
            )


def _postprocess_pufferdrive(output_path: str) -> None:
    """Merge PufferDrive binary files from subdirectories and rename sequentially."""
    import shutil

    logger.info("🔄 Merging PufferDrive binary files")

    # Collect all subdirectories
    subdirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]

    if subdirs:
        # Collect from subdirectories (full pipeline case)
        logger.info(f"Found {len(subdirs)} dataset subdirectories: {subdirs}")
        all_binary_files = []
        for subdir in subdirs:
            subdir_path = os.path.join(output_path, subdir)
            binary_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".bin")]
            all_binary_files.extend(binary_files)
            logger.info(f"  {subdir}: {len(binary_files)} binary files")
    else:
        # Collect from output root (Stage 3 alone case)
        logger.info("No subdirectories found, processing binaries in output root")
        all_binary_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".bin")]

    if not all_binary_files:
        logger.info("⚠️  No binary files found")
        return

    # Sort files to ensure consistent ordering
    all_binary_files.sort()

    logger.info(f"Renaming {'and moving ' if subdirs else ''}{len(all_binary_files)} files to output root")

    # Rename and move files to output root
    for idx, src_file in enumerate(all_binary_files):
        dst_file = os.path.join(output_path, f"map_{idx:03d}.bin")
        if src_file != dst_file:  # Avoid self-rename
            shutil.move(src_file, dst_file)

    # Clean up empty subdirectories
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(output_path, subdir)
            if os.path.exists(subdir_path) and not os.listdir(subdir_path):
                os.rmdir(subdir_path)
                logger.info(f"  Removed empty directory: {subdir}")

    logger.info(f"✅ Puffer binaries ready: map_000.bin to map_{len(all_binary_files) - 1:03d}.bin")


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: All 3 Stages
# ═══════════════════════════════════════════════════════════════════════════


def run_all_pipeline(
    datasets: dict[str, dict] | str,
    output_path: str,
    format: str,
    processors: list[Callable] | list[str] | None = None,
    processor_configs: dict[str, dict] | None = None,
    num_workers: int = 8,
    batch_size: int = 10,
    format_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: Run all 3 stages together.

    Raw → Unified → Processed → Target Format

    Args:
        datasets: Dict mapping dataset names to config dicts (with 'path' and options)
                  OR single path string (auto-detected dataset)
        output_path: Output directory
        format: Target format (waymax, gpudrive, pufferdrive)
        processors: Optional processors to apply (e.g., validation, traffic_lights)
        processor_configs: Processor configurations
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch
        format_config: Format-specific configuration dict

    Returns:
        Combined statistics dict
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("🚀 FULL PIPELINE: Raw → Unified → Processed → Target")
    logger.info("=" * 80)

    # Normalize datasets to dict
    if isinstance(datasets, str):
        datasets = {"auto": {"path": datasets}}

    if format_config is None:
        format_config = {}

    # Otherwise, run all 3 stages in memory (one pass)
    logger.info("Mode: In-memory streaming (no intermediate files)")

    # Validate format
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use {', '.join(sorted(SUPPORTED_FORMATS))}")

    clean_and_create_output_directory(output_path)

    total_scenarios = 0
    total_filtered = 0
    total_errors = 0

    # Resolve processors once (not per dataset) to avoid closure issues
    if processors:
        from scenariomax.stage2_process import apply_processors

        _apply_processors = partial(apply_processors, processor_names=processors, configs=processor_configs)
    else:
        _apply_processors = None

    _format_func = get_format_function(format)

    # Wrap format function with format_config for pufferdrive
    if format == FORMAT_PUFFERDRIVE:
        _format_func = partial(
            _format_func,
            min_route_valid_points=format_config.get("min_route_valid_points", 0),
            route_check_timestep=format_config.get("route_check_timestep", 0),
        )

    # Process each dataset
    for dataset_name, dataset_config in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset path and options from config
        dataset_path = dataset_config["path"]
        dataset_options = {k: v for k, v in dataset_config.items() if k != "path"}

        # Get dataset config from registry
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw file paths/metadata (don't preprocess yet - let worker do it)
        file_list = config.load_func(data_path=dataset_path, **dataset_options)
        logger.info(f"   • Found {len(file_list)} files")

        # Create batches of file paths/metadata
        file_batches = [file_list[i : i + batch_size] for i in range(0, len(file_list), batch_size)]
        logger.info(f"   • Created {len(file_batches)} batches")

        # Setup output path for this dataset
        dataset_output = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        # Process all 3 stages in parallel using dataset_config
        results = Parallel(n_jobs=num_workers)(
            delayed(worker_scenario_func)(
                input_data=batch,
                convert_func=config.convert_func,
                process_func=_apply_processors,
                format_func=_format_func,
                output_path=dataset_output,
                target_format=format,
                dataset_config=config,
            )
            for batch in tqdm(file_batches, desc="Processing batches", unit=" batch")
        )

        # Aggregate statistics
        successes = sum(r["successes"] for r in results)
        filtered = sum(r["filtered"] for r in results)
        failures = sum(r["failures"] for r in results)

        total_scenarios += successes
        total_filtered += filtered
        total_errors += failures

        logger.info(f"   ✅ Processed: {successes}, 🚫 Filtered: {filtered}, ❌ Errors: {failures}")

    # Postprocess based on format
    if format == FORMAT_WAYMAX:
        _postprocess_waymax(output_path, format_config)
    elif format == FORMAT_GPUDRIVE:
        logger.info("✅ JSON files ready")  # No postprocessing needed for JSON format
    elif format == FORMAT_PUFFERDRIVE:
        _postprocess_pufferdrive(output_path)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✅ PIPELINE COMPLETED in {total_time:.2f}s")
    logger.info("=" * 80)

    return {
        "mode": "in_memory",
        "format": format,
        "scenarios_processed": total_scenarios,
        "filtered": total_filtered,
        "errors": total_errors,
        "total_time": total_time,
    }

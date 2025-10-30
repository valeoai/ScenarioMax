"""
Simplified 3-stage pipeline for dataset conversion.

Core concept: One universal function that processes a single scenario/file through
any combination of stages (convert, process, format). Parallelization handled by joblib.
"""

import os
import time
from collections.abc import Callable
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax import dataset_registry, logger_utils
from scenariomax.core.types import FORMAT_JSON, FORMAT_PUFFER, FORMAT_TFEXAMPLE, SUPPORTED_FORMATS
from scenariomax.core.utils import NumpyEncoder, get_format_function, load_pickle, save_pickle


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
        target_format: Optional target format string (tfexample, json, puffer)
        dataset_config: Dataset configuration (enables dataset-specific optimizations)

    Returns:
        Dict with keys: 'successes', 'failures'
    """
    successes = 0
    failures = 0

    preprocess_func = getattr(dataset_config, "preprocess_func", None) if dataset_config else None
    dataset_version = getattr(dataset_config, "version", None) if dataset_config else None

    list_scenarios = preprocess_func(input_data) if preprocess_func else input_data

    for scenario in tqdm(list_scenarios, desc="  Processing scenarios", unit=" scenario", leave=False):
        try:
            if convert_func:
                scenario = convert_func(scenario, dataset_version)

            if process_func:
                scenario = process_func(scenario)

            scenario_id = scenario["id"]

            if format_func:
                scenario = format_func(scenario)

            if output_path:
                _save_result(scenario, scenario_id, output_path, format_func, target_format)

            successes += 1
        except Exception:
            logger.exception("Failed to process scenario from %s", scenario)
            failures += 1

    return {"successes": successes, "failures": failures}


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

    # For TFExample, we need special handling (write to TFRecord)
    if target_format == FORMAT_TFEXAMPLE:
        # This is a serialized TFExample
        tfrecord_file = os.path.join(output_path, f"{scenario_id}.tfrecord")
        from scenariomax.tf_utils import get_tensorflow

        tf = get_tensorflow()
        scenario = tf.train.Example(features=tf.train.Features(feature=scenario))

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            writer.write(scenario.SerializeToString())
    elif target_format in [FORMAT_JSON, FORMAT_PUFFER]:
        # This is JSON or Puffer format
        import json

        json_file = os.path.join(output_path, f"{scenario_id}.json")
        with open(json_file, "w") as f:
            json.dump(scenario, f, indent=2, cls=NumpyEncoder)
    else:
        # Fallback to pickle
        save_pickle(scenario, os.path.join(output_path, f"{scenario_id}.pkl"))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Raw → Unified
# ═══════════════════════════════════════════════════════════════════════════


def convert_raw_to_unified(
    datasets: dict[str, str] | str,
    output_path: str,
    num_workers: int = 8,
    batch_size: int = 10,
    **kwargs,
) -> dict[str, Any]:
    """
    Stage 1: Convert raw dataset(s) to unified pickle format.

    Args:
        datasets: Dict mapping dataset names to paths OR single path string
        output_path: Output directory for unified pickles
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch
        **kwargs: Dataset-specific arguments

    Returns:
        Statistics dict
    """
    start_time = time.time()

    # Normalize datasets to dict
    if isinstance(datasets, str):
        datasets = {"auto": datasets}

    logger.info(f"🚀 Stage 1: Converting {len(datasets)} dataset(s) → Unified")
    logger.info(f"   • Workers: {num_workers}, Batch size: {batch_size}")

    os.makedirs(output_path, exist_ok=True)

    total_scenarios = 0
    total_errors = 0

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset config
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load file paths/metadata
        file_list = config.load_func(data_path=dataset_path, **kwargs)
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
        failures = sum(r["failures"] for r in results)

        total_scenarios += successes
        total_errors += failures

        logger.info(f"   ✅ Processed: {successes}, ❌ Errors: {failures}")

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 1 completed in {elapsed_time:.2f}s")

    return {
        "stage": "raw_to_unified",
        "datasets_processed": len(datasets),
        "total_scenarios": total_scenarios,
        "errors": total_errors,
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
    save_output: bool = True,
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
        save_output: Whether to save processed scenarios

    Returns:
        Statistics dict
    """
    start_time = time.time()

    logger.info("🚀 Stage 2: Processing Unified Scenarios")
    logger.info(f"   • Processors: {len(processors) if processors else 0}")
    logger.info(f"   • Workers: {num_workers}, Batch size: {batch_size}")
    logger.info(f"   • Save output: {save_output}")

    # Resolve processor names to functions
    if processors and isinstance(processors[0], str):
        from scenariomax.stage2_process import get_processors

        processors = get_processors(processors, configs=processor_configs)

    # Create process function that applies all processors
    def process_all(scenario):
        if processors:
            for processor_fn in processors:
                scenario = processor_fn(scenario)
        return scenario

    # Get all pickle files
    pickle_files = []
    for root, _, files in os.walk(input_path):
        for file in sorted(files):
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    logger.info(f"   • Found {len(pickle_files)} pickle files")

    if save_output:
        os.makedirs(output_path, exist_ok=True)

    # Create batches
    file_batches = [pickle_files[i : i + batch_size] for i in range(0, len(pickle_files), batch_size)]
    logger.info(f"   • Created {len(file_batches)} batches")

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(worker_scenario_func)(
            input_data=batch,
            convert_func=load_pickle,
            process_func=process_all,
            output_path=output_path if save_output else None,
        )
        for batch in tqdm(file_batches, desc="Processing batches", unit=" batch")
    )

    # Aggregate statistics
    successes = sum(r["successes"] for r in results)
    failures = sum(r["failures"] for r in results)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 2 completed in {elapsed_time:.2f}s")
    logger.info(f"   ✅ Processed: {successes}, ❌ Errors: {failures}")

    return {
        "stage": "process_unified",
        "scenarios_processed": successes,
        "errors": failures,
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
    **format_options,
) -> dict[str, Any]:
    """
    Stage 3: Format unified scenarios to target format.

    Args:
        input_path: Directory containing unified pickles
        output_path: Output directory
        format: Target format (tfexample, json, puffer)
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch (default: 10)
        processors: Optional processors to apply before formatting
        processor_configs: Processor configurations
        **format_options: Format-specific options

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

    # Resolve processor names
    if processors and isinstance(processors[0], str):
        from scenariomax.stage2_process import get_processors

        processors = get_processors(processors, configs=processor_configs)

    # Create process function
    def process_all(scenario):
        if processors:
            for processor_fn in processors:
                scenario = processor_fn(scenario)
        return scenario

    # Create format function
    _format_func = get_format_function(format)

    # Get all pickle files
    pickle_files = []
    for root, _, files in os.walk(input_path):
        for file in sorted(files):
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    logger.info(f"   • Found {len(pickle_files)} pickle files")

    os.makedirs(output_path, exist_ok=True)

    # Create batches
    file_batches = [pickle_files[i : i + batch_size] for i in range(0, len(pickle_files), batch_size)]
    logger.info(f"   • Created {len(file_batches)} batches")

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(worker_scenario_func)(
            input_data=batch,
            convert_func=load_pickle,
            process_func=process_all if processors else None,
            format_func=_format_func,
            output_path=output_path,
            target_format=format,
        )
        for batch in tqdm(file_batches, desc="Formatting batches", unit=" batch")
    )

    # Aggregate statistics
    successes = sum(r["successes"] for r in results)
    failures = sum(r["failures"] for r in results)

    # Postprocess if needed (merge workers, shuffle, shard)
    if format == FORMAT_TFEXAMPLE:
        _postprocess_tfexample(output_path, format_options)
    elif format == FORMAT_JSON:
        _postprocess_json(output_path)
    elif format == FORMAT_PUFFER:
        _postprocess_puffer(output_path)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 3 completed in {elapsed_time:.2f}s")
    logger.info(f"   ✅ Processed: {successes}, ❌ Errors: {failures}")

    return {
        "stage": "unified_to_target",
        "format": format,
        "scenarios_processed": successes,
        "errors": failures,
        "elapsed_time": elapsed_time,
    }


def _postprocess_tfexample(output_path: str, format_options: dict) -> None:
    """Merge TFRecord files, shuffle, and optionally shard."""
    from scenariomax.stage3_format.tfexample import postprocess

    logger.info("🔄 Merging TFRecord files")

    # Merge all .tfrecord files into one
    tfrecord_name = format_options.get("tfrecord_name", "training")

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
        merged_file = os.path.join(output_path, f"{tfrecord_name}.tfrecord")
        logger.info(f"Merging {len(all_tfrecord_files)} files into {merged_file}")
        postprocess.merge_tfrecord_files(all_tfrecord_files, merged_file)
        postprocess.shuffle_tfrecord_file(merged_file)

        # Shard if requested
        num_shards = format_options.get("shard", 1)
        if num_shards > 1:
            from scenariomax.stage3_format.tfexample import shard

            logger.info(f"Sharding into {num_shards} shards")
            shard.shard_tfrecord(
                src=output_path,
                filename=tfrecord_name,
                num_threads=format_options.get("num_workers", 8),
                num_shards=num_shards,
            )


def _postprocess_json(output_path: str) -> None:
    """Merge JSON files if needed."""
    logger.info("✅ JSON files ready")


def _postprocess_puffer(output_path: str) -> None:
    """Merge Puffer files if needed."""
    logger.info("✅ Puffer files ready")


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: All 3 Stages
# ═══════════════════════════════════════════════════════════════════════════


def run_all_pipeline(
    datasets: dict[str, str] | str,
    output_path: str,
    format: str,
    processors: list[Callable] | list[str] | None = None,
    processor_configs: dict[str, dict] | None = None,
    num_workers: int = 8,
    batch_size: int = 10,
    **kwargs,
) -> dict[str, Any]:
    """
    Full pipeline: Run all 3 stages together.

    Raw → Unified → Processed → Target Format

    Args:
        datasets: Dict mapping dataset names to paths OR single path string
        output_path: Output directory
        format: Target format (tfexample, json, puffer)
        processors: Optional processors to apply (e.g., validation, traffic_lights)
        processor_configs: Processor configurations
        num_workers: Number of parallel workers
        batch_size: Number of files per worker batch
        **kwargs: Dataset-specific arguments

    Returns:
        Combined statistics dict
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("🚀 FULL PIPELINE: Raw → Unified → Processed → Target")
    logger.info("=" * 80)

    # Normalize datasets to dict
    if isinstance(datasets, str):
        datasets = {"auto": datasets}

    # Otherwise, run all 3 stages in memory (one pass)
    logger.info("Mode: In-memory streaming (no intermediate files)")

    # Validate format
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use {', '.join(sorted(SUPPORTED_FORMATS))}")

    os.makedirs(output_path, exist_ok=True)

    total_scenarios = 0
    total_errors = 0

    # Resolve processors once (not per dataset) to avoid closure issues
    process_all = None
    if processors:
        from scenariomax.stage2_process import get_processors

        if isinstance(processors[0], str):
            processor_funcs = get_processors(processors, configs=processor_configs)
        else:
            processor_funcs = processors

        def process_all(scenario):
            for processor_fn in processor_funcs:
                scenario = processor_fn(scenario)
            return scenario

    _format_func = get_format_function(format)

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset config
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw file paths/metadata (don't preprocess yet - let worker do it)
        file_list = config.load_func(data_path=dataset_path, **kwargs)
        logger.info(f"   • Found {len(file_list)} files")

        # Create batches of file paths/metadata
        file_batches = [file_list[i : i + batch_size] for i in range(0, len(file_list), batch_size)]
        logger.info(f"   • Created {len(file_batches)} batches")

        # Setup output path for this dataset
        dataset_output = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        # Process all 3 stages in parallel using dataset_config
        # This enables dataset-specific optimizations (e.g., DB connection reuse)
        results = Parallel(n_jobs=num_workers)(
            delayed(worker_scenario_func)(
                input_data=batch,
                convert_func=config.convert_func,
                process_func=process_all,
                format_func=_format_func,
                output_path=dataset_output,
                target_format=format,
                dataset_config=config,
            )
            for batch in tqdm(file_batches, desc="Processing batches", unit=" batch")
        )

        # Aggregate statistics
        successes = sum(r["successes"] for r in results)
        failures = sum(r["failures"] for r in results)

        total_scenarios += successes
        total_errors += failures

        logger.info(f"   ✅ Processed: {successes}, ❌ Errors: {failures}")

    # Postprocess based on format
    if format == FORMAT_TFEXAMPLE:
        _postprocess_tfexample(output_path, kwargs)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✅ PIPELINE COMPLETED in {total_time:.2f}s")
    logger.info("=" * 80)

    return {
        "mode": "in_memory",
        "format": format,
        "scenarios_processed": total_scenarios,
        "errors": total_errors,
        "total_time": total_time,
    }

"""
Simplified 3-stage pipeline for dataset conversion.

Core concept: One universal function that processes a single scenario/file through
any combination of stages (convert, process, format). Parallelization handled by joblib.
"""

import os
import pickle
import time
from collections.abc import Callable
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax import dataset_registry, logger_utils
from scenariomax.core.types import FORMAT_JSON, FORMAT_PUFFER, FORMAT_TFEXAMPLE, SUPPORTED_FORMATS


logger = logger_utils.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Processing Function - Handles One Scenario/File
# ═══════════════════════════════════════════════════════════════════════════


def process_single_scenario(
    input_data: Any,
    convert_func: Callable | None = None,
    process_func: Callable | None = None,
    format_func: Callable | None = None,
    output_path: str | None = None,
    target_format: str | None = None,
) -> dict[str, Any]:
    """
    Process a single scenario through the pipeline.

    This is the core function that handles one scenario at a time.
    It can execute any combination of the 3 stages:
    - Stage 1 (convert): raw → unified
    - Stage 2 (process): unified → unified
    - Stage 3 (format): unified → target format

    Args:
        input_data: Raw scenario object OR path to pickle file
        convert_func: Optional Stage 1 converter (raw → unified)
        process_func: Optional Stage 2 processor (unified → unified)
        format_func: Optional Stage 3 formatter (unified → target)
        output_path: Optional path to save result
        target_format: Optional target format string (tfexample, json, puffer)

    Returns:
        Dict with keys: 'scenario' (result), 'success' (bool), 'error' (str if failed)
    """
    # Load if input is a file path
    if isinstance(input_data, str) and input_data.endswith(".pkl"):
        with open(input_data, "rb") as f:
            scenario = pickle.load(f)
    else:
        scenario = input_data

    # Stage 1: Convert raw → unified (optional)
    if convert_func:
        scenario = convert_func(scenario)

    # Stage 2: Process unified → unified (optional)
    if process_func:
        scenario = process_func(scenario)

    scenario_id = scenario["id"]

    # Stage 3: Format unified → target (optional)
    if format_func:
        scenario = format_func(scenario)

    # Save result if output_path specified
    if output_path:
        _save_result(scenario, scenario_id, output_path, format_func, target_format)

    return {"scenario": scenario, "success": True, "error": None}


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
        pkl_file = os.path.join(output_path, f"{scenario_id}.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(scenario, f)
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
            json.dump(scenario, f, indent=2)
    else:
        # Fallback to pickle
        pkl_file = os.path.join(output_path, f"{scenario_id}.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(scenario, f)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Raw → Unified
# ═══════════════════════════════════════════════════════════════════════════


def convert_raw_to_unified(
    datasets: dict[str, str] | str,
    output_path: str,
    num_workers: int = 8,
    **kwargs,
) -> dict[str, Any]:
    """
    Stage 1: Convert raw dataset(s) to unified pickle format.

    Args:
        datasets: Dict mapping dataset names to paths OR single path string
        output_path: Output directory for unified pickles
        num_workers: Number of parallel workers
        **kwargs: Dataset-specific arguments

    Returns:
        Statistics dict
    """
    start_time = time.time()

    # Normalize datasets to dict
    if isinstance(datasets, str):
        datasets = {"auto": datasets}

    logger.info(f"🚀 Stage 1: Converting {len(datasets)} dataset(s) → Unified")
    logger.info(f"   • Workers: {num_workers}")

    os.makedirs(output_path, exist_ok=True)

    total_scenarios = 0
    total_errors = 0

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset config
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw scenarios
        raw_scenarios = config.load_func(data_path=dataset_path, **kwargs)

        # Get count (special handling for Waymo)
        if dataset_name == "waymo":
            from scenariomax.stage1_convert.datasets.waymo.load import count_waymo_scenarios

            scenario_count = count_waymo_scenarios(raw_scenarios)
        else:
            scenario_count = len(raw_scenarios)

        logger.info(f"   • Found {scenario_count} scenarios")

        if config.preprocess_func:
            raw_scenarios = config.preprocess_func(raw_scenarios)

        tqdm_iterator = tqdm(raw_scenarios, desc="Worker pool", unit=" scenario", total=scenario_count)

        # Create convert function
        def convert_func(s):
            return config.convert_func(s, config.version)

        # Setup output path for this dataset
        dataset_output = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        # Process in parallel using joblib
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_scenario)(
                input_data=scenario,
                convert_func=convert_func,
                output_path=dataset_output,
            )
            for scenario in tqdm_iterator
        )

        # Count successes/failures
        successes = sum(1 for r in results if r["success"])
        failures = sum(1 for r in results if not r["success"])

        total_scenarios += successes
        total_errors += failures

        logger.info(f"   ✅ Processed: {successes}, ❌ Errors: {failures}")

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Stage 1 completed in {elapsed_time:.2f}s")

    return {
        "stage": "raw_to_unified",
        "scenarios_processed": total_scenarios,
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
        save_output: Whether to save processed scenarios

    Returns:
        Statistics dict
    """
    start_time = time.time()

    logger.info("🚀 Stage 2: Processing Unified Scenarios")
    logger.info(f"   • Processors: {len(processors) if processors else 0}")
    logger.info(f"   • Workers: {num_workers}")
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

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(process_single_scenario)(
            input_data=pkl_file,
            process_func=process_all,
            output_path=output_path if save_output else None,
        )
        for pkl_file in tqdm(pickle_files, desc="Processing", unit=" file")
    )

    # Count successes/failures
    successes = sum(1 for r in results if r["success"])
    failures = sum(1 for r in results if not r["success"])

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
    logger.info(f"   • Workers: {num_workers}")

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
    if format == FORMAT_TFEXAMPLE:
        from scenariomax.stage3_format.tfexample import convert_to_tfexample

        def format_func(s):
            return convert_to_tfexample.convert(s)
    elif format == FORMAT_JSON:
        from scenariomax.stage3_format.json import convert_to_json

        def format_func(s):
            return convert_to_json.convert(s)
    elif format == FORMAT_PUFFER:
        from scenariomax.stage3_format.puffer import convert_to_puffer

        def format_func(s):
            return convert_to_puffer.convert(s)

    # Get all pickle files
    pickle_files = []
    for root, _, files in os.walk(input_path):
        for file in sorted(files):
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    logger.info(f"   • Found {len(pickle_files)} pickle files")

    os.makedirs(output_path, exist_ok=True)

    # Process in parallel
    results = Parallel(n_jobs=num_workers)(
        delayed(process_single_scenario)(
            input_data=pkl_file,
            process_func=process_all if processors else None,
            format_func=format_func,
            output_path=output_path,
            target_format=format,
        )
        for pkl_file in tqdm(pickle_files, desc="Formatting", unit=" file")
    )

    # Count successes/failures
    successes = sum(1 for r in results if r["success"])
    failures = sum(1 for r in results if not r["success"])

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


def process_scenarios(
    datasets: dict[str, str] | str,
    output_path: str,
    format: str,
    processors: list[Callable] | list[str] | None = None,
    processor_configs: dict[str, dict] | None = None,
    num_workers: int = 8,
    save_intermediate: bool = False,
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
        save_intermediate: If True, save intermediate unified pickles
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

    # If save_intermediate, run 3 separate stages
    if save_intermediate:
        logger.info("Mode: Save intermediate pickles")

        # Stage 1: Convert
        unified_path = os.path.join(output_path, "_unified")
        stats1 = convert_raw_to_unified(datasets, unified_path, num_workers, **kwargs)

        # Stage 2: Process (optional)
        if processors:
            processed_path = os.path.join(output_path, "_processed")
            stats2 = process_unified_scenarios(
                unified_path,
                processed_path,
                processors,
                processor_configs,
                num_workers,
                save_output=True,
            )
            input_for_stage3 = processed_path
        else:
            stats2 = {"scenarios_processed": 0}
            input_for_stage3 = unified_path

        # Stage 3: Format
        stats3 = format_unified_to_target(
            input_for_stage3,
            output_path,
            format,
            num_workers,
            processors=None,
            **kwargs,
        )

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ PIPELINE COMPLETED in {total_time:.2f}s")
        logger.info("=" * 80)

        return {
            "mode": "save_intermediate",
            "stage1": stats1,
            "stage2": stats2,
            "stage3": stats3,
            "total_time": total_time,
        }

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

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset config
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw scenarios
        raw_scenarios = config.load_func(data_path=dataset_path, **kwargs)

        # Get count
        if dataset_name == "waymo":
            from scenariomax.stage1_convert.datasets.waymo.load import count_waymo_scenarios

            scenario_count = count_waymo_scenarios(raw_scenarios)
        else:
            scenario_count = len(raw_scenarios)

        logger.info(f"   • Found {scenario_count} scenarios")

        if config.preprocess_func:
            raw_scenarios = config.preprocess_func(raw_scenarios)

        tqdm_iterator = tqdm(raw_scenarios, desc="Worker pool", unit=" scenario", total=scenario_count)

        # Create convert function
        def convert_func(s):
            return config.convert_func(s, config.version)

        # Create format function
        if format == FORMAT_TFEXAMPLE:
            from scenariomax.stage3_format.tfexample import convert_to_tfexample

            def format_func(s):
                return convert_to_tfexample.convert(s)
        elif format == FORMAT_JSON:
            from scenariomax.stage3_format.json import convert_to_json

            def format_func(s):
                return convert_to_json.convert(s)
        elif format == FORMAT_PUFFER:
            from scenariomax.stage3_format.puffer import convert_to_puffer

            def format_func(s):
                return convert_to_puffer.convert(s)

        # Setup output path for this dataset
        dataset_output = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        # Process all 3 stages in parallel
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_scenario)(
                input_data=scenario,
                convert_func=convert_func,
                process_func=process_all,
                format_func=format_func,
                output_path=dataset_output,
                target_format=format,
            )
            for scenario in tqdm_iterator
        )

        # Count successes/failures
        successes = sum(1 for r in results if r["success"])
        failures = sum(1 for r in results if not r["success"])

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

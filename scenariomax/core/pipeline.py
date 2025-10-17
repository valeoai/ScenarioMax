"""
Simplified 3-stage pipeline for dataset conversion.

This module provides 4 main functions:
1. convert_raw_to_unified: Stage 1 - Raw dataset(s) → Unified pickles
2. process_unified_scenarios: Stage 2 - Unified → Processed unified
3. format_unified_to_target: Stage 3 - Unified → Target format (tfrecord/json)
4. process_scenarios: Full pipeline - Run all 3 stages together
"""

import os
import time
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from scenariomax import dataset_registry, logger_utils
from scenariomax.core import processor, write
from scenariomax.core.exceptions import DatasetLoadError


logger = logger_utils.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Raw → Unified
# ═══════════════════════════════════════════════════════════════════════════


def convert_raw_to_unified(
    datasets: dict[str, str] | str,
    output_path: str,
    num_workers: int = 8,
    validate: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Stage 1: Convert raw dataset(s) to unified pickle format.

    Args:
        datasets: Dict mapping dataset names to paths (e.g., {'waymo': '/path'})
                 Or single path string (will auto-detect dataset type)
        output_path: Output directory for unified pickles
        num_workers: Number of parallel workers
        validate: If True, perform soft validation on converted scenarios
        **kwargs: Dataset-specific arguments (num_files, split, etc.)

    Returns:
        Dict with conversion statistics

    Examples:
        # Single dataset
        convert_raw_to_unified(
            datasets={'waymo': '/data/waymo'},
            output_path='/output/unified',
            num_workers=16
        )

        # Multiple datasets
        convert_raw_to_unified(
            datasets={'waymo': '/data/waymo', 'nuplan': '/data/nuplan'},
            output_path='/output/unified',
            num_workers=16
        )
    """
    start_time = time.time()

    # Normalize input to dict format
    if isinstance(datasets, str):
        datasets = {"dataset": datasets}

    logger.info(f"🚀 Stage 1: Converting {len(datasets)} dataset(s) to unified format")

    # Setup output directory
    processor.setup_output_directory(output_path, clean=True)

    stats = {
        "stage": "raw_to_unified",
        "datasets_processed": 0,
        "total_scenarios": 0,
    }

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"📊 Processing dataset: {dataset_name}")

        if not os.path.exists(dataset_path):
            raise DatasetLoadError(dataset_name, dataset_path, "Path not found")

        # Get dataset configuration
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw scenarios
        scenarios, additional_args = _load_raw_scenarios(dataset_name, dataset_path, config, **kwargs)

        # Create output subdirectory for this dataset
        dataset_output = os.path.join(output_path, dataset_name)

        # Convert using existing write infrastructure (saves as pickles)
        write.write_to_directory(
            convert_func=config.convert_func,
            postprocess_func=None,  # None = save as pickle
            scenarios=scenarios,
            output_path=dataset_output,
            dataset_name=config.name,
            dataset_version=config.version,
            num_workers=num_workers,
            preprocess=config.preprocess_func,
            **additional_args,
        )

        scenario_count = _get_scenario_count(dataset_name, scenarios)
        stats["datasets_processed"] += 1
        stats["total_scenarios"] += scenario_count

        logger.info(f"✅ Completed {dataset_name}: {scenario_count} scenarios")

    stats["processing_time"] = time.time() - start_time
    logger.info(f"🏁 Stage 1 completed in {stats['processing_time']:.2f}s")
    logger.info(f"   • Datasets: {stats['datasets_processed']}")
    logger.info(f"   • Total scenarios: {stats['total_scenarios']}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Unified → Processed
# ═══════════════════════════════════════════════════════════════════════════


def process_unified_scenarios(
    input_path: str,
    output_path: str,
    processors: list[Callable] | None = None,
    num_workers: int = 8,
) -> dict[str, Any]:
    """
    Stage 2: Process unified scenarios with transformations.

    Apply transformations like adding traffic lights, filtering,
    interpolation, data cleaning, etc.

    Args:
        input_path: Directory containing unified pickle files
        output_path: Output directory for processed pickles
        processors: List of processor functions to apply
                   Each function should take and return a UnifiedScenario
        num_workers: Number of parallel workers

    Returns:
        Dict with processing statistics

    Examples:
        # Apply traffic light processing
        from scenariomax.stage2_process import enhance_scenarios

        process_unified_scenarios(
            input_path='/output/unified',
            output_path='/output/processed',
            processors=[enhance_scenarios],
            num_workers=8
        )
    """
    start_time = time.time()

    if processors is None:
        from scenariomax.stage2_process import enhance_scenarios

        processors = [enhance_scenarios]

    logger.info("🚀 Stage 2: Processing unified scenarios")
    logger.info(f"   • Processors: {len(processors)}")

    # Setup output directory
    processor.setup_output_directory(output_path, clean=True)

    # Load all pickle files
    scenarios = processor.load_pickle_files(input_path)

    logger.info(f"Applying {len(processors)} processor(s) to {len(scenarios)} scenarios")

    # Apply each processor function sequentially
    processed_scenarios = scenarios
    for i, processor_fn in enumerate(processors):
        logger.info(f"Applying processor {i + 1}/{len(processors)}: {processor_fn.__name__}")

        def apply_processor(scenario):
            return processor_fn(scenario)

        processed_scenarios = processor.process_batch_parallel(
            items=processed_scenarios,
            process_fn=apply_processor,
            num_workers=num_workers,
            desc=f"Processor {i + 1}/{len(processors)}",
        )

    # Save processed scenarios
    processor.save_pickle_files(processed_scenarios, output_path)

    stats = {
        "stage": "process_unified",
        "scenarios_processed": len(scenarios),
        "processors_applied": len(processors),
        "processing_time": time.time() - start_time,
    }

    logger.info(f"🏁 Stage 2 completed in {stats['processing_time']:.2f}s")
    logger.info(f"   • Scenarios processed: {stats['scenarios_processed']}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Unified → Target Format
# ═══════════════════════════════════════════════════════════════════════════


def format_unified_to_target(
    input_path: str,
    output_path: str,
    format: str,
    num_workers: int = 8,
    **format_options,
) -> dict[str, Any]:
    """
    Stage 3: Convert unified pickles to target format.

    Args:
        input_path: Directory containing unified pickle files
        output_path: Output directory for target format
        format: Target format ('tfexample' or 'json')
        num_workers: Number of parallel workers
        **format_options: Format-specific options
            - For tfexample: shard (int), tfrecord_name (str)
            - For json: (none currently)

    Returns:
        Dict with conversion statistics

    Examples:
        # Convert to TFRecord with sharding
        format_unified_to_target(
            input_path='/output/enhanced',
            output_path='/output/tfrecord',
            format='tfexample',
            shard=10,
            tfrecord_name='training',
            num_workers=8
        )

        # Convert to JSON for GPUDrive
        format_unified_to_target(
            input_path='/output/enhanced',
            output_path='/output/json',
            format='json',
            num_workers=8
        )
    """
    start_time = time.time()

    if format not in ["tfexample", "json"]:
        raise ValueError(f"Unsupported format: {format}. Use 'tfexample' or 'json'")

    logger.info(f"🚀 Stage 3: Converting unified → {format.upper()}")

    # Setup output directory
    processor.setup_output_directory(output_path, clean=True)

    # Get the appropriate postprocess function
    postprocess_func = _get_postprocess_func(format)

    # Check if we have dataset subdirectories or flat structure
    dataset_dirs = _find_dataset_dirs(input_path)

    if dataset_dirs:
        # Multi-dataset structure: process each dataset separately
        logger.info(f"Found {len(dataset_dirs)} dataset directories")
        total_scenarios = 0

        for dataset_name, dataset_path in dataset_dirs.items():
            logger.info(f"Processing dataset: {dataset_name}")

            # Create output subdirectory for this dataset
            dataset_output = os.path.join(output_path, dataset_name)

            # Load scenarios for this dataset
            scenarios = processor.load_pickle_files(dataset_path)
            total_scenarios += len(scenarios)

            # Convert using write infrastructure
            _convert_scenarios_to_format(
                scenarios=scenarios,
                output_path=dataset_output,
                postprocess_func=postprocess_func,
                num_workers=num_workers,
            )

        # Final postprocessing (merge workers, shuffle, shard)
        _final_postprocess(format, output_path, **format_options)

    else:
        # Flat structure: single dataset
        logger.info("Processing single dataset")
        scenarios = processor.load_pickle_files(input_path)
        total_scenarios = len(scenarios)

        # Convert using write infrastructure
        _convert_scenarios_to_format(
            scenarios=scenarios,
            output_path=output_path,
            postprocess_func=postprocess_func,
            num_workers=num_workers,
        )

        # For single dataset, still need to merge workers
        if format == "tfexample":
            from scenariomax.stage3_format.tfexample import postprocess

            tfrecord_name = format_options.get("tfrecord_name", "training")
            postprocess.merge_dataset_workers(output_path, tfrecord_name)

            # Shuffle the merged file
            merged_file = os.path.join(output_path, f"{tfrecord_name}.tfrecord")
            if os.path.exists(merged_file):
                postprocess.shuffle_tfrecord_file(merged_file)

                # Shard if requested
                num_shards = format_options.get("shard", 1)
                if num_shards > 1:
                    from scenariomax.stage3_format.tfexample import shard

                    logger.info(f"Sharding into {num_shards} shards")
                    shard.shard_tfrecord(
                        src=output_path,
                        filename=tfrecord_name,
                        num_threads=num_workers,
                        num_shards=num_shards,
                    )

        elif format == "json":
            from scenariomax.stage3_format.json import postprocess

            postprocess.merge_dataset_workers(output_path, os.path.basename(output_path))

    stats = {
        "stage": "unified_to_target",
        "format": format,
        "scenarios_processed": total_scenarios,
        "processing_time": time.time() - start_time,
    }

    logger.info(f"🏁 Stage 3 completed in {stats['processing_time']:.2f}s")
    logger.info(f"   • Format: {format}")
    logger.info(f"   • Scenarios processed: {stats['scenarios_processed']}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: Raw → Unified → Processed → Target
# ═══════════════════════════════════════════════════════════════════════════


def process_scenarios(
    datasets: dict[str, str] | str,
    output_path: str,
    format: str,
    processors: list[Callable] | None = None,
    num_workers: int = 8,
    save_intermediate: bool = False,
    validate: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Full pipeline: Run all 3 stages in sequence.

    Args:
        datasets: Dict mapping dataset names to paths (e.g., {'waymo': '/path'})
        output_path: Base output directory
        format: Target format ('tfexample' or 'json')
        processors: Optional list of processor functions for Stage 2
        num_workers: Number of parallel workers
        save_intermediate: If True, save intermediate pickles to disk.
                          If False, process everything in memory (saves disk space)
        validate: If True, perform soft validation on unified scenarios
        **kwargs: Additional arguments for stages

    Returns:
        Dict with combined statistics from all stages

    Examples:
        # Full pipeline with processing (in-memory, no intermediate saves)
        process_scenarios(
            datasets={'waymo': '/data/waymo'},
            output_path='/output',
            format='tfexample',
            processors=[enhance_scenarios],
            num_workers=16,
            save_intermediate=False,  # Default - saves disk space
            shard=10
        )

        # Full pipeline with intermediate pickle saves (for debugging)
        process_scenarios(
            datasets={'waymo': '/data/waymo', 'nuplan': '/data/nuplan'},
            output_path='/output',
            format='json',
            num_workers=16,
            save_intermediate=True  # Saves _unified and _processed dirs
        )
    """
    start_time = time.time()

    logger.info("🚀 Starting full 3-stage pipeline")
    logger.info(f"   • Datasets: {list(datasets.keys()) if isinstance(datasets, dict) else 'auto-detect'}")
    logger.info(f"   • Target format: {format}")
    logger.info(f"   • Processors: {len(processors) if processors else 0}")
    logger.info(f"   • Mode: {'Disk I/O' if save_intermediate else 'In-memory (streaming)'}")
    logger.info(f"   • Validation: {'enabled' if validate else 'disabled'}")

    if save_intermediate:
        # Disk-based pipeline: save intermediate pickles
        stage1_stats, stage2_stats, stage3_stats = _run_pipeline_with_disk_io(
            datasets=datasets,
            output_path=output_path,
            format=format,
            processors=processors,
            num_workers=num_workers,
            validate=validate,
            **kwargs,
        )
    else:
        # In-memory pipeline: no intermediate saves
        stage1_stats, stage2_stats, stage3_stats = _run_pipeline_in_memory(
            datasets=datasets,
            output_path=output_path,
            format=format,
            processors=processors,
            num_workers=num_workers,
            validate=validate,
            **kwargs,
        )

    # Combine statistics
    total_time = time.time() - start_time
    stats = {
        "pipeline": "full_3_stage",
        "mode": "disk_io" if save_intermediate else "in_memory",
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "stage3": stage3_stats,
        "total_time": total_time,
    }

    logger.info(f"🏁 Full pipeline completed in {total_time:.2f}s")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Execution Modes
# ═══════════════════════════════════════════════════════════════════════════


def _run_pipeline_with_disk_io(
    datasets: dict[str, str],
    output_path: str,
    format: str,
    processors: list[Callable] | None,
    num_workers: int,
    validate: bool = False,
    **kwargs,
) -> tuple[dict, dict | None, dict]:
    """Run pipeline with intermediate pickle saves (for debugging/inspection)."""
    import shutil

    # Create intermediate paths
    unified_path = os.path.join(output_path, "_unified")
    processed_path = os.path.join(output_path, "_processed") if processors else None
    final_path = os.path.join(output_path, format)

    # Stage 1: Raw → Unified
    stage1_stats = convert_raw_to_unified(
        datasets=datasets,
        output_path=unified_path,
        num_workers=num_workers,
        validate=validate,
        **kwargs,
    )

    # Stage 2: Unified → Processed (optional)
    stage2_stats = None
    if processors:
        stage2_stats = process_unified_scenarios(
            input_path=unified_path,
            output_path=processed_path,
            processors=processors,
            num_workers=num_workers,
        )
        input_for_stage3 = processed_path
    else:
        input_for_stage3 = unified_path

    # Stage 3: Unified/Processed → Target Format
    stage3_stats = format_unified_to_target(
        input_path=input_for_stage3,
        output_path=final_path,
        format=format,
        num_workers=num_workers,
        **kwargs,
    )

    # Cleanup intermediate directories
    logger.info("Cleaning up intermediate directories")
    if os.path.exists(unified_path):
        shutil.rmtree(unified_path)
    if processed_path and os.path.exists(processed_path):
        shutil.rmtree(processed_path)

    return stage1_stats, stage2_stats, stage3_stats


def _run_pipeline_in_memory(
    datasets: dict[str, str],
    output_path: str,
    format: str,
    processors: list[Callable] | None,
    num_workers: int,
    validate: bool = False,
    **kwargs,
) -> tuple[dict, dict | None, dict]:
    """
    Run pipeline with true in-memory per-worker processing.

    Each worker independently processes chunks of raw scenarios through:
    1. Load raw scenario from file path
    2. Convert to unified format (Stage 1)
    3. Apply processors (Stage 2 - optional)
    4. Format to target format (Stage 3)
    5. Save directly to target format

    NO intermediate pickle saves. Workers process in chunks to control memory.
    Each worker is responsible for opening/closing its own files.
    """
    start_time_total = time.time()

    # Normalize input to dict format
    if isinstance(datasets, str):
        datasets = {"dataset": datasets}

    logger.info(f"🚀 In-memory pipeline: {len(datasets)} dataset(s)")
    logger.info(f"   • Workers: {num_workers}")
    logger.info(f"   • Format: {format}")

    # Setup output directory for target format
    final_path = os.path.join(output_path, format)
    processor.setup_output_directory(final_path, clean=True)

    # Track statistics
    total_scenarios = 0
    datasets_processed = 0

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"📊 Processing dataset: {dataset_name}")

        if not os.path.exists(dataset_path):
            raise DatasetLoadError(dataset_name, dataset_path, "Path not found")

        # Create dataset-specific output subdirectory (for multi-dataset support)
        dataset_output = os.path.join(final_path, dataset_name)

        # Get dataset configuration
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw scenario file paths
        scenario_files, additional_args = _load_raw_scenarios(dataset_name, dataset_path, config, **kwargs)

        scenario_count = _get_scenario_count(dataset_name, scenario_files)
        logger.info(f"   • Total scenarios: {scenario_count}")

        # Process scenarios: Stage 1→2→3 in workers without intermediate saves
        _process_scenarios_stage123_parallel(
            scenario_files=scenario_files,
            config=config,
            processors=processors,
            format=format,
            output_path=dataset_output,
            num_workers=num_workers,
            additional_args=additional_args,
            validate=validate,
        )

        total_scenarios += scenario_count
        datasets_processed += 1
        logger.info(f"✅ Completed {dataset_name}")

    # Final postprocessing (merge workers, shuffle, shard)
    logger.info("🔄 Final postprocessing")
    postprocess_start = time.time()

    if format == "tfexample":
        from scenariomax.stage3_format.tfexample import postprocess

        tfrecord_name = kwargs.get("tfrecord_name", "training")
        postprocess.merge_multiple_datasets(final_path, f"{tfrecord_name}.tfrecord")

        # Shuffle the merged file
        merged_file = os.path.join(final_path, f"{tfrecord_name}.tfrecord")
        if os.path.exists(merged_file):
            postprocess.shuffle_tfrecord_file(merged_file)

            # Shard if requested
            num_shards = kwargs.get("shard", 1)
            if num_shards > 1:
                from scenariomax.stage3_format.tfexample import shard

                logger.info(f"Sharding into {num_shards} shards")
                shard.shard_tfrecord(
                    src=final_path,
                    filename=tfrecord_name,
                    num_threads=num_workers,
                    num_shards=num_shards,
                )

    elif format == "json":
        from scenariomax.stage3_format.json import postprocess

        postprocess.merge_dataset_workers(final_path, os.path.basename(final_path))

    postprocess_time = time.time() - postprocess_start

    # Build statistics
    total_time = time.time() - start_time_total

    stage1_stats = {
        "stage": "raw_to_unified",
        "datasets_processed": datasets_processed,
        "total_scenarios": total_scenarios,
        "processing_time": total_time - postprocess_time,
    }

    stage2_stats = None
    if processors:
        stage2_stats = {
            "stage": "process_unified",
            "scenarios_processed": total_scenarios,
            "processors_applied": len(processors),
            "processing_time": 0,  # Included in stage1_stats
        }

    stage3_stats = {
        "stage": "unified_to_target",
        "format": format,
        "scenarios_processed": total_scenarios,
        "processing_time": postprocess_time,
    }

    logger.info(f"🏁 In-memory pipeline completed in {total_time:.2f}s")

    return stage1_stats, stage2_stats, stage3_stats


def _process_scenarios_stage123_parallel(
    scenario_files: list[str],
    config: Any,
    processors: list[Callable] | None,
    format: str,
    output_path: str,
    num_workers: int,
    additional_args: dict,
    validate: bool = False,
) -> None:
    """
    Process scenarios in parallel through full Stage 1→2→3 pipeline.

    Each worker:
    1. Receives a chunk of FILE PATHS
    2. Loads raw scenarios from those files
    3. Converts to unified (Stage 1)
    4. Validates unified scenarios (optional)
    5. Applies processors (Stage 2 - optional)
    6. Formats to target format (Stage 3)
    7. Saves directly to target format (tfexample or json)

    NO intermediate pickle saves. Workers process in chunks to control memory.
    """
    from functools import partial

    from joblib import Parallel, delayed

    # Setup worker directories
    write._create_worker_directories(output_path, num_workers)

    # Distribute file paths to workers
    total_files = len(scenario_files)
    if total_files < num_workers:
        logger.info(f"Using {total_files} worker(s) as file count < worker count ({num_workers})")
        num_workers = total_files

    files_per_worker = total_files // num_workers
    output_directory_basename = os.path.basename(output_path)

    # Prepare worker arguments
    worker_arguments_list = []
    for worker_index in range(num_workers):
        batch_start = worker_index * files_per_worker
        batch_end = total_files if worker_index == num_workers - 1 else batch_start + files_per_worker

        worker_file_chunk = scenario_files[batch_start:batch_end]
        worker_output_dir = os.path.join(output_path, f"{output_directory_basename}_{worker_index}")

        worker_arguments_list.append(
            {
                "worker_index": worker_index,
                "file_chunk": worker_file_chunk,
                "config": config,
                "processors": processors,
                "format": format,
                "output_path": worker_output_dir,
                "additional_args": additional_args,
                "validate": validate,
            },
        )

        logger.debug(
            f"Worker {worker_index} assigned {len(worker_file_chunk)} files (indices {batch_start}:{batch_end})",
        )

    # Execute parallel processing
    logger.debug(f"Starting parallel Stage 1→2→3 processing with {num_workers} workers")

    worker_func = partial(_worker_process_stage123)

    with Parallel(n_jobs=num_workers) as parallel_executor:
        worker_results = parallel_executor(delayed(worker_func)(**worker_args) for worker_args in worker_arguments_list)

    # Check for failures
    for worker_index, success in enumerate(worker_results):
        if not success:
            from scenariomax.core.exceptions import WorkerProcessingError

            raise WorkerProcessingError(worker_index, "Worker failed during Stage 1→2→3 processing")


def _worker_process_stage123(
    worker_index: int,
    file_chunk: list[str],
    config: Any,
    processors: list[Callable] | None,
    format: str,
    output_path: str,
    additional_args: dict,
    validate: bool = False,
) -> bool:
    """
    Worker function that processes a chunk of files through Stage 1→2→3.

    This is the core of the in-memory pipeline:
    - Load raw scenario from file path
    - Convert to unified (Stage 1)
    - Validate unified scenario (optional)
    - Apply processors (Stage 2 - optional)
    - Format to target format (Stage 3)
    - Save directly to target format

    NO intermediate saves. Each scenario flows through all stages in memory.

    Args:
        worker_index: Index of this worker
        file_chunk: List of file paths to process
        config: Dataset configuration
        processors: Optional processor functions for Stage 2
        format: Target format ('tfexample' or 'json')
        output_path: Worker's output directory
        additional_args: Additional arguments for conversion
        validate: If True, perform soft validation on unified scenarios

    Returns:
        True if successful, False otherwise
    """
    from scenariomax.stage1_convert.datasets import utils as converter_utils

    memory_before = converter_utils.process_memory()
    logger.debug(f"Worker {worker_index} starting - Memory: {memory_before:.2f} MB")

    # Get preprocessing function for this dataset
    from scenariomax.core.write import default_preprocess_func

    preprocess_func = config.preprocess_func if config.preprocess_func else default_preprocess_func

    # Preprocess to get actual scenario iterator/generator
    scenarios = preprocess_func(file_chunk, worker_index)

    # Get scenario count (special handling for Waymo)
    from scenariomax.core.write import _get_scenario_count

    num_scenarios = _get_scenario_count(scenarios, file_chunk, config.name)

    logger.debug(f"Worker {worker_index} processing {num_scenarios} scenarios")

    # Setup progress tracking
    pbar = tqdm(desc=f"Worker {worker_index}", total=num_scenarios, unit=" scenario", leave=True)

    processed_count = 0
    filtered_count = 0
    error_count = 0
    validation_error_count = 0

    # Setup format-specific writer
    if format == "tfexample":
        from scenariomax.core.unified_scenario import UnifiedScenario
        from scenariomax.stage3_format.tfexample import convert_to_tfexample, exceptions
        from scenariomax.tf_utils import get_tensorflow

        tf = get_tensorflow()
        tf_record_file = os.path.join(output_path, "training.tfrecord")
        logger.debug(f"Worker {worker_index} writing to TFRecord: {tf_record_file}")

        try:
            with tf.io.TFRecordWriter(tf_record_file) as writer:
                for raw_scenario in scenarios:
                    try:
                        # Stage 1: Convert to unified
                        unified_scenario = config.convert_func(raw_scenario, config.version, **additional_args)

                        # Soft validation (optional)
                        if validate:
                            if not isinstance(unified_scenario, UnifiedScenario):
                                unified_scenario = UnifiedScenario.from_dict(unified_scenario)

                            is_valid, errors, warnings = unified_scenario.strict_validate()
                            if not is_valid:
                                validation_error_count += 1
                                logger.warning(
                                    f"Worker {worker_index} validation failed for scenario {unified_scenario.get('id', 'unknown')}:"
                                )
                                for error in errors[:3]:  # Show first 3 errors
                                    logger.warning(f"  - {error}")
                                if len(errors) > 3:
                                    logger.warning(f"  ... and {len(errors) - 3} more errors")
                                pbar.update(1)
                                continue

                        # Stage 2: Apply processors (optional)
                        if processors:
                            for processor_fn in processors:
                                unified_scenario = processor_fn(unified_scenario)

                        # Stage 3: Convert to TFExample and save
                        if not isinstance(unified_scenario, UnifiedScenario):
                            unified_scenario = UnifiedScenario.from_dict(unified_scenario)

                        dict_to_convert = convert_to_tfexample.convert(unified_scenario)
                        example = tf.train.Example(features=tf.train.Features(feature=dict_to_convert))

                        if example is not None:
                            writer.write(example.SerializeToString())
                            processed_count += 1

                    except (exceptions.OverpassException, exceptions.NotEnoughValidObjectsException):
                        filtered_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Worker {worker_index} failed to process scenario: {e}")

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "processed": processed_count,
                            "filtered": filtered_count,
                            "validation_errors": validation_error_count,
                            "errors": error_count,
                        },
                    )

        except Exception as e:
            logger.error(f"Worker {worker_index} encountered critical error: {e}")
            pbar.close()
            return False
        finally:
            pbar.close()

    elif format == "json":
        from scenariomax.core.unified_scenario import UnifiedScenario
        from scenariomax.stage3_format.json import convert_to_json

        json_file = os.path.join(output_path, "scenarios.json")
        logger.debug(f"Worker {worker_index} writing to JSON: {json_file}")

        scenarios_list = []

        try:
            for raw_scenario in scenarios:
                try:
                    # Stage 1: Convert to unified
                    unified_scenario = config.convert_func(raw_scenario, config.version, **additional_args)

                    # Soft validation (optional)
                    if validate:
                        if not isinstance(unified_scenario, UnifiedScenario):
                            unified_scenario = UnifiedScenario.from_dict(unified_scenario)

                        is_valid, errors, warnings = unified_scenario.strict_validate()
                        if not is_valid:
                            validation_error_count += 1
                            logger.warning(
                                f"Worker {worker_index} validation failed for scenario {unified_scenario.get('id', 'unknown')}:"
                            )
                            for error in errors[:3]:  # Show first 3 errors
                                logger.warning(f"  - {error}")
                            if len(errors) > 3:
                                logger.warning(f"  ... and {len(errors) - 3} more errors")
                            pbar.update(1)
                            continue

                    # Stage 2: Apply processors (optional)
                    if processors:
                        for processor_fn in processors:
                            unified_scenario = processor_fn(unified_scenario)

                    # Stage 3: Convert to GPUDrive JSON format
                    json_scenario = convert_to_json.convert(unified_scenario)
                    scenarios_list.append(json_scenario)
                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    logger.error(f"Worker {worker_index} failed to process scenario: {e}")

                pbar.update(1)
                pbar.set_postfix(
                    {"processed": processed_count, "validation_errors": validation_error_count, "errors": error_count}
                )

            # Save all scenarios to JSON file
            import json

            with open(json_file, "w") as f:
                json.dump(scenarios_list, f, indent=2)

        except Exception as e:
            logger.error(f"Worker {worker_index} encountered critical error: {e}")
            pbar.close()
            return False
        finally:
            pbar.close()

    else:
        logger.error(f"Unsupported format: {format}")
        pbar.close()
        return False

    memory_final = converter_utils.process_memory()

    logger.debug(f"Worker {worker_index} COMPLETED:")
    logger.debug(f"  ✅ Processed: {processed_count} scenarios")
    if format == "tfexample":
        logger.debug(f"  🔍 Filtered: {filtered_count} scenarios")
    if validate and validation_error_count > 0:
        logger.debug(f"  ⚠️  Validation errors: {validation_error_count} scenarios")
    logger.debug(f"  ❌ Errors: {error_count} scenarios")
    logger.debug(f"  📊 Memory: {memory_final:.2f} MB")
    logger.debug(f"  📁 Output: {output_path}")

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def _load_raw_scenarios(
    dataset_name: str,
    dataset_path: str,
    config: Any,
    **kwargs,
) -> tuple[list[Any], dict[str, Any]]:
    """Load raw scenarios for a dataset."""
    load_args = {"data_path": dataset_path}

    if dataset_name == "waymo":
        load_args["num"] = kwargs.get("num_files")
    elif dataset_name == "nuscenes":
        load_args["split"] = kwargs.get("split", "v1.0-trainval")
        load_args["num_workers"] = kwargs.get("num_workers", 8)
        scenarios, nuscs = config.load_func(**load_args)
        return scenarios, {"nuscs": nuscs}
    elif dataset_name == "nuplan":
        load_args["maps_path"] = os.getenv("NUPLAN_MAPS_ROOT")
        load_args["num_files"] = kwargs.get("num_files")
    elif dataset_name == "openscenes":
        load_args.update(
            {
                "data_root": os.getenv("NUPLAN_DATA_ROOT"),
                "nuplan_src": dataset_path,
                "maps_path": os.getenv("NUPLAN_MAPS_ROOT"),
                "metadata_src": kwargs.get("openscenes_metadata_src"),
                "num_files": kwargs.get("num_files"),
            },
        )

    scenarios = config.load_func(**load_args)
    return scenarios, {}


def _get_scenario_count(dataset_name: str, scenarios: list) -> int:
    """Get accurate scenario count (special handling for Waymo)."""
    if dataset_name == "waymo":
        from scenariomax.stage1_convert.datasets.waymo.load import count_waymo_scenarios

        return count_waymo_scenarios(scenarios)
    return len(scenarios)


def _get_postprocess_func(format: str) -> Callable:
    """Get postprocess function for target format."""
    if format == "tfexample":
        from scenariomax.stage3_format.tfexample import postprocess

        return postprocess.postprocess_tfexample
    elif format == "json":
        from scenariomax.stage3_format.json import postprocess

        return postprocess.postprocess_gpudrive
    else:
        raise ValueError(f"Unknown format: {format}")


def _find_dataset_dirs(input_path: str) -> dict[str, str] | None:
    """
    Check if input has dataset subdirectories (multi-dataset structure).

    Returns dict of {dataset_name: path} or None if flat structure.
    """
    if not os.path.exists(input_path):
        return None

    subdirs = {}
    has_pickle_files = False

    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path):
            # Check if this subdir contains pickle files
            has_pickles = any(f.endswith(".pkl") for root, _, files in os.walk(item_path) for f in files)
            if has_pickles:
                subdirs[item] = item_path
        elif item.endswith(".pkl"):
            has_pickle_files = True

    # If we have both subdirs and top-level pickles, treat as flat
    if has_pickle_files:
        return None

    # If we have dataset subdirs, return them
    if subdirs:
        return subdirs

    return None


def _convert_scenarios_to_format(
    scenarios: list[Any],
    output_path: str,
    postprocess_func: Callable,
    num_workers: int,
) -> None:
    """Convert scenarios to target format using write infrastructure."""

    # Create identity converter (scenarios are already unified)
    def identity_converter(scenario, version):
        return scenario

    # Use write infrastructure with postprocess function
    write.write_to_directory(
        convert_func=identity_converter,
        postprocess_func=postprocess_func,
        scenarios=scenarios,
        output_path=output_path,
        dataset_name="unified",
        dataset_version="pickle",
        num_workers=num_workers,
    )


def _final_postprocess(format: str, output_path: str, **kwargs) -> None:
    """Final postprocessing for multi-dataset outputs."""
    if format == "tfexample":
        from scenariomax.stage3_format.tfexample import postprocess, shard

        # Step 1: Merge workers for each dataset
        logger.info("🔄 Merging TFExample workers for each dataset")
        for dataset_name in os.listdir(output_path):
            dataset_dir = os.path.join(output_path, dataset_name)
            if os.path.isdir(dataset_dir):
                logger.info(f"Merging {dataset_name} workers")
                postprocess.merge_dataset_workers(dataset_dir, dataset_name)

        # Step 2: Merge multiple datasets into final file
        tfrecord_name = kwargs.get("tfrecord_name", "training")
        logger.info("🔄 Merging TFExample datasets")
        postprocess.merge_multiple_datasets(output_path, f"{tfrecord_name}.tfrecord")

        # Step 3: Shard if requested
        num_shards = kwargs.get("shard", 1)
        if num_shards > 1:
            logger.info(f"Sharding into {num_shards} shards")
            shard.shard_tfrecord(
                src=output_path,
                filename=tfrecord_name,
                num_threads=kwargs.get("num_workers", 8),
                num_shards=num_shards,
            )

    elif format == "json":
        from scenariomax.stage3_format.json import postprocess

        # Step 1: Merge workers for each dataset
        logger.info("🔄 Merging JSON workers for each dataset")
        for dataset_name in os.listdir(output_path):
            dataset_dir = os.path.join(output_path, dataset_name)
            if os.path.isdir(dataset_dir):
                logger.info(f"Merging {dataset_name} workers")
                postprocess.merge_dataset_workers(dataset_dir, dataset_name)

    logger.info("✅ Final postprocessing completed")

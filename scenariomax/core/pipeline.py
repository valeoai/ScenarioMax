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
    **kwargs,
) -> dict[str, Any]:
    """
    Stage 1: Convert raw dataset(s) to unified pickle format.

    Args:
        datasets: Dict mapping dataset names to paths (e.g., {'waymo': '/path'})
                 Or single path string (will auto-detect dataset type)
        output_path: Output directory for unified pickles
        num_workers: Number of parallel workers
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

            postprocess.merge_dataset_workers(output_path, os.path.basename(output_path))

            # Shuffle the merged file
            tfrecord_name = format_options.get("tfrecord_name", "training")
            merged_file = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}.tfrecord")
            if os.path.exists(merged_file):
                postprocess.shuffle_tfrecord_file(merged_file)

                # Shard if requested
                num_shards = format_options.get("shard", 1)
                if num_shards > 1:
                    from scenariomax.stage3_format.tfexample import shard

                    logger.info(f"Sharding into {num_shards} shards")
                    shard.shard_tfrecord(
                        src=os.path.dirname(output_path),
                        filename=os.path.basename(output_path),
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

    if save_intermediate:
        # Disk-based pipeline: save intermediate pickles
        stage1_stats, stage2_stats, stage3_stats = _run_pipeline_with_disk_io(
            datasets=datasets,
            output_path=output_path,
            format=format,
            processors=processors,
            num_workers=num_workers,
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
    **kwargs,
) -> tuple[dict, dict | None, dict]:
    """
    Run pipeline with batch-based in-memory processing.

    Processes scenarios in batches to avoid loading entire dataset into memory.
    Each batch flows through: Raw → Unified → Process → Format → Save
    This minimizes memory footprint while avoiding intermediate disk I/O.
    """
    batch_size = kwargs.get("batch_size", 100)  # Default: 100 scenarios per batch

    start_time_total = time.time()

    # Normalize input to dict format
    if isinstance(datasets, str):
        datasets = {"dataset": datasets}

    logger.info(f"🚀 In-memory batch pipeline: {len(datasets)} dataset(s)")
    logger.info(f"   • Batch size: {batch_size} scenarios")
    logger.info(f"   • Workers: {num_workers}")

    # Setup output directory for target format
    final_path = os.path.join(output_path, format)
    processor.setup_output_directory(final_path, clean=True)

    # Get the appropriate postprocess function for Stage 3
    postprocess_func = _get_postprocess_func(format)

    # Track statistics
    total_scenarios = 0
    datasets_processed = 0
    total_stage1_time = 0
    total_stage2_time = 0
    total_stage3_time = 0

    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"📊 Processing dataset: {dataset_name}")

        if not os.path.exists(dataset_path):
            raise DatasetLoadError(dataset_name, dataset_path, "Path not found")

        # Create dataset-specific output subdirectory (for multi-dataset support)
        dataset_output = os.path.join(final_path, dataset_name)

        # Get dataset configuration
        config = dataset_registry.get_dataset_config(dataset_name)

        # Load raw scenarios
        scenarios, additional_args = _load_raw_scenarios(dataset_name, dataset_path, config, **kwargs)

        # Apply preprocessing to get actual scenario generator/list
        from scenariomax.core.write import default_preprocess_func

        preprocess_func = config.preprocess_func if config.preprocess_func else default_preprocess_func
        preprocessed_scenarios = preprocess_func(scenarios, worker_index=0)

        # Convert to iterator for batch processing (works for both generators and lists)
        import itertools

        scenario_iterator = iter(preprocessed_scenarios)

        scenario_count = _get_scenario_count(dataset_name, scenarios)
        logger.info(f"   • Total scenarios: {scenario_count}")
        logger.info(f"   • Batches: {(scenario_count + batch_size - 1) // batch_size}")

        # Process scenarios in batches using itertools
        batch_num = 0
        total_batches = (scenario_count + batch_size - 1) // batch_size

        while True:
            # Get next batch from iterator
            batch = list(itertools.islice(scenario_iterator, batch_size))
            if not batch:
                break  # No more scenarios

            batch_num += 1
            logger.info(f"   • Batch {batch_num}/{total_batches} ({len(batch)} scenarios)")

            # Stage 1: Raw → Unified (in-memory, batch)
            stage1_start = time.time()

            # Convert scenarios in parallel (already preprocessed)
            def convert_scenario(scenario):
                return config.convert_func(scenario, config.version, **additional_args)

            unified_batch = processor.process_batch_parallel(
                items=batch,
                process_fn=convert_scenario,
                num_workers=num_workers,
                desc=f"Batch {batch_num}/{total_batches} - Converting",
            )
            stage1_time = time.time() - stage1_start
            total_stage1_time += stage1_time

            # Stage 2: Process (in-memory, batch) - optional
            stage2_time = 0
            if processors:
                stage2_start = time.time()
                for processor_fn in processors:

                    def apply_processor(scenario):
                        return processor_fn(scenario)

                    unified_batch = processor.process_batch_parallel(
                        items=unified_batch,
                        process_fn=apply_processor,
                        num_workers=num_workers,
                        desc=f"Batch {batch_num}/{total_batches} - {processor_fn.__name__}",
                    )
                stage2_time = time.time() - stage2_start
                total_stage2_time += stage2_time

            # Stage 3: Format → Save (in-memory, batch)
            stage3_start = time.time()
            _convert_scenarios_to_format(
                scenarios=unified_batch,
                output_path=dataset_output,  # Use dataset-specific subdirectory
                postprocess_func=postprocess_func,
                num_workers=num_workers,
            )
            stage3_time = time.time() - stage3_start
            total_stage3_time += stage3_time

            total_scenarios += len(batch)

            # Clear batch from memory
            del batch
            del unified_batch

        datasets_processed += 1
        logger.info(f"✅ Completed {dataset_name}")

    # Final postprocessing (merge workers, shuffle, shard)
    logger.info("🔄 Final postprocessing")
    postprocess_start = time.time()

    if format == "tfexample":
        from scenariomax.stage3_format.tfexample import postprocess

        postprocess.merge_dataset_workers(final_path, os.path.basename(final_path))

        # Shuffle the merged file
        tfrecord_name = kwargs.get("tfrecord_name", "training")
        merged_file = os.path.join(os.path.dirname(final_path), f"{os.path.basename(final_path)}.tfrecord")
        if os.path.exists(merged_file):
            postprocess.shuffle_tfrecord_file(merged_file)

            # Shard if requested
            num_shards = kwargs.get("shard", 1)
            if num_shards > 1:
                from scenariomax.stage3_format.tfexample import shard

                logger.info(f"Sharding into {num_shards} shards")
                shard.shard_tfrecord(
                    src=os.path.dirname(final_path),
                    filename=os.path.basename(final_path),
                    num_threads=num_workers,
                    num_shards=num_shards,
                )

    elif format == "json":
        from scenariomax.stage3_format.json import postprocess

        postprocess.merge_dataset_workers(final_path, os.path.basename(final_path))

    postprocess_time = time.time() - postprocess_start

    # Build statistics
    stage1_stats = {
        "stage": "raw_to_unified",
        "datasets_processed": datasets_processed,
        "total_scenarios": total_scenarios,
        "processing_time": total_stage1_time,
    }

    stage2_stats = None
    if processors:
        stage2_stats = {
            "stage": "process_unified",
            "scenarios_processed": total_scenarios,
            "processors_applied": len(processors),
            "processing_time": total_stage2_time,
        }

    stage3_stats = {
        "stage": "unified_to_target",
        "format": format,
        "scenarios_processed": total_scenarios,
        "processing_time": total_stage3_time + postprocess_time,
    }

    total_time = time.time() - start_time_total
    logger.info(f"🏁 Batch pipeline completed in {total_time:.2f}s")
    logger.info(f"   • Stage 1 (convert): {total_stage1_time:.2f}s")
    if processors:
        logger.info(f"   • Stage 2 (process): {total_stage2_time:.2f}s")
    logger.info(f"   • Stage 3 (format): {total_stage3_time:.2f}s")
    logger.info(f"   • Postprocess: {postprocess_time:.2f}s")

    return stage1_stats, stage2_stats, stage3_stats


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

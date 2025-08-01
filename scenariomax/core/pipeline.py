"""
Unified conversion pipeline supporting all dataset transformation use cases.

This module provides a single, streamlined interface for converting between
raw datasets, pickle format, and target formats (TFRecord/GPUDrive) with
minimal disk I/O and maximum code reuse.
"""

import os
import pickle
import shutil
import time
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from scenariomax import dataset_registry, logger_utils
from scenariomax.core import write
from scenariomax.core.exceptions import DatasetLoadError, UnsupportedFormatError
from scenariomax.enhancement import enhance_scenarios


logger = logger_utils.get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# Convenience API Functions
# ════════════════════════════════════════════════════════════════════════════════


def raw_to_pickle(
    dataset_name: str,
    dataset_path: str,
    output_path: str,
    enhancement: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Convert raw dataset to pickle format (use cases 1-2)."""
    return convert_dataset(
        source_type="raw",
        source_paths={dataset_name: dataset_path},
        target_format="pickle",
        output_path=output_path,
        enhancement=enhancement,
        **kwargs,
    )


def pickle_to_target(
    pickle_path: str,
    target_format: str,
    output_path: str,
    enhancement: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Convert pickle to target format (use cases 3-4)."""
    return convert_dataset(
        source_type="pickle",
        source_paths={"pickle": pickle_path},
        target_format=target_format,
        output_path=output_path,
        enhancement=enhancement,
        **kwargs,
    )


def raw_to_target(
    dataset_name: str,
    dataset_path: str,
    target_format: str,
    output_path: str,
    enhancement: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Convert raw dataset directly to target format (use cases 5-6)."""
    return convert_dataset(
        source_type="raw",
        source_paths={dataset_name: dataset_path},
        target_format=target_format,
        output_path=output_path,
        enhancement=enhancement,
        stream_mode=True,  # Always stream for raw->target
        **kwargs,
    )


def convert_dataset(
    source_type: str,
    source_paths: dict[str, str],
    target_format: str,
    output_path: str,
    enhancement: bool = False,
    stream_mode: bool = True,
    num_workers: int = 8,
    **kwargs,
) -> dict[str, Any]:
    """
    Unified dataset conversion pipeline supporting all use cases.

    Use Case 1: Raw → Pickle: source_type='raw', target_format='pickle'
    Use Case 2: Raw → Pickle + Enhancement: source_type='raw', target_format='pickle', enhancement=True
    Use Case 3: Pickle → Target: source_type='pickle', target_format='tfexample/gpudrive'
    Use Case 4: Pickle → Enhanced → Target: source_type='pickle', target_format='tfexample/gpudrive', enhancement=True
    Use Case 5: Raw → Target (streaming): source_type='raw', target_format='tfexample/gpudrive', stream_mode=True
    Use Case 6: Raw → Enhanced → Target (streaming): source_type='raw', target_format='tfexample/gpudrive', enhancement=True, stream_mode=True

    Args:
        source_type: 'raw' or 'pickle'
        source_paths: Dict mapping dataset names to paths (e.g., {'waymo': '/path/to/waymo'})
        target_format: 'pickle', 'tfexample', or 'gpudrive'
        output_path: Output directory
        enhancement: Apply scenario enhancement
        stream_mode: Process without intermediate disk saves (for raw→target)
        num_workers: Number of parallel workers
        **kwargs: Additional arguments passed to dataset loaders

    Returns:
        Dict with conversion statistics
    """  # noqa: E501
    start_time = time.time()

    # Initialize pipeline
    _log_pipeline_start(source_type, target_format, stream_mode, enhancement)
    _validate_inputs(source_type, source_paths, target_format)
    _setup_output_directory(output_path)
    stats = _initialize_conversion_stats(source_type, target_format, enhancement, stream_mode)

    # Process all datasets
    stats = _process_all_datasets(
        source_paths,
        stats,
        source_type,
        target_format,
        output_path,
        enhancement,
        stream_mode,
        num_workers,
        **kwargs,
    )

    # Finalize pipeline
    if stats["datasets_processed"] > 0 and target_format != "pickle":
        _final_postprocess(target_format, output_path, **kwargs)

    _log_pipeline_completion(stats, start_time)
    return stats


def _process_single_dataset(
    dataset_name: str,
    dataset_path: str,
    source_type: str,
    target_format: str,
    output_path: str,
    enhancement: bool,
    stream_mode: bool,
    num_workers: int,
    **kwargs,
) -> int:
    """Process a single dataset through the unified pipeline."""

    if source_type == "pickle":
        # Load from pickle and convert
        return _process_from_pickle(dataset_path, target_format, output_path, enhancement, **kwargs)

    elif source_type == "raw":
        if target_format == "pickle":
            # Raw → Pickle (with optional enhancement)
            return _process_raw_to_pickle(dataset_name, dataset_path, output_path, enhancement, num_workers, **kwargs)
        else:
            # Raw → Target (streaming or with pickle intermediate)
            return _process_raw_to_target(
                dataset_name,
                dataset_path,
                target_format,
                output_path,
                enhancement,
                stream_mode,
                num_workers,
                **kwargs,
            )

    else:
        raise ValueError(f"Invalid source_type: {source_type}")


def _process_from_pickle(pickle_path: str, target_format: str, output_path: str, enhancement: bool, **kwargs) -> int:
    """Process scenarios from pickle format (use cases 3-4)."""
    logger.info(f"📋 Route: pickle → {target_format}")

    # Load unified scenarios
    scenarios = _load_pickle_scenarios(pickle_path)

    # Apply enhancement if requested
    if enhancement:
        scenarios = _apply_enhancement(scenarios)

    # Convert to target format
    if target_format == "pickle":
        # Just copy/save enhanced pickle
        _save_scenarios_as_pickle(scenarios, output_path)
    else:
        _convert_to_target_format(scenarios, target_format, output_path, "pickle")

    return len(scenarios)


def _process_raw_to_pickle(
    dataset_name: str,
    dataset_path: str,
    output_path: str,
    enhancement: bool,
    num_workers: int,
    **kwargs,
) -> int:
    """Process raw dataset to pickle format (use cases 1-2)."""
    logger.info(f"📋 Route: {dataset_name} → pickle")

    # Load raw scenarios
    scenarios, additional_args = _load_raw_scenarios(dataset_name, dataset_path, **kwargs)

    # Get dataset config
    config = dataset_registry.get_dataset_config(dataset_name)

    # Define conversion pipeline
    # Raw → Unified → Enhanced (?) → Pickle
    convert_func = _create_enhanced_converter(config.convert_func) if enhancement else config.convert_func

    # Use existing parallel processing infrastructure
    write.write_to_directory(
        convert_func=convert_func,
        postprocess_func=None,  # Save as pickle
        scenarios=scenarios,
        output_path=output_path,
        dataset_name=config.name,
        dataset_version=config.version,
        num_workers=num_workers,
        preprocess=config.preprocess_func,
        **additional_args,
    )

    return _get_scenario_count(dataset_name, scenarios)


def _process_raw_to_target(
    dataset_name: str,
    dataset_path: str,
    target_format: str,
    output_path: str,
    enhancement: bool,
    stream_mode: bool,
    num_workers: int,
    **kwargs,
) -> int:
    """Process raw dataset to target format (use cases 5-6)."""

    if stream_mode:
        logger.info(f"📋 Route: {dataset_name} → {target_format}")
        return _process_streaming(
            dataset_name,
            dataset_path,
            target_format,
            output_path,
            enhancement,
            num_workers,
            **kwargs,
        )
    else:
        logger.info(f"📋 Route: {dataset_name} → pickle → {target_format}")
        # Raw → Pickle → Target (two-stage)
        temp_pickle_path = output_path + "_temp_pickle"

        try:
            # Stage 1: Raw → Pickle
            scenario_count = _process_raw_to_pickle(
                dataset_name,
                dataset_path,
                temp_pickle_path,
                enhancement,
                num_workers,
                **kwargs,
            )

            # Stage 2: Pickle → Target
            _process_from_pickle(temp_pickle_path, target_format, output_path, False, **kwargs)

            return scenario_count

        finally:
            # Cleanup temporary pickle
            if os.path.exists(temp_pickle_path):
                shutil.rmtree(temp_pickle_path)


def _process_streaming(
    dataset_name: str,
    dataset_path: str,
    target_format: str,
    output_path: str,
    enhancement: bool,
    num_workers: int,
    **kwargs,
) -> int:
    """Stream processing: Raw → Unified → Enhanced → Target (in memory)."""

    # Load raw scenarios
    scenarios, additional_args = _load_raw_scenarios(dataset_name, dataset_path, **kwargs)

    # Get dataset config and target postprocess function
    config = dataset_registry.get_dataset_config(dataset_name)
    postprocess_func = _get_target_postprocess_func(target_format)

    # Create simple converter pipeline (unified format only)
    convert_func = _create_enhanced_converter(config.convert_func) if enhancement else config.convert_func

    # Use existing parallel processing with postprocess that handles target format conversion
    write.write_to_directory(
        convert_func=convert_func,
        postprocess_func=postprocess_func,
        scenarios=scenarios,
        output_path=output_path,
        dataset_name=config.name,
        dataset_version=config.version,
        num_workers=num_workers,
        preprocess=config.preprocess_func,
        **additional_args,
    )

    return _get_scenario_count(dataset_name, scenarios)


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline Setup and Coordination Functions
# ════════════════════════════════════════════════════════════════════════════════


def _log_pipeline_start(source_type: str, target_format: str, stream_mode: bool, enhancement: bool) -> None:
    """Log pipeline initialization information."""
    logger.info(f"🚀 Starting unified conversion: {source_type} → {target_format}")
    logger.info(f"Stream mode: {stream_mode}, Enhancement: {enhancement}")


def _setup_output_directory(output_path: str) -> None:
    """Setup clean output directory, removing existing if present."""
    if os.path.exists(output_path):
        logger.info(f"🗑️ Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=False)
    logger.info(f"📁 Created clean output directory: {output_path}")


def _initialize_conversion_stats(
    source_type: str,
    target_format: str,
    enhancement: bool,
    stream_mode: bool,
) -> dict[str, Any]:
    """Initialize conversion statistics tracking."""
    return {
        "source_type": source_type,
        "target_format": target_format,
        "enhancement": enhancement,
        "stream_mode": stream_mode,
        "datasets_processed": 0,
        "total_scenarios": 0,
        "processing_time": 0,
    }


def _process_all_datasets(
    source_paths: dict[str, str],
    stats: dict[str, Any],
    source_type: str,
    target_format: str,
    output_path: str,
    enhancement: bool,
    stream_mode: bool,
    num_workers: int,
    **kwargs,
) -> dict[str, Any]:
    """Process all datasets and update statistics."""
    for dataset_name, dataset_path in source_paths.items():
        logger.info(f"📊 Processing {dataset_name}")

        dataset_output = os.path.join(output_path, dataset_name)
        scenario_count = _process_single_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            source_type=source_type,
            target_format=target_format,
            output_path=dataset_output,
            enhancement=enhancement,
            stream_mode=stream_mode,
            num_workers=num_workers,
            **kwargs,
        )

        stats["datasets_processed"] += 1
        stats["total_scenarios"] += scenario_count
        logger.info(f"✅ Completed {dataset_name}: {scenario_count} scenarios")

    return stats


def _log_pipeline_completion(stats: dict[str, Any], start_time: float) -> None:
    """Log pipeline completion statistics."""
    stats["processing_time"] = time.time() - start_time
    logger.info(f"🏁 Pipeline completed in {stats['processing_time']:.2f}s")
    logger.info(f"   • Datasets: {stats['datasets_processed']}")
    logger.info(f"   • Total scenarios: {stats['total_scenarios']}")


# ════════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════════


def _validate_inputs(source_type: str, source_paths: dict[str, str], target_format: str):
    """Validate pipeline inputs."""
    if source_type not in ["raw", "pickle"]:
        raise ValueError(f"Invalid source_type: {source_type}")

    if target_format not in ["pickle", "tfexample", "gpudrive"]:
        raise UnsupportedFormatError(target_format, ["pickle", "tfexample", "gpudrive"])

    if not source_paths:
        raise ValueError("No source paths provided")

    for dataset_name, path in source_paths.items():
        if not os.path.exists(path):
            raise DatasetLoadError(dataset_name, path, "Source path not found")


def _load_pickle_scenarios(pickle_path: str) -> list[dict[str, Any]]:
    """Load scenarios from pickle directory."""
    scenarios = []

    for root, _, files in os.walk(pickle_path):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    scenario = pickle.load(f)
                    scenarios.append(scenario)

    logger.info(f"Loaded {len(scenarios)} scenarios from pickle")
    return scenarios


def _load_raw_scenarios(dataset_name: str, dataset_path: str, **kwargs) -> tuple:
    """Load raw scenarios for a dataset."""
    config = dataset_registry.get_dataset_config(dataset_name)

    # Prepare load arguments based on dataset
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


def _apply_enhancement(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply enhancement to scenarios."""
    logger.info("🔄 Applying scenario enhancement")
    enhanced = [enhance_scenarios(scenario) for scenario in scenarios]
    logger.info(f"✅ Enhanced {len(enhanced)} scenarios")
    return enhanced


def _save_scenarios_as_pickle(scenarios: list[dict[str, Any]], output_path: str):
    """Save scenarios as pickle files."""
    os.makedirs(output_path, exist_ok=True)
    worker_dir = os.path.join(output_path, f"{os.path.basename(output_path)}_0")
    os.makedirs(worker_dir, exist_ok=True)

    for i, scenario in enumerate(scenarios):
        pickle_file = os.path.join(worker_dir, f"scenario_{i:06d}.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(scenario, f)


def _convert_to_target_format(scenarios: list[dict[str, Any]], target_format: str, output_path: str, dataset_name: str):
    """Convert scenarios to target format using the proper postprocess functions."""
    # This function is now only used for pickle->target conversion
    # Use the actual postprocess functions to avoid duplication

    if target_format == "tfexample":
        from scenariomax.unified_to_tfexample import postprocess

        # Create a dummy converter that returns scenarios as-is (already unified)
        def identity_converter(scenario, version):
            return scenario

        # Create single worker directory
        worker_dir = os.path.join(output_path, f"{os.path.basename(output_path)}_0")
        os.makedirs(worker_dir, exist_ok=True)

        pbar = tqdm(desc=f"Converting to {target_format.upper()}", total=len(scenarios))

        postprocess.postprocess_tfexample(
            output_path=worker_dir,
            worker_index=0,
            scenarios=scenarios,
            convert_func=identity_converter,
            dataset_version="pickle",
            dataset_name=dataset_name,
            pbar=pbar,
            process_scenario_func=lambda scenario, convert_func, dataset_version, dataset_name, **kwargs: (
                scenario,
                f"scenario_{hash(str(scenario))}.tmp",
            ),
        )

        pbar.close()

    elif target_format == "gpudrive":
        from scenariomax.unified_to_gpudrive import postprocess

        # Create a dummy converter that returns scenarios as-is (already unified)
        def identity_converter(scenario, version):
            return scenario

        # Create single worker directory
        worker_dir = os.path.join(output_path, f"{os.path.basename(output_path)}_0")
        os.makedirs(worker_dir, exist_ok=True)

        pbar = tqdm(desc=f"Converting to {target_format.upper()}", total=len(scenarios))

        postprocess.postprocess_gpudrive(
            output_path=worker_dir,
            worker_index=0,
            scenarios=scenarios,
            convert_func=identity_converter,
            dataset_version="pickle",
            dataset_name=dataset_name,
            pbar=pbar,
            process_scenario_func=lambda scenario, convert_func, dataset_version, dataset_name, **kwargs: (
                scenario,
                f"scenario_{hash(str(scenario))}.tmp",
            ),
        )

        pbar.close()


def enhanced_converter(base_convert_func: Callable):
    """Create a converter that applies enhancement after conversion."""

    def converter(scenario, version):
        unified_scenario = base_convert_func(scenario, version)
        return enhance_scenarios(unified_scenario)

    return converter


def _create_enhanced_converter(base_convert_func: Callable) -> Callable:
    """Create converter that applies enhancement after conversion."""
    return enhanced_converter(base_convert_func)


def _get_target_postprocess_func(target_format: str) -> Callable | None:
    """Get postprocessing function for target format."""
    if target_format == "tfexample":
        from scenariomax.unified_to_tfexample import postprocess

        return postprocess.postprocess_tfexample
    elif target_format == "gpudrive":
        from scenariomax.unified_to_gpudrive import postprocess

        return postprocess.postprocess_gpudrive
    return None


def _get_scenario_count(dataset_name: str, scenarios: list) -> int:
    """Get accurate scenario count for dataset."""
    if dataset_name == "waymo":
        from scenariomax.raw_to_unified.datasets.waymo.load import count_waymo_scenarios

        return count_waymo_scenarios(scenarios)
    return len(scenarios)


def _final_postprocess(target_format: str, output_path: str, **kwargs):
    """Final postprocessing for multi-dataset outputs."""
    if target_format == "tfexample":
        from scenariomax.unified_to_tfexample import postprocess, shard_tfexample

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
            shard_tfexample.shard_tfrecord(
                src=output_path,
                filename=tfrecord_name,
                num_threads=kwargs.get("num_workers", 8),
                num_shards=num_shards,
            )

    elif target_format == "gpudrive":
        from scenariomax.unified_to_gpudrive import postprocess

        # Step 1: Merge workers for each dataset
        logger.info("🔄 Merging GPUDrive workers for each dataset")
        for dataset_name in os.listdir(output_path):
            dataset_dir = os.path.join(output_path, dataset_name)
            if os.path.isdir(dataset_dir):
                logger.info(f"Merging {dataset_name} workers")
                postprocess.merge_dataset_workers(dataset_dir, dataset_name)

    logger.info("✅ Final postprocessing completed")

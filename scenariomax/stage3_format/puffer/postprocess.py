"""
Postprocessing functions for Puffer format conversion.

Handles merging worker outputs and saving scenarios as individual JSON files.
Each scenario is saved as a separate JSON file named {scenario_id}.json.
"""

import json
import os
import shutil
from collections.abc import Callable, Generator, Iterable
from typing import Any

from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.stage3_format.puffer import convert_to_puffer


logger = logger_utils.get_logger(__name__)


def postprocess_puffer(
    output_path: str,
    worker_index: int,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    pbar: tqdm,
    process_scenario_func: Callable,
    **kwargs,
) -> None:
    """
    Process scenarios and write them as individual Puffer JSON files.

    Each scenario is saved as a separate JSON file named {scenario_id}.json.

    Args:
        output_path: Path to output directory
        worker_index: Worker index for parallel processing
        scenarios: List or generator of scenarios to process
        convert_func: Function to convert scenarios
        dataset_version: Dataset version
        dataset_name: Dataset name
        pbar: Progress bar instance
        process_scenario_func: Function to process individual scenarios
        **kwargs: Additional keyword arguments
    """
    logger.debug(f"Worker {worker_index} writing to Puffer format: {output_path}")
    processed_count = 0

    for scenario in scenarios:
        try:
            unified_scenario = process_scenario_func(
                scenario,
                convert_func,
                dataset_version,
                dataset_name,
                **kwargs,
            )

            if not isinstance(unified_scenario, UnifiedScenario):
                unified_scenario = UnifiedScenario.from_dict(unified_scenario)

            # Convert to Puffer format (already JSON-serializable)
            puffer_scenario = convert_to_puffer.convert(unified_scenario)

            if puffer_scenario is not None:
                # Save each scenario as individual JSON file
                scenario_id = puffer_scenario.get("scenario_id", f"scenario_{processed_count}")
                json_file_path = os.path.join(output_path, f"{scenario_id}.json")

                with open(json_file_path, "w") as f:
                    json.dump(puffer_scenario, f, indent=2)

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count})
        except Exception as e:
            logger.error(f"Worker {worker_index} failed to process scenario: {e!s}")
            pbar.close()
            raise e

    logger.debug(f"Worker {worker_index} saved {processed_count} scenarios to {output_path}")


def merge_dataset_workers(dataset_dir: str, dataset_name: str) -> None:
    """
    Merge JSON files from worker subdirectories into the parent directory.

    Each scenario JSON file is moved from worker subdirectories to the main dataset directory.

    Args:
        dataset_dir: Directory containing worker subdirectories with JSON files
        dataset_name: Name of the dataset (for logging)
    """
    json_files = []

    # Look for worker subdirectories and their JSON files
    logger.info(f"Merging {dataset_name} workers from: {dataset_dir}")

    for item in sorted(os.listdir(dataset_dir)):
        dir_path = os.path.join(dataset_dir, item)
        if os.path.isdir(dir_path):
            worker_json_files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith(".json")]
            json_files.extend(worker_json_files)
            logger.debug(f"Found {len(worker_json_files)} JSON files in worker dir {dir_path}")

    logger.info(f"Found {len(json_files)} worker JSON files for {dataset_name}")

    if not json_files:
        raise RuntimeError(f"No JSON files found for dataset {dataset_name} in {dataset_dir}")

    # Move all JSON files to the dataset directory (in-place)
    # dataset_dir already IS the output directory (e.g., output/puffer/nuplan)
    dirs_to_remove = set()
    for json_file in json_files:
        filename = os.path.basename(json_file)
        destination = os.path.join(dataset_dir, filename)
        shutil.move(json_file, destination)
        dirs_to_remove.add(os.path.dirname(json_file))

    # Remove worker directories (unconditionally - may contain cache/temp files)
    for dir_to_remove in dirs_to_remove:
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)
            logger.debug(f"Removed worker directory: {dir_to_remove}")

    logger.info(f"Successfully merged {len(json_files)} JSON files from {dataset_name} to {dataset_dir}")


def merge_multiple_datasets(output_dir: str) -> None:
    """
    Merge multiple dataset directories into a single output directory.

    For Puffer format, each scenario is already a separate JSON file, so we just
    need to move all JSON files from dataset subdirectories to the main output directory.

    Args:
        output_dir: Directory containing dataset-specific subdirectories
    """
    # Look for dataset subdirectories
    dataset_dirs = [
        os.path.join(output_dir, d)
        for d in sorted(os.listdir(output_dir))
        if os.path.isdir(os.path.join(output_dir, d))
    ]

    logger.info(
        f"Found {len(dataset_dirs)} dataset directories to merge: {[os.path.basename(d) for d in dataset_dirs]}",
    )

    if not dataset_dirs:
        logger.warning("No dataset directories found to merge")
        return

    if len(dataset_dirs) == 1:
        # Single dataset - move all JSON files to parent directory
        single_dir = dataset_dirs[0]
        json_files = [os.path.join(single_dir, f) for f in os.listdir(single_dir) if f.endswith(".json")]

        logger.info(f"Moving {len(json_files)} JSON files from single dataset to output directory")
        for json_file in json_files:
            filename = os.path.basename(json_file)
            destination = os.path.join(output_dir, filename)
            shutil.move(json_file, destination)

        # Remove dataset directory (unconditionally - may contain cache/temp files)
        if os.path.exists(single_dir):
            shutil.rmtree(single_dir)
            logger.debug(f"Removed dataset directory: {single_dir}")

        logger.info(f"Single dataset merged: {len(json_files)} scenarios moved to {output_dir}")
        return

    # Multiple datasets - move all JSON files to parent directory
    total_files = 0
    for dataset_dir in tqdm(dataset_dirs, desc="Merging datasets"):
        try:
            json_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".json")]

            logger.debug(f"Moving {len(json_files)} JSON files from {os.path.basename(dataset_dir)}")

            for json_file in json_files:
                filename = os.path.basename(json_file)
                destination = os.path.join(output_dir, filename)
                shutil.move(json_file, destination)
                total_files += 1

            # Remove dataset directory (unconditionally - may contain cache/temp files)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
                logger.debug(f"Removed dataset directory: {dataset_dir}")

        except Exception as e:
            logger.error(f"Error processing dataset directory {dataset_dir}: {e!s}")
            raise

    logger.info(f"Successfully merged {total_files} JSON files from {len(dataset_dirs)} datasets to {output_dir}")


def merge_json_files(output_dir: str) -> None:
    """
    Merge JSON files from subdirectories into the main output directory.

    Args:
        output_dir: Directory containing subdirectories with JSON files
    """
    json_files = []

    try:
        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                list_dir = os.listdir(dir_path)
                json_files += [os.path.join(dir_path, f) for f in list_dir if f.endswith(".json")]

        logger.debug(f"Found {len(json_files)} JSON files to merge")
        logger.debug(f"Merging files into: {output_dir}")

        for file in json_files:
            shutil.move(file, output_dir)

        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

        logger.debug(f"All JSON files moved to {output_dir} and subdirs deleted.")
    except Exception as e:
        logger.error(f"Error during JSON file merging: {e!s}")
        raise

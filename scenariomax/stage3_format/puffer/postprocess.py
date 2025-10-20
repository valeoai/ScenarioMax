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
        # try:
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
        # except Exception as e:
        #     logger.error(f"Worker {worker_index} failed to process scenario: {e!s}")
        #     pbar.close()
        #     raise e

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
    parent_dir = os.path.dirname(dataset_dir)

    # Look for worker subdirectories and their JSON files
    logger.info(f"Merging {dataset_name} workers from: {dataset_dir}")

    for item in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, item)
        if os.path.isdir(dir_path):
            worker_json_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")]
            json_files.extend(worker_json_files)
            logger.debug(f"Found {len(worker_json_files)} JSON files in worker dir {dir_path}")

    logger.info(f"Found {len(json_files)} worker JSON files for {dataset_name}")

    if not json_files:
        raise RuntimeError(f"No JSON files found for dataset {dataset_name} in {dataset_dir}")

    # Create dataset directory in parent if it doesn't exist
    dataset_output_dir = os.path.join(parent_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Move all JSON files to the dataset directory
    dirs_to_remove = set()
    for json_file in json_files:
        filename = os.path.basename(json_file)
        destination = os.path.join(dataset_output_dir, filename)
        shutil.move(json_file, destination)
        dirs_to_remove.add(os.path.dirname(json_file))

    # Remove worker directories
    for dir_to_remove in dirs_to_remove:
        if os.path.exists(dir_to_remove) and not os.listdir(dir_to_remove):
            shutil.rmtree(dir_to_remove)
            logger.debug(f"Removed empty worker directory: {dir_to_remove}")

    # Remove the worker container directory
    if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
        shutil.rmtree(dataset_dir)
        logger.debug(f"Removed empty dataset directory: {dataset_dir}")

    logger.info(f"Successfully merged {len(json_files)} JSON files from {dataset_name} to {dataset_output_dir}")


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

import json
import os
import shutil
from collections.abc import Callable, Generator, Iterable
from typing import Any

from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.unified_to_gpudrive import convert_to_json


logger = logger_utils.get_logger(__name__)


def postprocess_gpudrive(
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
    Process scenarios and write them as GPU Drive JSON files.

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
    logger.debug(f"Worker {worker_index} writing to JSON: {output_path}")
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

            scenario_json = convert_to_json.convert(unified_scenario)

            if scenario_json is not None:
                with open(os.path.join(output_path, f"{unified_scenario.export_file_name}.json"), "w") as f:
                    json.dump(scenario_json, f)

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed_count})
        except Exception as e:
            logger.error(f"Worker {worker_index} failed to process scenario: {e!s}")
            pbar.close()
            raise e


def merge_dataset_workers(dataset_dir: str, dataset_name: str) -> None:
    """
    Merge JSON files from worker subdirectories into the parent directory.

    Args:
        dataset_dir: Directory containing worker subdirectories with JSON files
        dataset_name: Name of the dataset (for logging)
    """
    import shutil

    json_files = []
    parent_dir = os.path.dirname(dataset_dir)

    # Look for worker subdirectories and their JSON files
    basename = os.path.basename(dataset_dir)
    logger.info(f"Merging {dataset_name} workers from: {dataset_dir}")

    for item in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, item)
        # Look for directories that match the worker pattern: {basename}_{worker_index}
        if os.path.isdir(dir_path) and item.startswith(f"{basename}_"):
            list_dir = os.listdir(dir_path)
            worker_json_files = [os.path.join(dir_path, f) for f in list_dir if f.endswith(".json")]
            json_files.extend(worker_json_files)
            logger.debug(f"Found {len(worker_json_files)} JSON files in worker dir {dir_path}")

    logger.info(f"Found {len(json_files)} worker JSON files for {dataset_name}")

    if not json_files:
        logger.warning(f"No JSON files found for dataset {dataset_name} in {dataset_dir}")
        return

    # Move files to parent directory
    for file in json_files:
        filename = os.path.basename(file)
        destination = os.path.join(parent_dir, filename)
        shutil.move(file, destination)

    # Remove worker directories after moving files
    basename = os.path.basename(dataset_dir)
    for item in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, item)
        # Remove directories that match the worker pattern: {basename}_{worker_index}
        if os.path.isdir(dir_path) and item.startswith(f"{basename}_"):
            shutil.rmtree(dir_path)
            logger.debug(f"Removed worker directory: {dir_path}")

    # Remove the empty dataset directory
    if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
        shutil.rmtree(dataset_dir)
        logger.debug(f"Removed empty dataset directory: {dataset_dir}")

    logger.info(f"Successfully moved {len(json_files)} JSON files from {dataset_name} to {parent_dir}")


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

        logger.debug(f"All json files moved to {output_dir} and subdirs deleted.")
    except Exception as e:
        logger.error(f"Error during JSON file merging: {e!s}")
        raise

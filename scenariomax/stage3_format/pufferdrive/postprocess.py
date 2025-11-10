"""
Postprocessing functions for Puffer format conversion.

Handles merging worker outputs and saving scenarios as individual JSON files.
Each scenario is saved as a separate JSON file named {scenario_id}.json.
"""

import os
import shutil

from tqdm import tqdm

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def merge_dataset_workers(dataset_dir: str, dataset_name: str) -> None:
    """
    Merge JSON files from worker subdirectories into the parent directory.

    Each scenario JSON file is moved from worker subdirectories to the main dataset directory.

    Args:
        dataset_dir: Directory containing worker subdirectories with JSON files
        dataset_name: Name of the dataset (for logging)

    Raises:
        RuntimeError: If no JSON files found or if merge operations fail
        OSError: If file operations fail due to permissions or disk issues
    """
    if not os.path.exists(dataset_dir):
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")

    json_files = []

    # Look for worker subdirectories and their JSON files
    logger.info(f"Merging {dataset_name} workers from: {dataset_dir}")

    try:
        for item in sorted(os.listdir(dataset_dir)):
            dir_path = os.path.join(dataset_dir, item)
            if os.path.isdir(dir_path):
                worker_json_files = [
                    os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith(".json")
                ]
                json_files.extend(worker_json_files)
                logger.debug(f"Found {len(worker_json_files)} JSON files in worker dir {dir_path}")
    except PermissionError as e:
        raise OSError(f"Permission denied while accessing {dataset_dir}: {e}") from e

    logger.info(f"Found {len(json_files)} worker JSON files for {dataset_name}")

    if not json_files:
        raise RuntimeError(f"No JSON files found for dataset {dataset_name} in {dataset_dir}")

    # Move all JSON files to the dataset directory with validation
    dirs_to_remove = set()
    moved_files = []  # Track successfully moved files for rollback if needed

    try:
        for json_file in json_files:
            if not os.path.exists(json_file):
                logger.warning(f"JSON file no longer exists, skipping: {json_file}")
                continue

            filename = os.path.basename(json_file)
            destination = os.path.join(dataset_dir, filename)

            # Check for destination conflicts
            if os.path.exists(destination):
                logger.warning(f"Destination file already exists, will be overwritten: {destination}")

            shutil.move(json_file, destination)

            # Verify the move succeeded
            if not os.path.exists(destination):
                raise OSError(f"Failed to move file to destination: {destination}")

            moved_files.append(destination)
            dirs_to_remove.add(os.path.dirname(json_file))

    except (OSError, PermissionError) as e:
        logger.error(f"Error moving JSON files: {e}")
        raise OSError(f"Failed to merge worker files for {dataset_name}: {e}") from e

    # Remove worker directories (unconditionally - may contain cache/temp files)
    for dir_to_remove in dirs_to_remove:
        try:
            if os.path.exists(dir_to_remove):
                shutil.rmtree(dir_to_remove)
                logger.debug(f"Removed worker directory: {dir_to_remove}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not remove worker directory {dir_to_remove}: {e}")
            # Continue cleanup despite errors - worker dirs are just temp files

    logger.info(f"Successfully merged {len(moved_files)} JSON files from {dataset_name} to {dataset_dir}")


def merge_multiple_datasets(output_dir: str) -> None:
    """
    Merge multiple dataset directories into a single output directory.

    For Puffer format, each scenario is already a separate JSON file, so we just
    need to move all JSON files from dataset subdirectories to the main output directory.

    Args:
        output_dir: Directory containing dataset-specific subdirectories

    Raises:
        RuntimeError: If output directory doesn't exist
        OSError: If file operations fail due to permissions or disk issues
    """
    if not os.path.exists(output_dir):
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    # Look for dataset subdirectories
    try:
        dataset_dirs = [
            os.path.join(output_dir, d)
            for d in sorted(os.listdir(output_dir))
            if os.path.isdir(os.path.join(output_dir, d))
        ]
    except PermissionError as e:
        raise OSError(f"Permission denied while accessing {output_dir}: {e}") from e

    logger.info(
        f"Found {len(dataset_dirs)} dataset directories to merge: {[os.path.basename(d) for d in dataset_dirs]}",
    )

    if not dataset_dirs:
        logger.warning("No dataset directories found to merge")
        return

    if len(dataset_dirs) == 1:
        # Single dataset - move all JSON files to parent directory
        single_dir = dataset_dirs[0]
        try:
            json_files = [os.path.join(single_dir, f) for f in os.listdir(single_dir) if f.endswith(".json")]
        except (OSError, PermissionError) as e:
            raise OSError(f"Error reading files from {single_dir}: {e}") from e

        logger.info(f"Moving {len(json_files)} JSON files from single dataset to output directory")

        try:
            for json_file in json_files:
                if not os.path.exists(json_file):
                    logger.warning(f"JSON file no longer exists, skipping: {json_file}")
                    continue

                filename = os.path.basename(json_file)
                destination = os.path.join(output_dir, filename)

                if os.path.exists(destination):
                    logger.warning(f"Destination file already exists, will be overwritten: {destination}")

                shutil.move(json_file, destination)

                # Verify the move succeeded
                if not os.path.exists(destination):
                    raise OSError(f"Failed to move file to destination: {destination}")

        except (OSError, PermissionError) as e:
            logger.error(f"Error moving files from single dataset: {e}")
            raise OSError(f"Failed to merge single dataset: {e}") from e

        # Remove dataset directory (unconditionally - may contain cache/temp files)
        try:
            if os.path.exists(single_dir):
                shutil.rmtree(single_dir)
                logger.debug(f"Removed dataset directory: {single_dir}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not remove dataset directory {single_dir}: {e}")

        logger.info(f"Single dataset merged: {len(json_files)} scenarios moved to {output_dir}")
        return

    # Multiple datasets - move all JSON files to parent directory
    total_files = 0
    for dataset_dir in tqdm(dataset_dirs, desc="Merging datasets"):
        try:
            json_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".json")]

            logger.debug(f"Moving {len(json_files)} JSON files from {os.path.basename(dataset_dir)}")

            for json_file in json_files:
                if not os.path.exists(json_file):
                    logger.warning(f"JSON file no longer exists, skipping: {json_file}")
                    continue

                filename = os.path.basename(json_file)
                destination = os.path.join(output_dir, filename)

                if os.path.exists(destination):
                    logger.warning(f"Destination file already exists, will be overwritten: {destination}")

                shutil.move(json_file, destination)

                # Verify the move succeeded
                if not os.path.exists(destination):
                    raise OSError(f"Failed to move file to destination: {destination}")

                total_files += 1

            # Remove dataset directory (unconditionally - may contain cache/temp files)
            try:
                if os.path.exists(dataset_dir):
                    shutil.rmtree(dataset_dir)
                    logger.debug(f"Removed dataset directory: {dataset_dir}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not remove dataset directory {dataset_dir}: {e}")

        except (OSError, PermissionError) as e:
            logger.error(f"Error processing dataset directory {dataset_dir}: {e!s}")
            raise OSError(f"Failed to merge dataset {dataset_dir}: {e}") from e

    logger.info(f"Successfully merged {total_files} JSON files from {len(dataset_dirs)} datasets to {output_dir}")


def merge_json_files(output_dir: str) -> None:
    """
    Merge JSON files from subdirectories into the main output directory.

    Args:
        output_dir: Directory containing subdirectories with JSON files

    Raises:
        RuntimeError: If output directory doesn't exist
        OSError: If file operations fail due to permissions or disk issues
    """
    if not os.path.exists(output_dir):
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    json_files = []
    dirs_to_remove = set()

    try:
        for out_dir in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, out_dir)
            if os.path.isdir(dir_path):
                list_dir = os.listdir(dir_path)
                worker_json_files = [os.path.join(dir_path, f) for f in list_dir if f.endswith(".json")]
                json_files.extend(worker_json_files)
                if worker_json_files:
                    dirs_to_remove.add(dir_path)

        logger.debug(f"Found {len(json_files)} JSON files to merge")
        logger.debug(f"Merging files into: {output_dir}")

        moved_count = 0
        for file in json_files:
            if not os.path.exists(file):
                logger.warning(f"JSON file no longer exists, skipping: {file}")
                continue

            filename = os.path.basename(file)
            destination = os.path.join(output_dir, filename)

            if os.path.exists(destination):
                logger.warning(f"Destination file already exists, will be overwritten: {destination}")

            shutil.move(file, destination)

            # Verify the move succeeded
            if not os.path.exists(destination):
                raise OSError(f"Failed to move file to destination: {destination}")

            moved_count += 1

        # Remove subdirectories after successful merge
        for dir_path in dirs_to_remove:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.debug(f"Removed subdirectory: {dir_path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not remove subdirectory {dir_path}: {e}")

        logger.debug(f"Successfully moved {moved_count} JSON files to {output_dir} and cleaned up subdirs.")

    except PermissionError as e:
        logger.error(f"Permission denied during JSON file merging: {e!s}")
        raise OSError(f"Permission denied: {e}") from e
    except Exception as e:
        logger.error(f"Error during JSON file merging: {e!s}")
        raise

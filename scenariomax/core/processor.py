"""
Shared multiprocessing utilities for the 3-stage pipeline.

This module provides reusable parallel processing functions for:
- Stage 1: Raw → Unified
- Stage 2: Unified → Enhanced
- Stage 3: Unified → Target Format
"""

import os
import pickle
import shutil
from collections.abc import Callable
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def process_batch_parallel(
    items: list[Any],
    process_fn: Callable,
    num_workers: int = 8,
    desc: str = "Processing",
) -> list[Any]:
    """
    Process a list of items in parallel using joblib.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item (must return processed item)
        num_workers: Number of parallel workers
        desc: Description for progress bar

    Returns:
        List of processed items
    """
    if not items:
        return []

    # Adjust workers if fewer items than workers
    num_workers = min(num_workers, len(items))

    logger.info(f"Processing {len(items)} items with {num_workers} workers")

    # Process with progress bar
    with tqdm(total=len(items), desc=desc) as pbar:

        def process_with_progress(item):
            result = process_fn(item)
            pbar.update(1)
            return result

        results = Parallel(n_jobs=num_workers)(delayed(process_with_progress)(item) for item in items)

    return results


def load_pickle_files(input_path: str) -> list[dict[str, Any]]:
    """
    Load all pickle files from a directory (recursively).

    Args:
        input_path: Directory containing .pkl files

    Returns:
        List of loaded scenarios
    """
    scenarios = []
    pickle_files = []

    for root, _, files in os.walk(input_path):
        for file in sorted(files):
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    logger.info(f"Found {len(pickle_files)} pickle files in {input_path}")

    for file_path in tqdm(pickle_files, desc="Loading pickles"):
        with open(file_path, "rb") as f:
            scenario = pickle.load(f)
            scenarios.append(scenario)

    logger.info(f"Loaded {len(scenarios)} scenarios")
    return scenarios


def save_pickle_files(scenarios: list[dict[str, Any]], output_path: str) -> None:
    """
    Save scenarios as pickle files to output directory.

    Args:
        scenarios: List of scenarios to save
        output_path: Output directory
    """
    os.makedirs(output_path, exist_ok=True)

    for i, scenario in enumerate(tqdm(scenarios, desc="Saving pickles")):
        # Use scenario ID if available, otherwise use index
        if hasattr(scenario, "export_file_name"):
            filename = f"{scenario.export_file_name}.pkl"
        elif isinstance(scenario, dict) and "id" in scenario:
            filename = f"{scenario['id']}.pkl"
        else:
            filename = f"scenario_{i:06d}.pkl"

        file_path = os.path.join(output_path, filename)
        with open(file_path, "wb") as f:
            pickle.dump(scenario, f)

    logger.info(f"Saved {len(scenarios)} scenarios to {output_path}")


def setup_output_directory(output_path: str, clean: bool = True) -> None:
    """
    Setup output directory, optionally removing existing content.

    Args:
        output_path: Output directory path
        clean: If True, remove existing directory first
    """
    if clean and os.path.exists(output_path):
        logger.info(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory ready: {output_path}")

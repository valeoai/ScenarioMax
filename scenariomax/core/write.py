import os
import pickle
import shutil
from collections.abc import Callable, Generator, Iterable
from functools import partial
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.core.exceptions import EmptyDatasetError, WorkerProcessingError
from scenariomax.raw_to_unified.datasets import utils as converter_utils


logger = logger_utils.get_logger(__name__)


def default_preprocess_func(
    scenarios: list[Any] | Generator[Any, None, None],
    worker_index: int,
) -> list[Any] | Generator[Any, None, None]:
    """
    Default preprocessing function that returns scenarios unchanged.
    Can be overridden for dataset-specific preprocessing (e.g., Waymo memory management).

    Args:
        scenarios: Input scenarios (list or generator)
        worker_index: Worker index, useful for logging

    Returns:
        Processed input scenarios (list or generator)
    """
    return scenarios


def write_to_directory(
    convert_func: Callable,
    scenarios: list[Any] | Generator[Any, None, None] | Iterable[Any],
    output_path: str,
    dataset_version: str,
    dataset_name: str,
    num_workers: int = 8,
    preprocess: Callable | None = None,
    postprocess_func: Callable | None = None,
    **kwargs,
) -> None:
    """
    Write scenarios to directory using multiple workers.

    Args:
        convert_func: Function to convert scenarios
        postprocess_func: Optional post-processing function
        scenarios: List or generator of scenarios to process
        output_path: Path to output directory
        dataset_version: Dataset version
        dataset_name: Dataset name
        num_workers: Number of parallel workers
        preprocess: Optional preprocessing function
        **kwargs: Additional keyword arguments for workers
    """
    if preprocess is None:
        preprocess = default_preprocess_func

    if scenarios == [] or not scenarios:
        logger.error(f"No scenarios provided for processing in {output_path}")
        raise EmptyDatasetError(dataset_name, output_path)

    # Setup directory and workers
    _setup_output_directory(output_path)
    _create_worker_directories(output_path, num_workers)

    # Prepare worker arguments
    kwargs_for_workers = [{**kwargs} for _ in range(num_workers)]
    num_workers, worker_arguments_list = _prepare_worker_arguments(
        scenarios,
        num_workers,
        output_path,
        kwargs_for_workers,
    )

    # Execute parallel processing
    _execute_parallel_processing(
        worker_arguments_list,
        convert_func,
        postprocess_func,
        dataset_version,
        dataset_name,
        preprocess,
        num_workers,
    )

    logger.info(f"✅ Successfully processed all scenarios to {output_path}")


def write_to_directory_single_worker(
    convert_func: Callable,
    postprocess_func: Callable | None,
    list_scenarios: list[Any],
    output_path: str,
    dataset_version: str,
    dataset_name: str,
    worker_index: int = 0,
    preprocess: Callable = default_preprocess_func,
    worker_kwargs: dict | None = None,
    **kwargs,
) -> bool:
    """
    Convert a batch of scenarios using a single worker.

    Args:
        convert_func: Function to convert scenarios
        postprocess_func: Optional post-processing function
        scenarios: List or generator of scenarios to process
        output_path: Path to output directory
        dataset_version: Dataset version
        dataset_name: Dataset name
        worker_index: Worker index for parallel processing
        preprocess: Function for preprocessing scenarios
        worker_kwargs: Worker-specific keyword arguments
        **kwargs: Additional keyword arguments

    Returns:
        True if processing was successful, False otherwise
    """
    if worker_kwargs is None:
        worker_kwargs = {}

    kwargs.update(worker_kwargs)

    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("The specified version in kwargs is replaced by argument: 'dataset_version'")

    memory_before = converter_utils.process_memory()
    logger.debug(f"Worker {worker_index} starting - Memory usage: {memory_before:.2f} MB")

    # Apply preprocessing and get scenario count
    scenarios = preprocess(list_scenarios, worker_index)
    num_scenarios = _get_scenario_count(scenarios, list_scenarios, dataset_name)

    memory_after = converter_utils.process_memory()
    logger.debug(
        f"Worker {worker_index} preprocessing complete - Memory: {memory_after:.2f} MB "
        f"(delta: {memory_after - memory_before:.2f} MB)",
    )

    # Set up progress tracking
    logger.debug(f"Worker {worker_index} processing {num_scenarios} scenarios")

    pbar = tqdm(desc=f"Worker {worker_index}", total=num_scenarios, unit=" item", leave=True)

    processed_count = 0
    error_count = 0

    if not postprocess_func:
        # Standard pickle processing
        processed_count = _process_scenarios_to_pickle(
            scenarios,
            convert_func,
            dataset_version,
            dataset_name,
            output_path,
            pbar,
            **kwargs,
        )
    else:
        # Format-specific processing (tfexample, gpudrive)
        logger.debug(f"Worker {worker_index} starting scenario processing (postprocess mode)")
        postprocess_func(
            output_path,
            worker_index,
            scenarios,
            convert_func,
            dataset_version,
            dataset_name,
            pbar,
            process_scenario,
            **kwargs,
        )
        processed_count = "N/A"  # Handled by postprocess function

    memory_final = converter_utils.process_memory()

    # Detailed completion logging
    _log_worker_completion(worker_index, processed_count, error_count, memory_final, output_path)

    return True


def process_scenario(
    scenario: Any,
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    **kwargs,
) -> tuple:
    """
    Process a single scenario and convert it to the desired format.

    Args:
        scenario: The scenario to be processed.
        convert_func: The function used to convert the scenario.
        dataset_version: The version of the dataset.
        dataset_name: The name of the dataset.
        **kwargs: Additional arguments for the conversion function.

    Returns:
        tuple: (unified_scenario, export_file_name)
            - unified_scenario: The converted scenario.
            - export_file_name: The name of the file where the scenario will be saved.
    """
    converted_scenario = convert_func(scenario, dataset_version, **kwargs)

    return converted_scenario


def _setup_output_directory(output_path: str) -> None:
    """Setup output directory, removing existing one if it exists."""
    if os.path.exists(output_path):
        logger.debug(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)
    logger.debug(f"Created output directory: {output_path}")


def _create_worker_directories(output_path: str, num_workers: int) -> None:
    """Create subdirectories for each worker."""
    basename = os.path.basename(output_path)
    for i in range(num_workers):
        subdir = os.path.join(output_path, f"{basename}_{i!s}")
        os.makedirs(subdir, exist_ok=False)
        logger.debug(f"Created worker directory: {subdir}")


def _prepare_worker_arguments(
    scenarios: list,
    num_workers: int,
    output_path: str,
    kwargs_for_workers: list,
) -> tuple[int, list]:
    """Prepare arguments for parallel workers by distributing scenarios evenly."""
    total_scenario_count = len(scenarios)
    if total_scenario_count < num_workers:
        logger.info(f"Using single worker as scenario count ({total_scenario_count}) < worker count ({num_workers})")
        num_workers = 1

    output_directory_basename = os.path.basename(output_path)
    worker_arguments_list = []
    scenarios_per_worker = total_scenario_count // num_workers

    for worker_index in range(num_workers):
        batch_start_index = worker_index * scenarios_per_worker
        batch_end_index = (
            total_scenario_count if worker_index == num_workers - 1 else batch_start_index + scenarios_per_worker
        )

        worker_output_directory = os.path.join(output_path, f"{output_directory_basename}_{worker_index!s}")
        worker_scenario_batch = scenarios[batch_start_index:batch_end_index]
        worker_arguments_list.append(
            [worker_scenario_batch, kwargs_for_workers[worker_index], worker_index, worker_output_directory],
        )

        logger.debug(
            f"Worker {worker_index} assigned {len(worker_scenario_batch)} scenarios "
            f"(indices {batch_start_index}:{batch_end_index})",
        )

    return num_workers, worker_arguments_list


def _execute_parallel_processing(
    worker_arguments_list: list,
    convert_func: Callable,
    postprocess_func: Callable | None,
    dataset_version: str,
    dataset_name: str,
    preprocess: Callable,
    num_workers: int,
) -> None:
    """Execute parallel processing of scenarios across multiple workers."""
    single_worker_processing_func = partial(
        write_to_directory_single_worker,
        convert_func=convert_func,
        postprocess_func=postprocess_func,
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
    )

    logger.debug(f"Starting parallel processing with {num_workers} workers")

    with Parallel(n_jobs=num_workers) as parallel_executor:
        worker_results = parallel_executor(
            delayed(single_worker_processing_func)(
                list_scenarios=worker_args[0],
                worker_kwargs=worker_args[1],
                worker_index=worker_args[2],
                output_path=worker_args[3],
            )
            for worker_args in worker_arguments_list
        )

    # Check for worker failures and raise detailed errors
    for worker_index, processing_success in enumerate(worker_results):
        if not processing_success:
            error_message = f"Worker {worker_index} failed during scenario processing"
            logger.error(error_message)
            raise WorkerProcessingError(worker_index, "Processing returned failure status")


def _get_scenario_count(scenarios: Any, list_scenarios: list, dataset_name: str) -> int:
    """Get the number of scenarios, with special handling for Waymo."""
    if dataset_name == "womd":
        from scenariomax.raw_to_unified.datasets.waymo.load import count_waymo_scenarios

        return count_waymo_scenarios(list_scenarios)
    else:
        return len(scenarios)


def _process_scenarios_to_pickle(
    scenarios: Any,
    convert_func: Callable,
    dataset_version: str,
    dataset_name: str,
    output_path: str,
    pbar: tqdm,
    **kwargs,
) -> int:
    """Process scenarios and save them as pickle files."""
    processed_count = 0

    for scenario in scenarios:
        try:
            unified_scenario = process_scenario(
                scenario,
                convert_func,
                dataset_version,
                dataset_name,
                **kwargs,
            )

            with open(os.path.join(output_path, f"{unified_scenario.export_file_name}.pkl"), "wb") as f:
                pickle.dump(unified_scenario, f)

            processed_count += 1
            pbar.update(1)
            pbar.set_postfix({"processed": processed_count})
        except Exception as e:
            logger.error(f"Failed to process scenario: {e}")
            pbar.close()
            raise e

    return processed_count


def _log_worker_completion(
    worker_index: int,
    processed_count: Any,
    error_count: int,
    memory_final: float,
    output_path: str,
) -> None:
    """Log worker completion details."""
    logger.debug(f"Worker {worker_index} COMPLETED:")
    logger.debug(f"  ✅ Processed: {processed_count} scenarios")
    logger.debug(f"  ❌ Errors: {error_count} scenarios")
    logger.debug(f"  📊 Memory: {memory_final:.2f} MB")
    logger.debug(f"  📁 Output: {output_path}")

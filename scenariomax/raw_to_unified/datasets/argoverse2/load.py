from pathlib import Path

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def get_av2_scenarios(av2_data_directory, start_index, num):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.info("\nReading raw data")

    all_scenario_files = sorted(Path(av2_data_directory).rglob("*.parquet"))

    return all_scenario_files


def preprocess_av2_scenarios(files, worker_index):
    """
    Convert the waymo files into scenario_pb2. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenario_pb2
    """

    for scenario_path in tqdm(files, desc=f"Process av2 scenarios for worker {worker_index}"):
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        scenario.static_map = static_map
        yield scenario

    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))  # return scenarios

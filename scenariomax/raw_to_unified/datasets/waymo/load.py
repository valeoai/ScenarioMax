import os

import scenariomax.raw_to_unified.datasets.waymo.waymo_protos.scenario_pb2 as scenario_pb2
from scenariomax import logger_utils
from scenariomax.raw_to_unified.datasets.waymo import utils as waymo_utils
from scenariomax.tf_utils import get_tensorflow


logger = logger_utils.get_logger(__name__)


def get_waymo_scenarios(data_path, start_index: int = 0, num: int | None = None):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.debug("Reading raw data")
    file_list = os.listdir(data_path)

    if num is None:
        num = len(file_list) - start_index

    assert len(file_list) >= start_index + num and start_index >= 0, (
        f"No sufficient files ({len(file_list)}) in raw_data_directory. need: {num}, start: {start_index}"
    )

    file_list = file_list[start_index : start_index + num]
    num_files = len(file_list)
    all_result = [os.path.join(data_path, f) for f in file_list]
    logger.debug(f"Find {num_files} waymo files")

    return all_result


def preprocess_waymo_scenarios(files, worker_index):
    """Convert the waymo files into scenario_pb2. This happens in each worker.

    Args:
        files: list of files to be converted
        worker_index: index of the worker

    Returns:
        Generator of scenario_pb2.Scenario
    """
    tf = get_tensorflow()

    for file in files:
        file_path = os.path.join(file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data)
            # a trick for loging file name
            scenario.scenario_id = scenario.scenario_id + waymo_utils.SPLIT_KEY + file

            yield scenario


def count_waymo_scenarios(files):
    """Count the number of waymo scenarios in the files.

    Args:
        files: list of files to be counted

    Returns:
        int: number of waymo scenarios
    """
    tf = get_tensorflow()
    count = 0
    for file in files:
        file_path = os.path.join(file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        count += sum(1 for _ in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator())

    return count

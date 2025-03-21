import logging
import os

import scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2 as scenario_pb2
from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.waymo.utils import SPLIT_KEY


logger = get_logger(__name__)

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    import tensorflow as tf

except ImportError as e:
    raise RuntimeError("TensorFlow package not found. Please install TensorFlow to use this module.") from e


def get_waymo_scenarios(waymo_data_directory, start_index: int = 0, num: int | None = None):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.info("Reading raw data")
    file_list = os.listdir(waymo_data_directory)

    if num is None:
        num = len(file_list) - start_index

    assert len(file_list) >= start_index + num and start_index >= 0, (
        f"No sufficient files ({len(file_list)}) in raw_data_directory. need: {num}, start: {start_index}"
    )

    file_list = file_list[start_index : start_index + num]
    num_files = len(file_list)
    all_result = [os.path.join(waymo_data_directory, f) for f in file_list]
    logger.info(f"Find {num_files} waymo files")

    return all_result


def preprocess_waymo_scenarios(files, worker_index):
    """Convert the waymo files into scenario_pb2. This happens in each worker.

    Args:
        files: list of files to be converted
        worker_index: index of the worker

    Returns:
        Generator of scenario_pb2.Scenario
    """

    for file in files:
        file_path = os.path.join(file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data)
            # a trick for loging file name
            scenario.scenario_id = scenario.scenario_id + SPLIT_KEY + file

            yield scenario

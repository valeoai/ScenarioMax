from collections.abc import Callable
from dataclasses import dataclass

from scenariomax import logger_utils
from scenariomax.core.exceptions import UnsupportedDatasetError


logger = logger_utils.get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    version: str
    load_func: Callable
    convert_func: Callable
    preprocess_func: Callable | None = None
    additional_args: dict = None

    def __post_init__(self):
        if self.additional_args is None:
            self.additional_args = {}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""

    if dataset_name == "waymo":
        from scenariomax.stage1_convert.datasets import waymo

        return DatasetConfig(
            name="waymo",
            version="v1.3",
            load_func=waymo.get_waymo_scenarios,
            convert_func=waymo.convert_waymo_scenario,
            preprocess_func=waymo.preprocess_waymo_scenarios,
        )
    if dataset_name == "nuplan":
        from scenariomax.stage1_convert.datasets import nuplan

        return DatasetConfig(
            name="nuplan",
            version="v1.1",
            load_func=nuplan.get_nuplan_scenarios,
            convert_func=nuplan.convert_nuplan_scenario,
        )
    if dataset_name == "openscenes":
        from scenariomax.stage1_convert.datasets import openscenes

        return DatasetConfig(
            name="openscenes",
            version="v1.1",
            load_func=openscenes.get_openscenes_scenarios,
            convert_func=openscenes.convert_openscenes_scenario,
        )

    # Unsupported dataset - show list of supported options
    supported_datasets = ["waymo", "nuplan", "openscenes"]
    raise UnsupportedDatasetError(dataset_name, supported_datasets)

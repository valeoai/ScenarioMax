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
        from scenariomax.raw_to_unified.datasets import waymo

        return DatasetConfig(
            name="womd",
            version="v1.3",
            load_func=waymo.get_waymo_scenarios,
            convert_func=waymo.convert_waymo_scenario,
            preprocess_func=waymo.preprocess_waymo_scenarios,
        )
    if dataset_name == "nuscenes":
        raise UnsupportedDatasetError(dataset_name, "nuScenes dataset not yet supported in ScenarioMax")
        from scenariomax.raw_to_unified.datasets import nuscenes

        return DatasetConfig(
            name="nuscenes",
            version="v1.0",
            load_func=nuscenes.get_nuscenes_scenarios,
            convert_func=nuscenes.convert_nuscenes_scenario,
            preprocess_func=nuscenes.preprocess_nuscenes_scenarios,
        )
    if dataset_name == "nuplan":
        from scenariomax.raw_to_unified.datasets import nuplan

        return DatasetConfig(
            name="nuplan",
            version="v1.1",
            load_func=nuplan.get_nuplan_scenarios,
            convert_func=nuplan.convert_nuplan_scenario,
        )
    if dataset_name == "openscenes":
        from scenariomax.raw_to_unified.datasets import openscenes

        return DatasetConfig(
            name="openscenes",
            version="v1.1",
            load_func=openscenes.get_openscenes_scenarios,
            convert_func=openscenes.convert_openscenes_scenario,
        )
    if dataset_name == "argoverse2":
        raise UnsupportedDatasetError(dataset_name, "Argoverse2 dataset not yet supported in ScenarioMax")
        from scenariomax.raw_to_unified.datasets import argoverse2

        return DatasetConfig(
            name="argoverse",
            version="v2.0",
            load_func=argoverse2.get_av2_scenarios,
            convert_func=argoverse2.convert_av2_scenario,
            preprocess_func=argoverse2.preprocess_av2_scenarios,
        )

    supported_datasets = ["waymo", "nuplan", "openscenes"]
    raise UnsupportedDatasetError(dataset_name, supported_datasets)

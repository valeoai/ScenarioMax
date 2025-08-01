import os
import tempfile
from dataclasses import dataclass
from os.path import join

import nuplan
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder


NuPlanEgoType = TrackedObjectType.EGO

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)


def get_nuplan_scenarios(
    data_path,
    maps_path,
    num_files: int | None = None,
    logs: list | None = None,
    builder="nuplan_mini",
) -> list[NuPlanScenario]:
    """Gets NuPlan scenarios based on provided parameters.

    Retrieves scenarios from the NuPlan dataset using the specified parameters.

    Args:
        data_root: Path containing .db files, like /nuplan-v1.1/splits/mini.
        map_root: Path to map files.
        num_files: Maximum number of scenarios to retrieve. If None, retrieves all.
        logs: A list of logs, like ['2021.07.16.20.45.29_veh-35_01095_01486'].
            If None, loads all files in data_root.
        builder: Builder file name, defaults to "nuplan_mini".

    Returns:
        A collection of NuPlan scenario objects.
    """
    import hydra

    nuplan_package_path = NUPLAN_PACKAGE_PATH
    logs = logs or [file for file in os.listdir(data_path)]
    log_string = ""

    for log in logs:
        if log[-3:] == ".db":
            log = log[:-3]
        log_string += log
        log_string += ","
    log_string = log_string[:-1]

    dataset_parameters = [
        # builder setting
        f"scenario_builder={builder}",
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",  # 10 hz
        f"scenario_builder.data_root={data_path}",
        f"scenario_builder.map_root={maps_path}",
        # filter
        "scenario_filter=all_scenarios",  # simulate only one log
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.shuffle=true",
        f"scenario_filter.log_names=[{log_string}]",
        # "scenario_filter.scenario_types={}".format(all_scenario_types),
        # "scenario_filter.scenario_tokens=[]",
        # "scenario_filter.num_scenarios_per_type=1",
        # "scenario_filter.expand_scenarios=true",
        # "scenario_filter.limit_scenarios_per_type=10",  # use 10 scenarios per scenario type
        "scenario_filter.timestamp_threshold_s=10",  # minial scenario duration (s)
    ]

    if num_files is not None:
        dataset_parameters.append(f"scenario_filter.limit_total_scenarios={num_files}")

    base_config_path = os.path.join(nuplan_package_path, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

    save_dir = tempfile.mkdtemp()
    ego_controller = "perfect_tracking_controller"  # [log_play_back_controller, perfect_tracking_controller]
    observation = "box_observation"  # [box_observation, idm_agents_observation, lidar_pc_observation]

    # Compose the configuration
    overrides = [
        f"group={save_dir}",
        "worker=single_machine_thread_pool",  # ray_distributed, sequential, single_machine_thread_pool
        f"ego_controller={ego_controller}",
        f"observation={observation}",
        f"hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]",
        "output_dir=${group}/${experiment}",
        "metric_dir=${group}/${experiment}",
        *dataset_parameters,
    ]
    overrides.extend(
        [
            "job_name=planner_tutorial",
            "experiment=${experiment_name}/${job_name}",
            "experiment_name=planner_tutorial",
        ],
    )

    # get config
    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)

    profiler_name = "building_simulation"
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    # get scenarios from database
    return scenario_builder.get_scenarios(scenario_filter, common_builder.worker)


def construct_simulation_hydra_paths(base_config_path: str):
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, "config", "common")
    config_name = "default_simulation"
    config_path = join(base_config_path, "config", "simulation")
    experiment_dir = "file://" + join(base_config_path, "experiments")

    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def get_nuplan_scenarios_by_scene(
    data_path: str,
    maps_path: str,
    num_files: int | None = None,
    logs: list | None = None,
) -> list[NuPlanScenario]:
    """Gets NuPlan scenarios by directly parsing scenes from logs.
    Does not use the scenario builder
    Args:
        data_root: Path containing .db files, like /nuplan-v1.1/splits/mini.
        map_root: Path to map files.
        num_files: Maximum number of scenarios to retrieve. If None, retrieves all.
        logs: A list of logs, like ['2021.07.16.20.45.29_veh-35_01095_01486'].
            If None, loads all files in data_root.
    Returns:
        A collection of NuPlan scenario objects.
    """
    scenarios = []

    logs = logs or [file for file in os.listdir(data_path)]
    map_version = "nuplan-maps-v1.0"
    maps_db = GPKGMapsDB(map_version=map_version, map_root=maps_path)

    for log in logs:
        log_file_path = join(data_path, log)
        log_db = NuPlanDB(data_path, log_file_path, maps_db=maps_db)

        for scene in log_db.scene:
            tokens = log_db.lidar_pc.select_many(scene_token=scene.token)
            first_frame = tokens[0]
            last_frame = tokens[-1]
            scene_duration = (last_frame.timestamp - first_frame.timestamp) / 1_000_000

            if scene_duration < 19.0:  # What should we use?
                continue

            extraction_info = ScenarioExtractionInfo(scenario_duration=scene_duration, subsample_ratio=0.5)
            scenarios.append(
                NuPlanScenario(
                    data_root=data_path,
                    log_file_load_path=log_file_path,
                    initial_lidar_token=first_frame.token,
                    initial_lidar_timestamp=first_frame.timestamp,
                    scenario_type=f"{scene.name}+{scene.token}",
                    map_root=maps_path,
                    map_version=map_version,
                    map_name=log_db.log.map_version,
                    scenario_extraction_info=extraction_info,
                    ego_vehicle_parameters=get_pacifica_parameters(),
                ),
            )

            if len(scenarios) == num_files:
                return scenarios

    return scenarios

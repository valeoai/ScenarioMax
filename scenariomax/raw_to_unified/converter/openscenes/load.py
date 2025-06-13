import os
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm
try:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
except ImportError as e:
    raise RuntimeError("NuPlan package not found. Please install NuPlan to use this module.") from e


def get_nuplan_scenarios(
    nuplan_data_root,
    nuplan_logs_root,
    nuplan_maps_root,
    metadata_root,
    num_files: int | None = None,
    logs: list | None = None,
):
    scenarios = []
    openscenes_metadata = Path(metadata_root)
    print("Proccessing OpenScenes data, found ", len(list(openscenes_metadata.iterdir())), " log metadata files")
    for metadata_path in tqdm(openscenes_metadata.iterdir()):
        if logs and os.path.basename(metadata_path)[:-3] not in logs:
            continue
        with open(metadata_path, 'rb') as f:
            metdata = pickle.load(f)
        frames_in_scenes = defaultdict(list)
        for frame in metdata:
            frames_in_scenes[frame["scene_name"]].append(
                {
                    "lidar_pc_token": frame["token"],
                    "frame_idx": frame["frame_idx"],
                    "scene_token": frame["scene_token"],
                    "scene_name": frame["scene_name"],
                    "log_name": frame["log_name"],
                    "log_token": frame["log_token"],
                    "timestamp": frame["timestamp"],
                    "map_location": frame["map_location"]
                }
            )
        for frames_in_scene in frames_in_scenes.values():
            first_frame = frames_in_scene[0]
            last_frame = frames_in_scene[-1]
            scene_duration = (last_frame["timestamp"] - first_frame["timestamp"]) / 1_000_000 # convert microseconds to seconds
            extraction_info = ScenarioExtractionInfo(
                scenario_duration=scene_duration, subsample_ratio=0.5 # 10hz
            )
            scenario = NuPlanScenario(
                data_root=nuplan_data_root,
                log_file_load_path=os.path.join(nuplan_logs_root, first_frame["log_name"]+'.db'),
                initial_lidar_token=first_frame["lidar_pc_token"],
                initial_lidar_timestamp=first_frame["timestamp"],
                scenario_type=f'{frame["scene_name"]}+{frame["scene_token"]}',
                map_root=nuplan_maps_root,
                map_version="nuplan-maps-v1.0",
                map_name=first_frame["map_location"],
                scenario_extraction_info=extraction_info,
                ego_vehicle_parameters=get_pacifica_parameters(),
            )
            scenarios.append(scenario)
            if num_files and len(scenarios) == num_files:
                return scenarios
    return scenarios

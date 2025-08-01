import numpy as np
import trimesh

from scenariomax.unified_to_gpudrive import utils as gpudrive_utils
from scenariomax.unified_to_gpudrive.converter import roadgraph, state, traffic_light


def convert(unified_scenario):
    scenario = gpudrive_utils.AttrDict(unified_scenario)

    scenario_id = scenario.id

    # Construct the traffic light states (keeping them as requested)
    scenario_net_tl_states = unified_scenario["dynamic_map_elements"]
    tl_dict = traffic_light.convert_traffic_lights(scenario_net_tl_states)

    scenario_net_map_features = unified_scenario["static_map_elements"]
    roads, edge_segments = roadgraph.convert_map_features(scenario_net_map_features)

    # Skip scenario if it has 3D structures (like GPUDrive template)
    if roads is None or edge_segments is None:
        return None

    # Construct road edges for collision checking
    edge_segments = gpudrive_utils.filter_small_segments(edge_segments)
    edge_mesh = gpudrive_utils.generate_mesh(edge_segments)

    # Create collision managers
    road_collision_manager = trimesh.collision.CollisionManager()
    if edge_mesh and len(edge_mesh.vertices) > 0:
        road_collision_manager.add_object("road_edges", edge_mesh)
    agent_collision_manager = trimesh.collision.CollisionManager()
    trajectory_collision_manager = trimesh.collision.CollisionManager()

    scenario_net_track_features = unified_scenario["dynamic_agents"]
    objects, objects_distance_traveled = state.convert_track_features_to_objects(
        scenario_net_track_features,
        agent_collision_manager,
        trajectory_collision_manager,
    )

    gpudrive_utils.mark_colliding_agents(
        objects=objects,
        agent_collision_manager=agent_collision_manager,
        road_collision_manager=road_collision_manager,
        trajectory_collision_manager=trajectory_collision_manager,
    )

    metadata = unified_scenario["metadata"]
    if metadata["dataset_name"] == "WOMD":
        sdc_track_index = metadata["sdc_track_index"]
        objects_of_interest = metadata["objects_of_interest"]
        tracks_to_predict = metadata["tracks_to_predict"]
        metadata = {
            "sdc_track_index": sdc_track_index,
            "objects_of_interest": objects_of_interest,
            "tracks_to_predict": tracks_to_predict,
        }
    elif metadata["dataset_name"] == "nuPlan":
        sdc_index = [index for index, object in enumerate(objects) if object["is_sdc"]]
        metadata = {
            "sdc_track_index": sdc_index[0],
            "log_name": metadata["source_file"],
            # "ts": metadata["timesteps"],
            "initial_lidar_timestamp": metadata["initial_lidar_timestamp"],
            "map_name": metadata["map_name"],
            "objects_of_interest": [],
            "tracks_to_predict": [],
            "average_distance_traveled": gpudrive_utils.ensure_scalar(np.mean(objects_distance_traveled)),
            "scenario_type": metadata["scenario_type"],  # for openscenes data this contains the scenario token
        }

    scenario_dict = {
        "name": f"{unified_scenario.export_file_name}.json",
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict,
        "metadata": metadata,
    }

    return gpudrive_utils.convert_numpy(scenario_dict)

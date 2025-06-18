import numpy as np
import trimesh
from collections import defaultdict
import math
from scenariomax.unified_to_gpudrive.data.map_element_ids import (
    FILTERED_TYPES,
    TYPE_TO_MAP_FEATURE_NAME,
    TYPE_TO_MAP_ID,
    TRAFFIC_LIGHT_STATES_MAP,
)
from scenariomax.unified_to_gpudrive.utils import convert_numpy


ERR_VAL = -1e4
### BEGIN CODE ADAPTED FROM GPUDRIVE https://github.com/Emerge-Lab/gpudrive/blob/main/data_utils/process_waymo_files.py


def _filter_small_segments(segments, min_length=1e-6):
    """Filter out segments that are too short."""
    valid_segments = []
    for segment in segments:
        start, end = segment
        length = np.linalg.norm(np.array(end) - np.array(start))
        if length >= min_length:
            valid_segments.append(segment)
    return valid_segments


def _generate_mesh(segments, height=2.0, width=0.2):
    segments = np.array(segments, dtype=np.float64)
    starts, ends = segments[:, 0, :], segments[:, 1, :]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / lengths

    # Create the base box mesh with the height along the z-axis
    base_box = trimesh.creation.box(extents=[1.0, width, height])
    base_box.apply_translation([0.5, 0, 0])  # Align box's origin to its start
    z_axis = np.array([0, 0, 1])
    angles = np.arctan2(unit_directions[:, 1], unit_directions[:, 0])  # Rotation in the XY plane

    rectangles = []
    lengths = lengths.flatten()

    for i, (start, length, angle) in enumerate(zip(starts, lengths, angles)):
        # Copy the base box and scale to match segment length
        scaled_box = base_box.copy()
        scaled_box.apply_scale([length, 1.0, 1.0])

        # Apply rotation around the z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, z_axis)
        scaled_box.apply_transform(rotation_matrix)

        # Translate the box to the segment's starting point
        scaled_box.apply_translation(start)

        rectangles.append(scaled_box)

    # Concatenate all boxes into a single mesh
    mesh = trimesh.util.concatenate(rectangles)
    return mesh


def _create_agent_box_mesh(position, heading, length, width, height):
    """Create a box mesh for an agent at a given position and orientation.

    Args:
        position (list): [x, y, z] position
        heading (float): yaw angle in radians
        length (float): length of the box
        width (float): width of the box
        height (float): height of the box

    Returns:
        trimesh.Trimesh: Box mesh positioned and oriented correctly
    """
    # Create box centered at origin
    box = trimesh.creation.box(extents=[length, width, height])

    # Rotate box to align with heading
    z_axis = np.array([0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(heading, z_axis)
    box.apply_transform(rotation_matrix)

    # Move box to position
    box.apply_translation(position)

    return box


def _wrap_yaws(yaws):
    """Wraps yaw angles between pi and -pi radians."""
    return (yaws + np.pi) % (2 * np.pi) - np.pi


def _add_trajectories_to_mesh(obj, first_valid_idx, agent_collision_manager, trajectory_collision_manager):
    # Create agent at initial position
    initial_pos = [
        obj["position"][first_valid_idx]["x"],
        obj["position"][first_valid_idx]["y"],
        obj["position"][first_valid_idx]["z"],
    ]
    initial_heading = obj["heading"][first_valid_idx]
    initial_box = _create_agent_box_mesh(initial_pos, initial_heading, obj["length"], obj["width"], obj["height"])
    agent_collision_manager.add_object(str(obj["id"]), initial_box)

    # Create trajectory mesh
    if False in obj["valid"]:
        # Create trajectory segments of only valid positions
        trajectory_segments = []
        for i in range(len(obj["position"]) - 1):
            if obj["valid"][i] and obj["valid"][i + 1]:
                trajectory_segments.append(
                    [
                        [
                            obj["position"][i]["x"],
                            obj["position"][i]["y"],
                            obj["position"][i]["z"],
                        ],
                        [
                            obj["position"][i + 1]["x"],
                            obj["position"][i + 1]["y"],
                            obj["position"][i + 1]["z"],
                        ],
                    ],
                )
    else:
        obj_vertices = [[pos["x"], pos["y"], pos["z"]] for pos in obj["position"]]
        trajectory_segments = [[obj_vertices[i], obj_vertices[i + 1]] for i in range(len(obj_vertices) - 1)]

    trajectory_segments = _filter_small_segments(trajectory_segments)
    if len(trajectory_segments) > 0:
        trajectory_mesh = _generate_mesh(trajectory_segments)
        trajectory_collision_manager.add_object(str(obj["id"]), trajectory_mesh)


def _mark_colliding_agents(objects, agent_collision_manager, road_collision_manager, trajectory_collision_manager):
    # Check collisions between all init agent positions
    _, agent_collision_pairs = agent_collision_manager.in_collision_internal(return_names=True)

    # Check collisions between init agent positions and road edges
    _, road_collision_pairs = agent_collision_manager.in_collision_other(road_collision_manager, return_names=True)

    # Check trajectory collisions with road edges
    _, trajectory_collision_pairs = trajectory_collision_manager.in_collision_other(
        road_collision_manager,
        return_names=True,
    )

    # Create sets of colliding agent IDs
    colliding_agents = set()

    # Add agents that collide with each other at first step
    for agent1, agent2 in agent_collision_pairs:
        colliding_agents.add(agent1)
        colliding_agents.add(agent2)

    # Add agents that collide with road edges
    road_colliding_agents = {agent_id for agent_id, _ in road_collision_pairs}
    colliding_agents.update(road_colliding_agents)

    # Add agents whose trajectories collide with road edges
    trajectory_colliding_agents = {agent_id for agent_id, _ in trajectory_collision_pairs}
    colliding_agents.update(trajectory_colliding_agents)

    # Update mark_as_expert based on initial collisions
    for index, obj in enumerate(objects):
        if obj["type"] in ["vehicle", "cyclist"]:
            if str(obj["id"]) in colliding_agents:
                objects[index]["mark_as_expert"] = True
            else:
                objects[index]["mark_as_expert"] = False


### END CODE ADAPTED FROM GPUDRIVE https://github.com/Emerge-Lab/gpudrive/blob/main/data_utils/process_waymo_files.py


def _map_type_to_mapfeature(type):
    for key, value in TYPE_TO_MAP_FEATURE_NAME.items():
        if key in type:
            return value
    return str.lower(type)


def _convert_map_features(scenario_net_map_features):
    road_features = []
    edge_segments = []
    index = 0

    filtered_road_types = ["ROAD_EDGE_SIDEWALK", "DRIVEWAY"]

    for map_feature_id in scenario_net_map_features:
        feature = scenario_net_map_features[map_feature_id]
        feature_type = feature["type"]

        if feature_type in filtered_road_types:
            continue

        new_feature = {
            "geometry": [],
            "type": _map_type_to_mapfeature(feature_type),
            "map_element_id": TYPE_TO_MAP_ID[feature_type].value if feature_type in TYPE_TO_MAP_ID else -1,
            "id": int(map_feature_id) if map_feature_id.isdigit() else index,
        }
        geometry_key = next((k for k in ["polyline", "polygon", "position"] if k in feature), None)

        if not geometry_key:
            continue

        original_geometry = feature[geometry_key]
        original_geometry = (
            np.expand_dims(original_geometry, 0) if len(original_geometry.shape) == 1 else original_geometry
        )

        if original_geometry.shape[-1] == 3:
            geometry = [{"x": point[0], "y": point[1], "z": point[2]} for point in original_geometry]
        elif original_geometry.shape[-1] == 2:
            geometry = [{"x": point[0], "y": point[1], "z": 0.0} for point in original_geometry]

        new_feature["geometry"] = geometry

        if "ROAD_EDGE" in feature_type:
            edge_vertices = [[r["x"], r["y"], r["z"]] for r in new_feature["geometry"]]
            edge_segments += [[edge_vertices[i], edge_vertices[i + 1]] for i in range(len(edge_vertices) - 1)]

        road_features.append(new_feature)
        index += 1

    return road_features, edge_segments

def _check_object_distance_traveled(positions, valids):
    valid_positions = positions[valids]
    if len(valid_positions) < 2:
        return 0.0
    diffs = valid_positions[1:] - valid_positions[:-1]
    step_distances = np.linalg.norm(diffs, axis=1)
    return np.sum(step_distances)

def _ensure_scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.size == 1 else value

def _extract_obj(index, object_id, scenario_net_object):
    state = scenario_net_object["state"]
    metadata = scenario_net_object["metadata"]
    valids = state["valid"].astype(bool)
    positions = state["position"]
    headings = state["heading"]
    velocities = state["velocity"]

    position = [
        {"x": point[0], "y": point[1], "z": point[2]} if valids[index] else {"x": ERR_VAL, "y": ERR_VAL, "z": ERR_VAL}
        for index, point in enumerate(positions)
    ]

    final_valid_index = 0
    for i, is_valid in enumerate(valids):
        if is_valid:
            final_valid_index = i

    length = _ensure_scalar(state["length"][final_valid_index])
    width = _ensure_scalar(state["width"][final_valid_index])
    height = _ensure_scalar(state["height"][final_valid_index])

    heading = [_wrap_yaws(heading) if valids[index] else ERR_VAL for index, heading in enumerate(headings)]

    velocity = [
        {
            "x": point[0],
            "y": point[1],
        }
        if valids[index]
        else {
            "x": ERR_VAL,
            "y": ERR_VAL,
        }
        for index, point in enumerate(velocities)
    ]

    goalPosition = {
        "x": positions[final_valid_index][0],
        "y": positions[final_valid_index][1],
        "z": positions[final_valid_index][2],
    }

    return {
        "id": int(object_id) if object_id.isdigit() else index,
        "type": str.lower(metadata["type"]),
        "position": position,
        "width": width,
        "length": length,
        "height": height,
        "heading": heading,
        "velocity": velocity,
        "valid": valids,
        "goalPosition": goalPosition,
        "is_sdc": object_id == "ego",
        "mark_as_expert": False,
        "total_distance_traveled": _ensure_scalar(_check_object_distance_traveled(positions, valids))
    }


def _convert_track_features_to_objects(scenario_net_tracks, agent_collision_manager, trajectory_collision_manager):
    objects = []
    objects_distance_traveled = []
    for index, object_id in enumerate(scenario_net_tracks):
        scenario_net_object = scenario_net_tracks[object_id]
        if scenario_net_object["metadata"]["type"] in FILTERED_TYPES:
            continue

        obj = _extract_obj(index, object_id, scenario_net_object)

        if obj["type"] not in ["vehicle", "cyclist"]:
            obj["mark_as_expert"] = False
            objects.append(obj)
        else:
            # Find first valid position
            first_valid_idx = next((i for i, valid in enumerate(obj["valid"]) if valid), None)
            if first_valid_idx is not None:
                _add_trajectories_to_mesh(obj, first_valid_idx, agent_collision_manager, trajectory_collision_manager)
                objects.append(obj)
                objects_distance_traveled.append(obj["total_distance_traveled"])
    return objects, objects_distance_traveled


def _convert_traffic_lights(scenario_net_tl_states):
    tl_dict = defaultdict(
        lambda: {"state": [], "x": [], "y": [], "z": [], "time_index": [], "lane_id": []}
    )
    for i, (lane_id, tl_state) in enumerate(scenario_net_tl_states.items()):
        x, y = tl_state["stop_point"]
        light_states = tl_state["state"]["object_state"]
        for i, state in enumerate(light_states):
            tl_dict[lane_id]["state"].append(TRAFFIC_LIGHT_STATES_MAP[state])
            tl_dict[lane_id]["x"].append(x)
            tl_dict[lane_id]["y"].append(y)
            tl_dict[lane_id]["time_index"].append(i)
            tl_dict[lane_id]["lane_id"].append(lane_id)
    return tl_dict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def build_gpudrive_example(name, scenario_net_scene, debug=False):
    scenario = AttrDict(scenario_net_scene)

    scenario_id = scenario.id

    # Construct the traffic light states
    scenario_net_tl_states = scenario_net_scene["dynamic_map_states"]
    tl_dict = _convert_traffic_lights(scenario_net_tl_states)

    scenario_net_map_features = scenario_net_scene["map_features"]
    roads, edge_segments = _convert_map_features(scenario_net_map_features)

    # Construct road edges for collision checking
    edge_segments = _filter_small_segments(edge_segments)
    edge_mesh = _generate_mesh(edge_segments)

    # Create collision managers
    road_collision_manager = trimesh.collision.CollisionManager()
    road_collision_manager.add_object("road_edges", edge_mesh)
    agent_collision_manager = trimesh.collision.CollisionManager()
    trajectory_collision_manager = trimesh.collision.CollisionManager()

    scenario_net_track_features = scenario_net_scene["tracks"]
    objects, objects_distance_traveled = _convert_track_features_to_objects(
        scenario_net_track_features,
        agent_collision_manager,
        trajectory_collision_manager,
    )

    # _mark_colliding_agents(
    #     objects=objects,
    #     agent_collision_manager=agent_collision_manager,
    #     road_collision_manager=road_collision_manager,
    #     trajectory_collision_manager=trajectory_collision_manager,
    # )

    metadata = scenario_net_scene["metadata"]
    if metadata["dataset"] == "waymo":
        sdc_track_index = metadata["sdc_track_index"]
        objects_of_interest = metadata["objects_of_interest"]
        tracks_to_predict = [
            {"track_index": track["track_index"], "difficulty": track["difficulty"]}
            for _, track in metadata["tracks_to_predict"].items()
        ]
        metadata = {
            "sdc_track_index": sdc_track_index,
            "objects_of_interest": objects_of_interest,
            "tracks_to_predict": tracks_to_predict,
        }
    elif metadata["dataset"] == "nuplan":
        sdc_index = [index for index, object in enumerate(objects) if object["is_sdc"]]
        metadata = {
            "sdc_track_index": sdc_index[0],
            "log_name": metadata["log_name"],
            "ts": metadata["ts"],
            "initial_lidar_timestamp": metadata["initial_lidar_timestamp"],
            "map_name": metadata["map"],
            "objects_of_interest": [],
            "tracks_to_predict": [],
            "average_distance_traveled": _ensure_scalar(np.mean(objects_distance_traveled)),
            "scenario_info": metadata["scenario_type"] # for openscenes data this contains the scenario token
        }

    scenario_dict = {
        "name": name,
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict,
        "metadata": metadata,
    }

    return convert_numpy(scenario_dict)

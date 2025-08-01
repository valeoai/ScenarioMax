import numpy as np
import trimesh

from scenariomax.unified_to_gpudrive import constants
from scenariomax.unified_to_gpudrive import utils as gpudrive_utils


FILTERED_ELEMENT_TYPES = ["TRAFFIC_CONE", "TRAFFIC_BARRIER"]

# Object type mapping to match GPUDrive template Waymo protobuf values
WAYMO_OBJECT_TYPE_MAPPING = {
    "TYPE_UNSET": "unset",
    "TYPE_VEHICLE": "vehicle",
    "TYPE_PEDESTRIAN": "pedestrian",
    "TYPE_CYCLIST": "cyclist",
    "TYPE_OTHER": "other",
    # Legacy mappings for unified scenario format
    "VEHICLE": "vehicle",
    "PEDESTRIAN": "pedestrian",
    "CYCLIST": "cyclist",
    "OTHER": "other",
    "UNSET": "unset",
}


def _check_object_distance_traveled(positions, valids):
    valid_positions = positions[valids]
    if len(valid_positions) < 2:
        return 0.0
    diffs = valid_positions[1:] - valid_positions[:-1]
    step_distances = np.linalg.norm(diffs, axis=1)
    return np.sum(step_distances)


def _extract_obj(index, object_id, scenario_net_object):
    state = scenario_net_object["states"]
    type = scenario_net_object["type"]
    valids = state["valid"].astype(bool)
    positions = state["position"]
    headings = state["heading"]
    velocities = state["velocity"]

    position = [
        {"x": point[0], "y": point[1], "z": point[2]}
        if valids[index]
        else {"x": constants.ERR_VAL, "y": constants.ERR_VAL, "z": constants.ERR_VAL}
        for index, point in enumerate(positions)
    ]

    final_valid_index = 0
    for i, is_valid in enumerate(valids):
        if is_valid:
            final_valid_index = i

    length = gpudrive_utils.ensure_scalar(state["length"][final_valid_index])
    width = gpudrive_utils.ensure_scalar(state["width"][final_valid_index])
    height = gpudrive_utils.ensure_scalar(state["height"][final_valid_index])

    heading = [_wrap_yaws(heading) if valids[index] else constants.ERR_VAL for index, heading in enumerate(headings)]

    velocity = [
        {
            "x": point[0],
            "y": point[1],
        }
        if valids[index]
        else {
            "x": constants.ERR_VAL,
            "y": constants.ERR_VAL,
        }
        for index, point in enumerate(velocities)
    ]

    goalPosition = {
        "x": gpudrive_utils.ensure_scalar(positions[final_valid_index][0]),
        "y": gpudrive_utils.ensure_scalar(positions[final_valid_index][1]),
        "z": gpudrive_utils.ensure_scalar(positions[final_valid_index][2]),
    }

    # Standardize object type using GPUDrive template mapping
    normalized_type = WAYMO_OBJECT_TYPE_MAPPING.get(type.upper(), str.lower(type))

    return {
        "id": int(object_id) if object_id.isdigit() else index,
        "type": normalized_type,
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
        "total_distance_traveled": gpudrive_utils.ensure_scalar(_check_object_distance_traveled(positions, valids)),
    }


def _wrap_yaws(yaws):
    """Wraps yaw angles between pi and -pi radians."""
    return (yaws + np.pi) % (2 * np.pi) - np.pi


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

    trajectory_segments = gpudrive_utils.filter_small_segments(trajectory_segments)
    if len(trajectory_segments) > 0:
        trajectory_mesh = gpudrive_utils.generate_mesh(trajectory_segments)
        trajectory_collision_manager.add_object(str(obj["id"]), trajectory_mesh)


def convert_track_features_to_objects(scenario_net_tracks, agent_collision_manager, trajectory_collision_manager):
    objects = []
    objects_distance_traveled = []
    for index, object_id in enumerate(scenario_net_tracks):
        scenario_net_object = scenario_net_tracks[object_id]
        if scenario_net_object["type"] in FILTERED_ELEMENT_TYPES:
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

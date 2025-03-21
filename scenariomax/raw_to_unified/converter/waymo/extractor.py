from typing import Any

import numpy as np

from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.utils import mph_to_kmh
from scenariomax.raw_to_unified.converter.waymo.types import (
    WaymoAgentType,
    WaymoLaneType,
    WaymoRoadEdgeType,
    WaymoRoadLineType,
)
from scenariomax.raw_to_unified.converter.waymo.utils import (
    SPLIT_KEY,
    compute_polygon,
    convert_values_to_str,
    nearest_point,
)
from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.type import ScenarioType


logger = get_logger(__name__)


def convert_waymo_scenario(scenario, version):
    _scenario = SD()

    id_end = scenario.scenario_id.find(SPLIT_KEY)

    _scenario[SD.ID] = scenario.scenario_id[:id_end]
    _scenario[SD.VERSION] = version

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.

    track_length = len(list(scenario.timestamps_seconds))

    tracks, sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index, track_length)

    _scenario[SD.LENGTH] = track_length

    _scenario[SD.TRACKS] = tracks
    _scenario[SD.DYNAMIC_MAP_STATES] = extract_dynamic_map_states(scenario.dynamic_map_states, track_length)
    _scenario[SD.MAP_FEATURES] = extract_map_features(scenario.map_features)

    compute_width(_scenario[SD.MAP_FEATURES])

    _scenario[SD.METADATA] = {}
    _scenario[SD.METADATA][SD.ID] = _scenario[SD.ID]
    _scenario[SD.METADATA][SD.TIMESTEP] = np.asarray(list(scenario.timestamps_seconds), dtype=np.float32)
    _scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    _scenario[SD.METADATA]["dataset"] = "waymo"
    _scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id[:id_end]
    _scenario[SD.METADATA]["source_file"] = scenario.scenario_id[id_end + 1 :]
    _scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    _scenario[SD.METADATA]["current_time_index"] = scenario.current_time_index
    _scenario[SD.METADATA]["sdc_track_index"] = scenario.sdc_track_index

    # obj id
    _scenario[SD.METADATA]["objects_of_interest"] = [str(obj) for obj in scenario.objects_of_interest]

    track_index = [obj.track_index for obj in scenario.tracks_to_predict]
    track_id = [str(scenario.tracks[ind].id) for ind in track_index]
    track_difficulty = [obj.difficulty for obj in scenario.tracks_to_predict]
    track_obj_type = [tracks[id]["type"] for id in track_id]
    _scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": track_difficulty[count],
            "object_type": track_obj_type[count],
        }
        for count, id in enumerate(track_id)
    }

    return _scenario


def extract_tracks(tracks, sdc_idx, track_length):
    tracks_dict = {}

    def _object_state_template(object_id):
        return {
            "type": None,
            "state": {
                # Never add extra dim if the value is scalar.
                "position": np.zeros([track_length, 3], dtype=np.float32),
                "length": np.zeros([track_length], dtype=np.float32),
                "width": np.zeros([track_length], dtype=np.float32),
                "height": np.zeros([track_length], dtype=np.float32),
                "heading": np.zeros([track_length], dtype=np.float32),
                "velocity": np.zeros([track_length, 2], dtype=np.float32),
                "valid": np.zeros([track_length], dtype=bool),
            },
            "metadata": {"track_length": track_length, "type": None, "object_id": object_id, "dataset": "waymo"},
        }

    for obj in tracks:
        object_id = str(obj.id)

        obj_state = _object_state_template(object_id)

        waymo_string = WaymoAgentType.from_waymo(obj.object_type)  # Load waymo type string
        scenarionet_type = ScenarioType.from_waymo(waymo_string)  # Transform it to Waymo type string
        obj_state["type"] = scenarionet_type

        for step_count, state in enumerate(obj.states):
            if step_count >= track_length:
                break

            obj_state["state"]["position"][step_count][0] = state.center_x
            obj_state["state"]["position"][step_count][1] = state.center_y
            obj_state["state"]["position"][step_count][2] = state.center_z

            # l = [state.length for state in obj.states]
            # w = [state.width for state in obj.states]
            # h = [state.height for state in obj.states]
            # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
            obj_state["state"]["length"][step_count] = state.length
            obj_state["state"]["width"][step_count] = state.width
            obj_state["state"]["height"][step_count] = state.height

            # heading = [state.heading for state in obj.states]
            obj_state["state"]["heading"][step_count] = state.heading

            obj_state["state"]["velocity"][step_count][0] = state.velocity_x
            obj_state["state"]["velocity"][step_count][1] = state.velocity_y

            obj_state["state"]["valid"][step_count] = state.valid

        obj_state["metadata"]["type"] = scenarionet_type

        tracks_dict[object_id] = obj_state

    return tracks_dict, str(tracks[sdc_idx].id)


def extract_map_features(map_features: list[Any]) -> dict[str, dict[str, Any]]:
    """Extract map features from Waymo format into a standardized dictionary.

    Args:
        map_features: Waymo lane features

    Returns:
        Dictionary of map features

    """
    map_features_dict = {}

    for lane_state in map_features:
        lane_id = str(lane_state.id)

        if lane_state.HasField("lane"):
            _lane = lane_state.lane
            map_features_dict[lane_id] = {
                "speed_limit_mph": _lane.speed_limit_mph,
                "speed_limit_kmh": mph_to_kmh(_lane.speed_limit_mph),
                "type": WaymoLaneType.from_waymo(_lane.type),
                "polyline": compute_polygon(_lane.polyline),
                "interpolating": _lane.interpolating,
                "entry_lanes": list(_lane.entry_lanes),
                "exit_lanes": list(_lane.exit_lanes),
                "left_boundaries": extract_boundaries(_lane.left_boundaries),
                "right_boundaries": extract_boundaries(_lane.right_boundaries),
                "left_neighbor": extract_neighbors(_lane.left_neighbors),
                "right_neighbor": extract_neighbors(_lane.right_neighbors),
            }

        if lane_state.HasField("road_line"):
            _road_line = lane_state.road_line
            map_features_dict[lane_id] = {
                "type": WaymoRoadLineType.from_waymo(_road_line.type),
                "polyline": compute_polygon(_road_line.polyline),
            }

        if lane_state.HasField("road_edge"):
            _road_egde = lane_state.road_edge
            map_features_dict[lane_id] = {
                "type": WaymoRoadEdgeType.from_waymo(_road_egde.type),
                "polyline": compute_polygon(_road_egde.polyline),
            }

        if lane_state.HasField("stop_sign"):
            _stop_sign = lane_state.stop_sign
            map_features_dict[lane_id] = {
                "type": ScenarioType.STOP_SIGN,
                "lane": [x for x in _stop_sign.lane],
                "position": np.array(
                    [_stop_sign.position.x, _stop_sign.position.y, _stop_sign.position.z],
                    dtype="float32",
                ),
            }

        if lane_state.HasField("crosswalk"):
            _crosswalk = lane_state.crosswalk
            map_features_dict[lane_id] = {
                "type": ScenarioType.CROSSWALK,
                "polygon": compute_polygon(_crosswalk.polygon),
            }

        if lane_state.HasField("speed_bump"):
            _speed_bump = lane_state.speed_bump
            map_features_dict[lane_id] = {
                "type": ScenarioType.SPEED_BUMP,
                "polygon": compute_polygon(_speed_bump.polygon),
            }

        if lane_state.HasField("driveway"):
            _driveway = lane_state.driveway
            map_features_dict[lane_id] = {
                "type": ScenarioType.DRIVEWAY,
                "polygon": compute_polygon(_driveway.polygon),
            }

    return map_features_dict


def extract_dynamic_map_states(dynamic_map_states, track_length):
    processed_dynamics_map_states = {}

    def _traffic_light_state_template(object_id):
        return {
            "type": ScenarioType.TRAFFIC_LIGHT,
            "state": {"object_state": [None] * track_length},
            "lane": None,
            "stop_point": np.zeros([3], dtype=np.float32),
            "metadata": {
                "track_length": track_length,
                "type": ScenarioType.TRAFFIC_LIGHT,
                "object_id": object_id,
                "dataset": "waymo",
            },
        }

    for step_count, step_states in enumerate(dynamic_map_states):
        # Each step_states is the state of all objects in one time step
        lane_states = step_states.lane_states

        if step_count >= track_length:
            break

        for object_state in lane_states:
            lane = object_state.lane
            object_id = str(lane)  # Always use string to specify object id

            # We will use lane index to serve as the traffic light index.
            if object_id not in processed_dynamics_map_states:
                processed_dynamics_map_states[object_id] = _traffic_light_state_template(object_id=object_id)

            if processed_dynamics_map_states[object_id]["lane"] is not None:
                assert lane == processed_dynamics_map_states[object_id]["lane"]
            else:
                processed_dynamics_map_states[object_id]["lane"] = lane

            object_state_string = object_state.State.Name(object_state.state)
            processed_dynamics_map_states[object_id]["state"]["object_state"][step_count] = object_state_string

            processed_dynamics_map_states[object_id]["stop_point"][0] = object_state.stop_point.x
            processed_dynamics_map_states[object_id]["stop_point"][1] = object_state.stop_point.y
            processed_dynamics_map_states[object_id]["stop_point"][2] = object_state.stop_point.z

    for obj in processed_dynamics_map_states.values():
        assert len(obj["state"]["object_state"]) == obj["metadata"]["track_length"]

    return processed_dynamics_map_states


def extract_boundaries(boundaries) -> list[dict[str, Any]]:
    """
    Extract boundary information from Waymo format into a standardized dictionary.

    Args:
        boundaries: Waymo boundary features

    Returns:
        List of boundary information dictionaries
    """
    result = []
    for boundary in boundaries:
        boundary_info = {
            "lane_start_index": boundary.lane_start_index,
            "lane_end_index": boundary.lane_end_index,
            "boundary_type": WaymoRoadLineType.from_waymo(boundary.boundary_type),
            "boundary_feature_id": boundary.boundary_feature_id,
        }
        boundary_info = convert_values_to_str(boundary_info)

        result.append(boundary_info)

    return result


def extract_neighbors(neighbors) -> list[dict[str, Any]]:
    """
    Extract neighbor lane information from Waymo format.

    Args:
        neighbors: Waymo neighbor features

    Returns:
        List of neighbor information dictionaries
    """
    result = []
    for neighbor in neighbors:
        neighbor_info = {
            "feature_id": neighbor.feature_id,
            "self_start_index": neighbor.self_start_index,
            "self_end_index": neighbor.self_end_index,
            "neighbor_start_index": neighbor.neighbor_start_index,
            "neighbor_end_index": neighbor.neighbor_end_index,
        }
        neighbor_info = convert_values_to_str(neighbor_info)
        neighbor_info["boundaries"] = extract_boundaries(neighbor.boundaries)

        result.append(neighbor_info)

    return result


def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0], dtype="float32")

    for b in boundary:
        boundary_int = {k: int(v) if k != "boundary_type" else v for k, v in b.items()}  # All values are int

        b_feat_id = str(boundary_int["boundary_feature_id"])
        lb = map[b_feat_id]
        b_polyline = lb["polyline"][:, :2]

        start_p = polyline[boundary_int["lane_start_index"]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = boundary_int["lane_end_index"] - boundary_int["lane_start_index"]
        end_index = min(start_index + seg_len, lb["polyline"].shape[0] - 1)
        length = min(end_index - start_index, seg_len) + 1
        self_range = range(boundary_int["lane_start_index"], boundary_int["lane_start_index"] + length)
        bound_range = range(start_index, start_index + length)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist

    return l_width


def compute_width(map):
    for map_feat_id, lane in map.items():
        if "LANE" not in lane["type"]:
            continue

        width = np.zeros((lane["polyline"].shape[0], 2), dtype="float32")

        width[:, 0] = extract_width(map, lane["polyline"][:, :2], lane["left_boundaries"])
        width[:, 1] = extract_width(map, lane["polyline"][:, :2], lane["right_boundaries"])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        lane["width"] = width

    return

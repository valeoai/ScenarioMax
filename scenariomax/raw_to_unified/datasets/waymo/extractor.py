from typing import Any

import numpy as np

from scenariomax.core import types
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.raw_to_unified.datasets import utils as converter_utils
from scenariomax.raw_to_unified.datasets.waymo import types as waymo_types
from scenariomax.raw_to_unified.datasets.waymo import utils as waymo_utils


def convert_waymo_scenario(waymo_scenario: Any, version: str) -> UnifiedScenario:
    """Convert WOMD scenario to unified format."""
    # Extract scenario ID from Waymo scenario
    scenario_id = waymo_scenario.scenario_id.split(waymo_utils.SPLIT_KEY)[0]

    # Create unified scenario
    scenario = UnifiedScenario(scenario_id=scenario_id, dataset_name="WOMD", dataset_version=version)

    # Convert data
    dynamic_agents, ego_id = extract_dynamic_agents(waymo_scenario)
    static_map_elements = extract_static_map_elements(waymo_scenario)
    dynamic_map_elements = extract_dynamic_map_elements(waymo_scenario)

    # Populate unified scenario
    scenario["dynamic_agents"] = dynamic_agents
    scenario["static_map_elements"] = static_map_elements
    scenario["dynamic_map_elements"] = dynamic_map_elements

    # Add metadata
    scenario["metadata"].update(
        {
            # General metadata
            "scenario_id": scenario_id,
            "source_file": waymo_scenario.scenario_id.split(waymo_utils.SPLIT_KEY)[1],
            "length": len(waymo_scenario.timestamps_seconds),
            "ego_id": ego_id,
            "timesteps": np.array(waymo_scenario.timestamps_seconds, dtype=np.float32),
            # Waymo-specific metadata
            "current_time_index": waymo_scenario.current_time_index,
            "sdc_track_index": waymo_scenario.sdc_track_index,
            "objects_of_interest": [int(obj) for obj in waymo_scenario.objects_of_interest],
            "tracks_to_predict": [
                {"track_index": track.track_index, "difficulty": track.difficulty}
                for track in waymo_scenario.tracks_to_predict
            ],
        },
    )

    return scenario


def extract_dynamic_agents(waymo_scenario: Any) -> dict[str, dict]:
    """Extract dynamic agent data from Waymo scenario."""
    dynamic_agents = {}

    for track in waymo_scenario.tracks:
        obj_id = str(track.id)  # Use track index as ID

        # Map Waymo object type to unified type
        waymo_type = track.object_type
        unified_type = waymo_types.get_agent_type(waymo_type)

        # Extract states over time
        positions = []
        headings = []
        velocities = []
        valid_flags = []
        length = []
        width = []
        height = []

        for state in track.states:
            positions.append([state.center_x, state.center_y, state.center_z])
            headings.append(state.heading)
            velocities.append([state.velocity_x, state.velocity_y])
            valid_flags.append(state.valid)
            length.append(state.length)
            width.append(state.width)
            height.append(state.height)

        # Convert to numpy arrays
        position = np.array(positions, dtype=np.float32)
        heading = np.array(headings, dtype=np.float32)
        velocity = np.array(velocities, dtype=np.float32)
        valid = np.array(valid_flags, dtype=bool)
        length = np.array(length, dtype=np.float32)
        width = np.array(width, dtype=np.float32)
        height = np.array(height, dtype=np.float32)

        # # Validate arrays
        # position, heading, valid = self.validate_and_convert_arrays(
        #     position,
        #     heading,
        #     valid,
        #     len(waymo_scenario.timestamps_seconds),
        # )

        # Add dynamic agent
        dynamic_agents[obj_id] = {
            "type": unified_type,
            "states": {
                "position": position,
                "heading": heading,
                "velocity": velocity,
                "length": length,
                "width": width,
                "height": height,
                "valid": valid,
            },
        }

    ego_id = str(waymo_scenario.tracks[waymo_scenario.sdc_track_index].id)

    return dynamic_agents, ego_id


def extract_static_map_elements(waymo_scenario: Any) -> dict[str, dict]:
    """Extract static map elements from Waymo scenario."""
    static_map_elements = {}

    for map_feature in waymo_scenario.map_features:
        element_id = str(map_feature.id)

        if map_feature.HasField("lane"):
            # Lane element
            _lane = map_feature.lane
            static_map_elements[element_id] = {
                "type": waymo_types.get_lane_type(_lane.type),
                "polyline": waymo_utils.compute_polygon(_lane.polyline),
                "speed_limit_mph": _lane.speed_limit_mph,
                "speed_limit_kmh": converter_utils.mph_to_kmh(_lane.speed_limit_mph),
                "entry_lanes": list(_lane.entry_lanes),
                "exit_lanes": list(_lane.exit_lanes),
                "left_boundaries": _extract_boundaries(_lane.left_boundaries),
                "right_boundaries": _extract_boundaries(_lane.right_boundaries),
                "left_neighbor": _extract_neighbors(_lane.left_neighbors),
                "right_neighbor": _extract_neighbors(_lane.right_neighbors),
            }

        elif map_feature.HasField("road_line"):
            # Road line element
            _road_line = map_feature.road_line

            static_map_elements[element_id] = {
                "type": waymo_types.get_road_line_type(_road_line.type),
                "polyline": waymo_utils.compute_polygon(_road_line.polyline),
            }

        elif map_feature.HasField("road_edge"):
            # Road edge element
            _road_edge = map_feature.road_edge
            static_map_elements[element_id] = {
                "type": waymo_types.get_road_edge_type(_road_edge.type),
                "polyline": waymo_utils.compute_polygon(_road_edge.polyline),
            }

        elif map_feature.HasField("crosswalk"):
            # Crosswalk element
            _crosswalk = map_feature.crosswalk

            static_map_elements[element_id] = {
                "type": types.CROSSWALK,
                "polygon": waymo_utils.compute_polygon(_crosswalk.polygon),
            }

        elif map_feature.HasField("stop_sign"):
            # Stop sign element
            _stop_sign = map_feature.stop_sign

            static_map_elements[element_id] = {
                "type": types.STOP_SIGN,
                "lane": [x for x in _stop_sign.lane],
                "position": np.array(
                    [_stop_sign.position.x, _stop_sign.position.y, _stop_sign.position.z],
                    dtype="float32",
                ),
            }
        elif map_feature.HasField("speed_bump"):
            # Speed bump element
            _speed_bump = map_feature.speed_bump

            static_map_elements[element_id] = {
                "type": types.SPEED_BUMP,
                "polygon": waymo_utils.compute_polygon(_speed_bump.polygon),
            }
        elif map_feature.HasField("driveway"):
            # Driveway element
            _driveway = map_feature.driveway

            static_map_elements[element_id] = {
                "type": types.DRIVEWAY,
                "polygon": waymo_utils.compute_polygon(_driveway.polygon),
            }

    return static_map_elements


def extract_dynamic_map_elements(waymo_scenario: Any) -> dict[str, dict]:
    """Extract dynamic map element data from Waymo scenario."""
    dynamic_map_elements = {}

    # Each step_states is the state of all objects in one time step
    for i, step_states in enumerate(waymo_scenario.dynamic_map_states):
        lane_states = step_states.lane_states

        for traffic_light_states in lane_states:
            lane = traffic_light_states.lane
            traffic_light_id = str(lane)

            if traffic_light_id not in dynamic_map_elements:
                dynamic_map_elements[traffic_light_id] = {
                    "type": types.TRAFFIC_LIGHT,
                    "position": np.array(
                        [
                            traffic_light_states.stop_point.x,
                            traffic_light_states.stop_point.y,
                            traffic_light_states.stop_point.z,
                        ],
                        dtype="float32",
                    ),
                    "states": [None] * len(waymo_scenario.timestamps_seconds),
                    "lane": lane,
                }

            # Map traffic light state
            state = waymo_types.get_traffic_light_state(traffic_light_states.state)
            dynamic_map_elements[traffic_light_id]["states"][i] = state

    assert i == len(waymo_scenario.timestamps_seconds) - 1, "Mismatch in number of time steps"

    return dynamic_map_elements


def _extract_boundaries(boundaries) -> list[dict[str, Any]]:
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
            "boundary_type": waymo_types.get_road_line_type(boundary.boundary_type),
            "boundary_feature_id": boundary.boundary_feature_id,
        }
        boundary_info = waymo_utils.convert_values_to_str(boundary_info)

        result.append(boundary_info)

    return result


def _extract_neighbors(neighbors) -> list[dict[str, Any]]:
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
        neighbor_info = waymo_utils.convert_values_to_str(neighbor_info)
        neighbor_info["boundaries"] = _extract_boundaries(neighbor.boundaries)

        result.append(neighbor_info)

    return result

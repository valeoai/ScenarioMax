"""
Convert dynamic agents from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types
from scenariomax.stage3_format.pufferdrive.converter import routes


logger = logger_utils.get_logger(__name__)


def convert_dynamic_agents(
    dynamic_agents: dict,
    road_map_elements: dict,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> list[dict]:
    """
    Convert dynamic agents from unified format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from unified scenario
        road_map_elements: Dict of static map elements (for reference)
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        route_check_timestep: Timestep at which agent must be valid for route computation (default: 0)

    Returns:
        List of dynamic agent dictionaries in Puffer format
    """
    puffer_agents = []

    # Extract lane centers once for all agents (optimization)
    lane_data = routes.extract_lane_centers(road_map_elements)

    for idx, (agent_id, agent_data) in enumerate(dynamic_agents.items()):
        states = agent_data["states"]

        # Get position data (x, y, z)
        position = states["position"]
        if position.shape[1] == 2:
            # Add z=0 if only x,y provided
            position = np.column_stack([position, np.zeros(len(position))])

        # Get heading, velocity, dimensions
        heading = states["heading"]
        velocity = states["velocity"]
        agent_length = states["length"]
        width = states["width"]
        height = states["height"]
        valid = states["valid"]

        # Convert agent type to int
        agent_type_int = _convert_agent_type_to_int(agent_data["type"])

        # Routes are computed only for:
        # 1. VEHICLE type (type == 1)
        # 2. Agents valid at route_check_timestep (configurable, default: 0)
        # 3. Agents with sufficient valid trajectory points (configurable, default: 0)
        # 4. Agents not offroad (bbox crosses road edge OR >5m from lane) - checked inside compute_agent_route()
        should_compute_routes = (
            agent_type_int == 1
            and route_check_timestep < len(valid)
            and valid[route_check_timestep]
            and np.sum(valid) >= min_route_valid_points
        )

        if should_compute_routes:
            agent_routes = _compute_routes(
                (agent_id, position, heading, valid, agent_length, width),
                road_map_elements,
                lane_data,
                min_route_valid_points,
                route_check_timestep,
            )
        else:
            agent_routes = []

        puffer_agent = {
            "id": idx,  # Use int ID directly
            "type": agent_type_int,
            "states": {
                "xyz": position,
                "heading": heading,
                "velocity": velocity,
                "length": agent_length,
                "width": width,
                "height": height,
                "valid": valid,
            },
            "routes": agent_routes,
        }

        puffer_agents.append(puffer_agent)

    return puffer_agents


def _convert_agent_type_to_int(agent_type: str) -> int:
    """
    Convert agent type string to integer.

    Args:
        agent_type: Agent type string from types.py

    Returns:
        Integer representation
    """
    type_map = {
        types.VEHICLE: 1,
        types.PEDESTRIAN: 2,
        types.CYCLIST: 3,
        types.OTHER: 4,
    }
    return type_map.get(agent_type, 0)  # Default to 0 for unknown


def _compute_routes(
    agent_data: tuple,
    road_map_elements: dict,
    lane_data: tuple,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> list:
    """
    Compute route an agent follows based on ground truth trajectory.

    Route is a list of lane center IDs that covers the ground truth trajectory
    and extends beyond using greedy lane selection.

    Args:
        agent_data: Tuple of (agent_id, position, heading, valid, length, width)
        road_map_elements: Dict of static map elements (for reference)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        min_route_valid_points: Minimum valid trajectory points required (0 = no filtering)
        route_check_timestep: Timestep to check if agent is offroad (default: 0)

    Returns:
        List containing single route, where route is a list of lane IDs
    """
    # Compute single route using greedy algorithm
    # Returns list with one route: [[lane1, lane2, ...]]
    route_paths = routes.compute_agent_route(
        agent_data=agent_data,
        static_map_elements=road_map_elements,
        lane_data=lane_data,
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    # Return list of route paths
    if not route_paths:
        return []

    # Lane IDs are already integers, return directly
    return route_paths

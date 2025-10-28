"""
Convert dynamic agents from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types
from scenariomax.stage3_format.puffer.converter import routes


logger = logger_utils.get_logger(__name__)


def convert_dynamic_agents(dynamic_agents: dict, road_map_elements: dict, length: int, ego_id: str) -> list[dict]:
    """
    Convert dynamic agents from unified format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from unified scenario
        road_map_elements: Dict of static map elements (for reference)
        length: Number of timesteps
        ego_id: ID of ego vehicle

    Returns:
        List of dynamic agent dictionaries in Puffer format
    """
    puffer_agents = []

    # Extract lane centers once for all agents (optimization)
    lane_data = routes.extract_lane_centers(road_map_elements)

    # Put ego agent first
    sorted_agent_items = sorted(dynamic_agents.items(), key=lambda item: (item[0] != ego_id, item[0]))

    for idx, (agent_id, agent_data) in enumerate(sorted_agent_items):
        states = agent_data.get("states", {})

        # Get position data (x, y, z)
        position = states.get("position", np.zeros((length, 3)))
        if position.shape[1] == 2:
            # Add z=0 if only x,y provided
            position = np.column_stack([position, np.zeros(len(position))])

        # Get heading, velocity, dimensions
        heading = states.get("heading", np.zeros(length))
        velocity = states.get("velocity", np.zeros((length, 2)))
        agent_length = states.get("length", np.zeros(length))
        width = states.get("width", np.zeros(length))
        height = states.get("height", np.zeros(length))
        valid = states.get("valid", np.ones(length, dtype=bool))

        # Calculate total distance traveled
        total_distance = _calculate_distance_traveled(position, valid)

        # Determine if this agent should be marked as expert (ego vehicle)
        mark_as_expert = agent_id == ego_id

        # Convert agent type to int
        agent_type_int = _convert_agent_type_to_int(agent_data.get("type", "TYPE_UNSET"))

        # Routes are computed only for:
        # 1. VEHICLE type (type == 1)
        # 2. Agents with sufficient valid trajectory points (>10)
        # 3. Agents close to lanes (within 2m) - checked inside compute_agent_route()
        # 4. Agents not off-map/parked (≥50% of trajectory near lanes) - checked inside compute_agent_route()
        if agent_type_int == 1 and np.sum(valid) > 10:
            agent_routes = _compute_routes(position, heading, valid, road_map_elements, lane_data, agent_id)
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
            "mark_as_expert": mark_as_expert,
            "total_distance_traveled": total_distance,
        }

        puffer_agents.append(puffer_agent)

    return puffer_agents


def _calculate_distance_traveled(position: np.ndarray, valid: np.ndarray) -> float:
    """
    Calculate total distance traveled by an agent.

    Args:
        position: (N, 3) array of positions
        valid: (N,) boolean array indicating valid timesteps

    Returns:
        Total distance traveled in meters
    """
    if len(position) < 2:
        return 0.0

    # Calculate distances between consecutive valid positions
    distances = np.linalg.norm(np.diff(position[valid], axis=0), axis=1)
    return float(np.sum(distances))


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
    position: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    road_map_elements: dict,
    lane_data: tuple,
    agent_id: int | str,
) -> list:
    """
    Compute routes an agent follows based on ground truth trajectory.

    Routes are lists of lane center IDs that:
    1. Cover the ground truth trajectory
    2. Extend beyond the trajectory using lane connectivity
    3. Explore multiple possible paths through exit lanes

    Args:
        position: Agent position trajectory (N, 3) array
        heading: Agent heading at each timestep (N,) array
        valid: Validity mask for trajectory (N,) array
        road_map_elements: Dict of static map elements (for reference)
        lane_data: Precomputed lane data (lane_ids, lane_polylines, lane_metadata)
        agent_id: Agent identifier for debugging

    Returns:
        List of route paths, where each path is a list of lane IDs
    """
    # Compute routes using the new route computation algorithm
    # Returns list of route paths: [[lane1, lane2, ...], [lane1, lane3, ...], ...]
    route_paths = routes.compute_agent_route(
        agent_trajectory=position,
        agent_heading=heading,
        agent_valid=valid,
        static_map_elements=road_map_elements,
        lane_data=lane_data,
        agent_id=agent_id,
    )

    # Return list of route paths
    if not route_paths:
        return []

    # Lane IDs are already integers, return directly
    return route_paths

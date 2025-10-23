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

        # Compute routes based on ground truth trajectory
        agent_routes = _compute_routes(position, heading, valid, road_map_elements)

        puffer_agent = {
            "id": idx,  # Use int ID directly
            "type": agent_type_int,
            "states": {
                "xyz": position.astype(np.float32),
                "heading": heading.astype(np.float32),
                "velocity": velocity.astype(np.float32),
                "length": agent_length.astype(np.float32),
                "width": width.astype(np.float32),
                "height": height.astype(np.float32),
                "valid": valid.astype(bool),
            },
            "routes": agent_routes,
            "mark_as_expert": mark_as_expert,
            "total_distance_traveled": float(total_distance),
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
) -> dict:
    """
    Compute routes an agent follows based on ground truth trajectory.

    Routes are lists of lane center IDs that:
    1. Cover the ground truth trajectory
    2. Extend beyond the trajectory using lane connectivity

    Args:
        position: Agent position trajectory (N, 3) array
        heading: Agent heading at each timestep (N,) array
        valid: Validity mask for trajectory (N,) array
        road_map_elements: Dict of static map elements (for reference)

    Returns:
        Dict of routes in Puffer format ({"0": {"lanes": [lane_ids]}})
    """
    # Compute route using the new route computation algorithm
    route_lane_ids = routes.compute_agent_route(
        agent_trajectory=position,
        agent_heading=heading,
        agent_valid=valid,
        static_map_elements=road_map_elements,
    )

    # Format as Puffer route dict
    if not route_lane_ids:
        return []

    # Lane IDs are already integers, return directly
    return route_lane_ids

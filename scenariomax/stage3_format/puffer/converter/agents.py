"""
Convert dynamic agents from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


def convert_dynamic_agents(dynamic_agents: dict, length: int, ego_id: str) -> list[dict]:
    """
    Convert dynamic agents from unified format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from unified scenario
        length: Number of timesteps
        ego_id: ID of ego vehicle

    Returns:
        List of dynamic agent dictionaries in Puffer format
    """
    puffer_agents = []

    for agent_id, agent_data in dynamic_agents.items():
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

        puffer_agent = {
            "id": int(hash(agent_id) & 0x7FFFFFFF),  # Convert string ID to positive int
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

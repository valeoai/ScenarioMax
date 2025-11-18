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
    length: int,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
    max_routes: int = 10,
) -> list[dict]:
    """
    Convert dynamic agents from unified format to Puffer format.

    Args:
        dynamic_agents: Dict of dynamic agents from unified scenario
        road_map_elements: Dict of static map elements (for reference)
        length: Number of timesteps
        ego_id: ID of ego vehicle
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        route_check_timestep: Timestep at which agent must be valid for route computation (default: 0)
        max_routes: Number of route paths to generate per agent (default: 10)

    Returns:
        List of dynamic agent dictionaries in Puffer format
    """
    if dynamic_map_elements is None:
        dynamic_map_elements = {}

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
        # 4. Agents close to lanes (within 2m) - checked inside compute_agent_route()
        # 5. Agents not off-map/parked (≥50% of trajectory near lanes) - checked inside compute_agent_route()
        should_compute_routes = (
            agent_type_int == 1
            and route_check_timestep < len(valid)
            and valid[route_check_timestep]
            and np.sum(valid) >= min_route_valid_points
        )

        if should_compute_routes:
            agent_routes = _compute_routes(
                position,
                heading,
                valid,
                road_map_elements,
                lane_data,
                agent_id,
                min_route_valid_points,
                max_routes,
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
    position: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    road_map_elements: dict,
    lane_data: tuple,
    agent_id: int | str,
    min_route_valid_points: int = 0,
    max_routes: int = 10,
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
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        max_routes: Number of route paths to generate per agent (default: 10)

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
        min_route_valid_points=min_route_valid_points,
        max_routes=max_routes,
    )

    # Return list of route paths
    if not route_paths:
        return []

    # Lane IDs are already integers, return directly
    return route_paths

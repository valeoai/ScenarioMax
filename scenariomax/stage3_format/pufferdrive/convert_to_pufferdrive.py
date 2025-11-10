"""
Convert unified scenarios to Puffer format.

Puffer is a simulator format that requires specific data structures for
dynamic agents, road map elements, and traffic control elements.
Output is JSON format with numpy arrays converted to lists.
"""

import numpy as np
from scenariomax.stage3_format.puffer.converter import agents, roadgraph, traffic_lights

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def convert(
    unified_scenario,
    polyline_reduction_threshold: float = 0.1,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> dict:
    """
    Convert a UnifiedScenario to Puffer format.

    Args:
        unified_scenario: UnifiedScenario object or dict
        polyline_reduction_threshold: Minimum triangle area threshold for roadgraph polyline simplification.
                                       If 0.0 (default), no simplification is applied.
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)
        route_check_timestep: Timestep at which agent must be valid for route computation (default: 0)

    Returns:
        Dictionary in Puffer format with dynamic_agents, road_map_elements,
        traffic_control_elements, and metadata
    """
    # Runtime type validation
    if not isinstance(unified_scenario, dict):
        raise TypeError(f"Expected unified_scenario to be dict, got {type(unified_scenario).__name__}")

    # Validate required fields
    required_fields = ["id", "dynamic_agents", "static_map_elements", "dynamic_map_elements", "metadata"]
    missing_fields = [f for f in required_fields if f not in unified_scenario]
    if missing_fields:
        raise ValueError(f"unified_scenario missing required fields: {missing_fields}")

    scenario_id = unified_scenario.get("id", "")
    if not scenario_id:
        logger.warning("Scenario has empty ID")

    scenario_metadata = unified_scenario.get("metadata", {})

    # Convert static map elements to road_map_elements
    road_map_elements = roadgraph.convert_road_map_elements(
        unified_scenario.get("static_map_elements", {}),
        polyline_reduction_threshold,
    )

    # Convert dynamic agents
    dynamic_agents = agents.convert_dynamic_agents(
        unified_scenario.get("dynamic_agents", {}),
        unified_scenario.get("static_map_elements", {}),
        scenario_metadata.get("length", 0),
        min_route_valid_points=min_route_valid_points,
        route_check_timestep=route_check_timestep,
    )

    # Convert dynamic map elements to traffic_control_elements
    traffic_control_elements = traffic_lights.convert_traffic_control_elements(
        unified_scenario.get("dynamic_map_elements", {}),
        scenario_metadata.get("length", 0),
    )

    # Convert metadata
    metadata = unified_scenario.get("metadata", {})
    puffer_metadata = {
        "dataset_name": metadata.get("dataset_name", ""),
        "length": metadata.get("length", 0),
        "timesteps": metadata.get("timesteps", np.array([])),
        "ego_id": metadata.get("ego_id", ""),
    }

    # Add Waymo-specific metadata if available
    if metadata.get("dataset_name") == "waymo":
        objects_of_interest = metadata.get("objects_of_interest", [])
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        puffer_metadata["objects_of_interests"] = [int(oi) for oi in objects_of_interest]
        puffer_metadata["tracks_to_predict"] = [t.get("track_index") for t in tracks_to_predict]

    puffer_scenario = {
        "scenario_id": scenario_id,
        "dynamic_agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": puffer_metadata,
    }

    return puffer_scenario

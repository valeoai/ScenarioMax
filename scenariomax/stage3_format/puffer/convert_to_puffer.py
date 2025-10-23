"""
Convert unified scenarios to Puffer format.

Puffer is a simulator format that requires specific data structures for
dynamic agents, road map elements, and traffic control elements.
Output is JSON format with numpy arrays converted to lists.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.stage3_format.puffer.converter import agents, roadgraph, traffic_lights, utils


logger = logger_utils.get_logger(__name__)


def convert(unified_scenario) -> dict:
    """
    Convert a UnifiedScenario to Puffer format.

    Args:
        unified_scenario: UnifiedScenario object or dict

    Returns:
        Dictionary in Puffer format with dynamic_agents, road_map_elements,
        traffic_control_elements, and metadata
    """
    scenario_id = unified_scenario.get("id", "")
    scenario_metadata = unified_scenario.get("metadata", {})

    # Convert static map elements to road_map_elements
    road_map_elements = roadgraph.convert_road_map_elements(unified_scenario.get("static_map_elements", {}))

    # Convert dynamic agents
    dynamic_agents = agents.convert_dynamic_agents(
        unified_scenario.get("dynamic_agents", {}),
        unified_scenario.get("static_map_elements", {}),
        scenario_metadata.get("length", 0),
        scenario_metadata.get("ego_id", ""),
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
        "dataset_version": metadata.get("dataset_version", ""),
        "source_file": metadata.get("source_file", ""),
        "length": metadata.get("length", 0),
        "timesteps": metadata.get("timesteps", np.array([])).astype(np.float32),
        "ego_id": metadata.get("ego_id", ""),
    }

    # Add Waymo-specific metadata if available
    if metadata.get("dataset_name") == "waymo":
        objects_of_interest = metadata.get("objects_of_interest", [])
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        puffer_metadata["objects_of_interests"] = [int(oi) for oi in objects_of_interest]

        # Convert tracks_to_predict format
        if tracks_to_predict and isinstance(tracks_to_predict[0], dict):
            puffer_metadata["tracks_to_predict"] = [
                {"track_index": int(t.get("track_index", 0)), "difficulty": float(t.get("difficulty", 0))}
                for t in tracks_to_predict
            ]
        else:
            puffer_metadata["tracks_to_predict"] = [{"track_index": int(t), "difficulty": 0.0} for t in tracks_to_predict]

    puffer_scenario = {
        "scenario_id": scenario_id,
        "dynamic_agents": dynamic_agents,
        "road_map_elements": road_map_elements,
        "traffic_control_elements": traffic_control_elements,
        "metadata": puffer_metadata,
    }

    # Convert all numpy arrays to lists for JSON serialization
    puffer_scenario = utils.convert_numpy_to_json(puffer_scenario)

    return puffer_scenario

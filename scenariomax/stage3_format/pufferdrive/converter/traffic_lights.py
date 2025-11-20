"""
Convert traffic control elements from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


def convert_traffic_control_elements(dynamic_map_elements: dict, static_map_elements: dict) -> list[dict]:
    """
    Convert dynamic map elements to Puffer traffic_control_elements.

    Args:
        dynamic_map_elements: Dict of dynamic map elements from unified scenario
        length: Number of timesteps

    Returns:
        List of traffic control element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in dynamic_map_elements.items():
        element_type = element_data["type"]
        position = element_data["position"]
        states = element_data["states"]
        controlled_lane = element_data["controlled_lane"]

        # Convert traffic light type to int
        element_type_int = _convert_traffic_control_type_to_int(element_type)

        # Convert states to int array
        # States might be a list or numpy array
        states_list = states.tolist() if isinstance(states, np.ndarray) else states

        states_int = [_convert_traffic_light_state_to_int(s) if isinstance(s, str) else int(s) for s in states_list]
        states_int = np.array(states_int, dtype=np.int32)

        # Normalize controlled_lane to list (PufferDrive expects list)
        if isinstance(controlled_lane, int):
            controlled_lanes = [controlled_lane]
        elif isinstance(controlled_lane, list):
            controlled_lanes = controlled_lane
        else:
            raise TypeError(f"controlled_lane must be int or list[int], got {type(controlled_lane).__name__}")

        puffer_element = {
            "id": int(element_id),
            "type": element_type_int,
            "xyz": position,
            "states": states_int,
            "controlled_lanes": controlled_lanes,
        }

        puffer_elements.append(puffer_element)

    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        # Convert traffic light type to int
        element_type_int = _convert_traffic_control_type_to_int(element_type)

        if element_type_int == 0:
            continue

        position = element_data["position"]
        lanes = element_data["lanes"]

        # Normalize lanes to list (PufferDrive expects list)
        if isinstance(lanes, int):
            controlled_lanes = [lanes]
        elif isinstance(lanes, list):
            controlled_lanes = lanes
        else:
            raise TypeError(f"lanes must be int or list[int], got {type(lanes).__name__}")

        puffer_element = {
            "id": int(element_id),
            "type": element_type_int,
            "xyz": position,
            "states": [],
            "controlled_lanes": controlled_lanes,
        }

        puffer_elements.append(puffer_element)

    return puffer_elements


def _convert_traffic_control_type_to_int(element_type: str) -> int:
    """
    Convert traffic light type string to integer.

    Args:
        traffic_light_type: Traffic light type string from types.py

    Returns:
        Integer representation
    """
    # Map traffic light states to types
    type_map = {
        types.TRAFFIC_LIGHT: 1,
        types.STOP_SIGN: 2,
        types.YIELD_SIGN: 3,
        types.TRAFFIC_CONE: 4,
        types.TRAFFIC_BARRIER: 5,
        types.GUARDRAIL: 6,
    }
    return type_map.get(element_type, 0)


def _convert_traffic_light_state_to_int(state: str) -> int:
    """
    Convert traffic light state string to integer.

    Args:
        state: Traffic light state string from types.py

    Returns:
        Integer representation
    """
    state_map = {
        types.TRAFFIC_LIGHT_UNKNOWN: 0,
        types.TRAFFIC_LIGHT_ARROW_RED: 1,
        types.TRAFFIC_LIGHT_ARROW_YELLOW: 2,
        types.TRAFFIC_LIGHT_ARROW_GREEN: 3,
        types.TRAFFIC_LIGHT_RED: 4,
        types.TRAFFIC_LIGHT_YELLOW: 5,
        types.TRAFFIC_LIGHT_GREEN: 6,
        types.TRAFFIC_LIGHT_FLASHING_RED: 7,
        types.TRAFFIC_LIGHT_FLASHING_YELLOW: 8,
    }
    return state_map.get(state, 0)

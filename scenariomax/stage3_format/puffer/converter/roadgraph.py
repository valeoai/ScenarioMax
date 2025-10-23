"""
Convert road map elements from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


def convert_road_map_elements(static_map_elements: dict) -> list[dict]:
    """
    Convert static map elements from unified format to Puffer road_map_elements.

    Args:
        static_map_elements: Dict of static map elements from unified scenario

    Returns:
        List of road map element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        element_type = element_data.get("type")

        if not element_type:
            logger.warning(f"Skipping map element with unset type: {element_id}")
            continue

        polyline = element_data.get("polyline", np.zeros((0, 3)))

        # Ensure polyline has 3D coordinates
        if polyline.shape[1] == 2:
            polyline = np.column_stack([polyline, np.zeros(len(polyline))])

        # Calculate direction vectors
        dir_xyz = _calculate_direction_vectors(polyline)

        # Convert element type to int
        element_type_int = _convert_map_element_type_to_int(element_type)

        puffer_element = {
            "id": element_id,  # Use int ID directly
            "type": element_type_int,
            "xyz": polyline.astype(np.float32),
            "dir_xyz": dir_xyz.astype(np.float32),
        }

        # Add lane-specific attributes if this is a lane
        if types.is_lane(element_type):
            puffer_element["speed_limit_mph"] = float(element_data.get("speed_limit_mph", 0.0))
            puffer_element["speed_limit_kmh"] = float(element_data.get("speed_limit_kmh", 0.0))

            # Convert lane connectivity (entry/exit/neighbors)
            entry_lanes = element_data.get("entry_lanes", [])
            exit_lanes = element_data.get("exit_lanes", [])
            left_neighbor = element_data.get("left_neighbor", [])
            right_neighbor = element_data.get("right_neighbor", [])

            # Use int IDs directly
            puffer_element["entry"] = list(entry_lanes) if entry_lanes else []
            puffer_element["exit"] = list(exit_lanes) if exit_lanes else []

            # Combine left and right neighbors
            neighbors = []
            if left_neighbor:
                neighbors.extend(left_neighbor)
            if right_neighbor:
                neighbors.extend(right_neighbor)
            puffer_element["neighbors"] = neighbors

        puffer_elements.append(puffer_element)

    return puffer_elements


def _calculate_direction_vectors(polyline: np.ndarray) -> np.ndarray:
    """
    Calculate direction vectors for a polyline.

    Args:
        polyline: (N, 3) array of 3D points

    Returns:
        (N, 3) array of direction vectors (normalized)
    """
    if len(polyline) < 2:
        return np.zeros_like(polyline)

    # Calculate differences between consecutive points
    directions = np.diff(polyline, axis=0)

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    directions = directions / norms

    # Duplicate last direction for last point
    directions = np.vstack([directions, directions[-1]])

    return directions


def _convert_map_element_type_to_int(element_type: str) -> int:
    """
    Convert map element type string to integer.

    Args:
        element_type: Map element type string from types.py

    Returns:
        Integer representation
    """
    # Lane types (1-10)
    lane_type_map = {
        types.LANE_UNKNOWN: 0,
        types.LANE_FREEWAY: 1,
        types.LANE_SURFACE_STREET: 2,
        types.LANE_BIKE_LANE: 3,
    }

    # Road line types (11-20)
    road_line_type_map = {
        types.ROAD_LINE_UNKNOWN: 10,
        types.ROAD_LINE_BROKEN_SINGLE_WHITE: 11,
        types.ROAD_LINE_SOLID_SINGLE_WHITE: 12,
        types.ROAD_LINE_SOLID_DOUBLE_WHITE: 13,
        types.ROAD_LINE_BROKEN_SINGLE_YELLOW: 14,
        types.ROAD_LINE_BROKEN_DOUBLE_YELLOW: 15,
        types.ROAD_LINE_SOLID_SINGLE_YELLOW: 16,
        types.ROAD_LINE_SOLID_DOUBLE_YELLOW: 17,
        types.ROAD_LINE_PASSING_DOUBLE_YELLOW: 18,
    }

    # Road edge types (21-30)
    road_edge_type_map = {
        types.ROAD_EDGE_UNKNOWN: 20,
        types.ROAD_EDGE_BOUNDARY: 21,
        types.ROAD_EDGE_MEDIAN: 22,
        types.ROAD_EDGE_SIDEWALK: 23,
    }

    # Other map element types (31+)
    other_type_map = {
        types.CROSSWALK: 31,
        types.SPEED_BUMP: 32,
        types.STOP_SIGN: 33,
        types.DRIVEWAY: 34,
    }

    # Check all mappings
    for type_map in [lane_type_map, road_line_type_map, road_edge_type_map, other_type_map]:
        if element_type in type_map:
            return type_map[element_type]

    return 0  # Default to undefined

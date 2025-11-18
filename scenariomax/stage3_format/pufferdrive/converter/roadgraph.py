"""
Convert road map elements from unified format to Puffer format.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


def calculate_area(p1: dict, p2: dict, p3: dict) -> float:
    """
    Calculate the area of the triangle using the determinant method.

    Args:
        p1: First point dict with 'x' and 'y' keys
        p2: Second point dict with 'x' and 'y' keys
        p3: Third point dict with 'x' and 'y' keys

    Returns:
        Triangle area
    """
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def simplify_polyline(geometry: list[dict], polyline_reduction_threshold: float) -> list[dict]:
    """
    Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python.

    Args:
        geometry: List of point dicts with 'x' and 'y' keys
        polyline_reduction_threshold: Minimum triangle area threshold for point removal

    Returns:
        Simplified polyline (list of point dicts)
    """
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)

            if area < polyline_reduction_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def convert_road_map_elements(static_map_elements: dict, polyline_reduction_threshold: float = 0.1) -> list[dict]:
    """
    Convert static map elements from unified format to Puffer road_map_elements.

    Args:
        static_map_elements: Dict of static map elements from unified scenario
        polyline_reduction_threshold: Minimum triangle area threshold for polyline simplification.
                                       If 0.0 (default), no simplification is applied.

    Returns:
        List of road map element dictionaries in Puffer format
    """
    puffer_elements = []

    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        if element_type in ["DRIVEWAY", "SPEED_BUMP", "STOP_SIGN", "CROSSWALK"]:
            # Skip driveways as they are not supported in Puffer format
            continue

        if not element_type:
            logger.warning(f"Skipping map element with unset type: {element_id}")
            continue

        polyline = element_data["polyline"]

        # Ensure polyline has 3D coordinates
        if polyline.shape[1] == 2:
            polyline = np.column_stack([polyline, np.zeros(len(polyline))])

        # Apply polyline reduction if threshold is set
        if polyline_reduction_threshold > 0.0 and len(polyline) >= 3:
            # Convert numpy array to list of dicts for simplification algorithm
            geometry = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in polyline]
            simplified_geometry = simplify_polyline(geometry, polyline_reduction_threshold)
            # Convert back to numpy array
            polyline = np.array([[p["x"], p["y"], p["z"]] for p in simplified_geometry])

        # Convert element type to int
        element_type_int = _convert_map_element_type_to_int(element_type)

        puffer_element = {
            "id": element_id,
            "type": element_type_int,
            "xyz": polyline,
        }

        # Add lane-specific attributes if this is a lane
        if types.is_lane(element_type):
            # Convert speed limit from km/h to m/s
            speed_limit_kmh = element_data["speed_limit_kmh"]
            puffer_element["speed_limit"] = speed_limit_kmh / 3.6  # m/s

            # Convert lane connectivity (entry/exit/neighbors)
            left_neighbor = element_data["left_neighbor"]
            right_neighbor = element_data["right_neighbor"]

            # Use int IDs directly
            puffer_element["entry_lanes"] = element_data["entry_lanes"]
            puffer_element["exit_lanes"] = element_data["exit_lanes"]

            # Combine left and right neighbors
            neighbors = []
            if left_neighbor:
                neighbors.extend(left_neighbor)
            if right_neighbor:
                neighbors.extend(right_neighbor)
            puffer_element["neighbors"] = neighbors

        puffer_elements.append(puffer_element)

    return puffer_elements


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

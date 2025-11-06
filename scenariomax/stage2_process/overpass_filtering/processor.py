from typing import Any

import numpy as np
from scipy.spatial import KDTree

from scenariomax import logger_utils
from scenariomax.core.types import is_road_edge


logger = logger_utils.get_logger(__name__)


class OverpassDetectedException(Exception):
    """Raised when an overpass is detected in a scenario."""


def detect_overpass_in_scenario(
    unified_scenario: dict[str, Any],
    xy_distance_threshold: float = 0.8,
    z_difference_threshold: float = 4.0,
    **kwargs,
) -> dict[str, Any]:
    """
    Detect overpasses in a scenario by analyzing road edge elements.

    Args:
        unified_scenario: The scenario to check
        xy_distance_threshold: Max XY distance (meters) to check for vertical separation
        z_difference_threshold: Min Z difference (meters) to consider as overpass

    Returns:
        The unchanged scenario if no overpass is detected

    Raises:
        OverpassDetectedException: If an overpass is detected
    """
    scenario_id = unified_scenario.get("id", "<unknown>")
    static_map_elements = unified_scenario.get("static_map_elements", {})

    # Extract all road edge polylines
    road_edge_points = []
    for element_id, element in static_map_elements.items():
        element_type = element.get("type")
        if not element_type or not is_road_edge(element_type):
            continue

        polyline = element.get("polyline")
        if polyline is None:
            continue

        polyline = np.array(polyline)
        if len(polyline.shape) == 1:
            polyline = np.expand_dims(polyline, axis=0)

        # Ensure 3D coordinates
        if polyline.shape[1] == 2:
            polyline = np.hstack([polyline, np.zeros((polyline.shape[0], 1))])

        road_edge_points.append(polyline)

    # If not enough road edge points, no overpass
    if not road_edge_points:
        return unified_scenario

    # Concatenate all road edge points
    all_points = np.vstack(road_edge_points)

    if len(all_points) < 2:
        return unified_scenario

    # Build KD-tree on XY coordinates
    tree = KDTree(all_points[:, :2])

    # Check for overpass
    for i, point in enumerate(all_points):
        neighbors = tree.query_ball_point(point[:2], xy_distance_threshold)
        neighbors = [idx for idx in neighbors if idx > i]

        if not neighbors:
            continue

        z_differences = np.abs(point[2] - all_points[neighbors, 2])
        if np.any(z_differences > z_difference_threshold):
            raise OverpassDetectedException(f"Overpass detected in scenario {scenario_id}")

    return unified_scenario

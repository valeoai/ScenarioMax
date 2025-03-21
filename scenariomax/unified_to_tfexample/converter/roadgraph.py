import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from scenariomax.unified_to_tfexample.constants import DEFAULT_NUM_ROADMAPS, DIST_INTERPOLATION
from scenariomax.unified_to_tfexample.converter.datatypes import RoadGraphSamples
from scenariomax.unified_to_tfexample.exceptions import OverpassException


warnings.filterwarnings("ignore")

# Road types that are filtered out
FILTERED_ROAD_TYPES = ["ROAD_EDGE_SIDEWALK", "DRIVEWAY"]

# Mapping from scenario types to Waymax types
TYPE_MAPPING = {
    "LANE_FREEWAY": 1,
    "LANE_CENTER_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_SURFACE_UNSTRUCTURE": 2,
    "LANE_BIKE_LANE": 3,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    "ROAD_EDGE_BOUNDARY": 15,
    "ROAD_EDGE_MEDIAN": 16,
    "STOP_SIGN": 17,
    "CROSSWALK": 18,
    "SPEED_BUMP": 19,
}

# Colors for debug visualization
DEBUG_COLORS = {
    0: "cyan",
    1: "pink",
    2: "whitesmoke",
    3: "navajowhite",
    6: "silver",
    7: "silver",
    8: "silver",
    9: "khaki",
    10: "khaki",
    11: "khaki",
    12: "khaki",
    13: "khaki",
    15: "black",
    16: "black",
    17: "crimson",
    18: "lightskyblue",
    19: "salmon",
}


def get_scenario_map_points(scenario: dict[str, Any], debug: bool = False) -> tuple[RoadGraphSamples, int, bool]:
    """
    Extract map points from a scenario and convert them to roadgraph samples.

    Args:
        scenario: The scenario data containing map features
        debug: Whether to print debug information and visualize points

    Returns:
        A tuple containing:
        - RoadGraphSamples object with the extracted roadgraph data
        - Number of points added
        - Boolean indicating if the extraction was cropped due to size limits
    """
    roadgraph_samples = RoadGraphSamples()

    # Sort map features in reverse order for consistent processing
    map_features_keys = {key: value for key, value in reversed(sorted(scenario.map_features.items()))}

    num_points = 0
    count_types = {}
    mean_distances_types = {}
    std_distances_types = {}
    cropped = False

    for mp_block in map_features_keys:
        feature = scenario.map_features[mp_block]
        feature_type = feature["type"]

        # Track counts for each feature type
        count_types[feature_type] = count_types.get(feature_type, 0) + 1

        if feature_type in FILTERED_ROAD_TYPES:
            continue

        # Find the key containing the points data
        key = next((k for k in ["polyline", "lane", "polygon"] if k in feature), None)
        if not key:
            continue

        points_to_add = np.array(feature["position"]) if feature_type == "STOP_SIGN" else np.array(feature[key])

        points_type_to_add = _scenariomax_type_to_waymax_type(feature_type)
        points_to_add = _prepare_points(points_to_add, feature_type)

        # Check if adding these points would exceed the limit
        num_points_to_add = points_to_add.shape[0]
        if num_points + num_points_to_add >= DEFAULT_NUM_ROADMAPS:
            cropped = True
            break

        dir_points_to_add = _compute_dir_points(points_to_add)
        speed_limit = feature.get("speed_limit_mph", -1)

        if debug:
            _plot_debug_info(points_to_add, dir_points_to_add, speed_limit, points_type_to_add)

        # Track mean distances for each feature type
        dist_inter_points = np.linalg.norm(points_to_add[1:] - points_to_add[:-1], axis=1)
        mean_distance = np.mean(dist_inter_points)
        std_distance = np.std(dist_inter_points)

        if feature_type not in mean_distances_types:
            mean_distances_types[feature_type] = mean_distance
            std_distances_types[feature_type] = std_distance
        else:
            mean_distances_types[feature_type] = (mean_distances_types[feature_type] + mean_distance) / 2
            std_distances_types[feature_type] = (std_distances_types[feature_type] + std_distance) / 2

        block_id = _extract_block_id(mp_block)

        # Update roadgraph samples
        roadgraph_samples.xyz[num_points : num_points + num_points_to_add] = points_to_add
        roadgraph_samples.type[num_points : num_points + num_points_to_add] = points_type_to_add
        roadgraph_samples.id[num_points : num_points + num_points_to_add] = block_id
        roadgraph_samples.dir[num_points : num_points + num_points_to_add] = dir_points_to_add
        roadgraph_samples.speed_limit[num_points : num_points + num_points_to_add] = speed_limit
        num_points += num_points_to_add

    if debug:
        for key, value in mean_distances_types.items():
            print(
                f"- {key} - Num types: {count_types[key]}, "
                f"Mean distance: {round(value, 2)} - std: {round(std_distances_types[key], 2)}",
            )

    roadgraph_samples.valid[:num_points] = 1

    if _detect_overpass(roadgraph_samples.xyz[:num_points], roadgraph_samples.type[:num_points]):
        raise OverpassException()

    return roadgraph_samples, num_points, cropped


def _extract_block_id(block_key: str) -> int:
    """
    Extract a numeric ID from a block key string.

    Args:
        block_key: String identifier for a map block

    Returns:
        Integer ID extracted from the block key

    Raises:
        ValueError: If no numeric ID can be extracted from the block key
    """
    try:
        return int(block_key)
    except ValueError:
        try:
            return int(block_key.split("_")[0])
        except ValueError:
            try:
                return int(block_key.split("_")[-1])
            except ValueError:
                # If all attempts fail, use a hash-based approach
                return abs(hash(block_key)) % (10**8)


def _scenariomax_type_to_waymax_type(road_object_type: str) -> int:
    """
    Convert ScenarioMax road object type to Waymax type ID.

    Args:
        road_object_type: The road object type string

    Returns:
        The corresponding Waymax type ID (0 if not found)
    """
    return TYPE_MAPPING.get(road_object_type, 0)


def _compute_dir_points(points: np.ndarray) -> np.ndarray:
    """
    Computes the direction vectors between consecutive points.

    Args:
        points: A NumPy array of points, where each point is a 2D or 3D coordinate

    Returns:
        A NumPy array of direction vectors
    """
    points = np.array(points)

    if len(points) < 2:
        return np.zeros((1, points.shape[1]))

    # Compute direction vectors
    vectors = points[1:] - points[:-1]
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    safe_magnitudes = np.where(magnitudes > 1e-6, magnitudes, 1e-6)
    dir_points = vectors / safe_magnitudes

    # Append the last direction vector to maintain the same length
    dir_points = np.vstack([dir_points, dir_points[-1]])

    return dir_points


def _detect_overpass(xyz: np.ndarray, type: np.ndarray) -> bool:
    """
    Detect if the points form an overpass using KD-tree for efficient neighbor search.

    Args:
        xyz: A NumPy array of points, where each point is a 3D coordinate (x, y, z)
        type: A NumPy array of road types

    Returns:
        Boolean indicating if the points form an overpass
    """
    # Get road edge points
    road_edge_points = xyz[type == 15]

    # If not enough points, return False
    if len(road_edge_points) < 2:
        return False

    # Build KD-tree on XY coordinates for efficient nearest neighbor search
    tree = KDTree(road_edge_points[:, :2])

    # For each point, find all neighbors within 0.8 distance in XY plane
    for i, point in enumerate(road_edge_points):
        # Get indices of neighbors (excluding the point itself)
        neighbors = tree.query_ball_point(point[:2], 0.8)
        neighbors = [idx for idx in neighbors if idx > i]  # Only check each pair once

        if not neighbors:
            continue

        # Check Z differences
        z_differences = np.abs(point[2] - road_edge_points[neighbors, 2])
        if np.any(z_differences > 4.0):
            return True

    return False


def _add_interpolated_roadgraph_samples(points: np.ndarray, target_distance: float = DIST_INTERPOLATION) -> np.ndarray:
    """
    Interpolate points along a polyline with handling for unevenly spaced input points.

    Args:
        points: Array of source points
        target_distance: The desired distance between interpolated points

    Returns:
        Interpolated points with consistent spacing
    """
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])

    if len(points) <= 1:
        return points.copy()

    # Calculate segment distances and cumulative distances along the polyline
    segment_vectors = points[1:] - points[:-1]
    segment_distances = np.linalg.norm(segment_vectors, axis=1)
    cumulative_distances = np.zeros(len(points))
    cumulative_distances[1:] = np.cumsum(segment_distances)

    # Total length of the polyline
    total_length = cumulative_distances[-1]

    if total_length <= 0.0:
        return points.copy()

    # Add the first point
    result_points = [points[0]]

    # Interpolate at regular intervals
    current_distance = target_distance
    while current_distance < total_length:
        # Find the segment containing the current distance
        segment_idx = np.searchsorted(cumulative_distances, current_distance) - 1

        # Calculate interpolation factor within this segment
        segment_start_dist = cumulative_distances[segment_idx]
        segment_length = segment_distances[segment_idx]
        alpha = (current_distance - segment_start_dist) / segment_length

        # Choose interpolation method based on position
        if segment_idx < 1 or segment_idx > len(points) - 3:
            # Linear interpolation for boundary segments
            interpolated_point = _interpolate_point(points[segment_idx], points[segment_idx + 1], alpha)
        else:
            # Cubic spline for interior segments
            interpolated_point = _interpolate_cubic_spline(
                points[segment_idx - 1],
                points[segment_idx],
                points[segment_idx + 1],
                points[segment_idx + 2],
                alpha,
            )

        result_points.append(interpolated_point)
        current_distance += target_distance

    # Add the last point if it's not too close to the previous point
    last_point = points[-1]
    if result_points:
        threshold = 0.01
        if np.sum((last_point - result_points[-1]) ** 2) > threshold * threshold:
            result_points.append(last_point)
    else:
        result_points.append(last_point)

    return np.array(result_points)


def _interpolate_point(p1: np.ndarray, p2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolation between two points.

    Args:
        p1: First point
        p2: Second point
        alpha: Interpolation factor (0-1)

    Returns:
        Interpolated point
    """
    return p1 + alpha * (p2 - p1)


def _interpolate_cubic_spline(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Cubic spline interpolation among four points.

    Args:
        p0: First control point
        p1: Second control point (start point)
        p2: Third control point (end point)
        p3: Fourth control point
        alpha: Interpolation factor (0-1)

    Returns:
        Interpolated point
    """
    # Catmull-Rom spline coefficients
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha

    coef0 = -0.5 * alpha3 + alpha2 - 0.5 * alpha
    coef1 = 1.5 * alpha3 - 2.5 * alpha2 + 1.0
    coef2 = -1.5 * alpha3 + 2 * alpha2 + 0.5 * alpha
    coef3 = 0.5 * alpha3 - 0.5 * alpha2

    return coef0 * p0 + coef1 * p1 + coef2 * p2 + coef3 * p3


def _add_polygon_samples(points: np.ndarray, target_distance: float = DIST_INTERPOLATION) -> np.ndarray:
    """
    Process polygon points with proper segment handling and ensure the polygon is closed.

    Args:
        points: Array of polygon vertices
        type: The type identifier for the points
        target_distance: The desired distance between consecutive points

    Returns:
        Properly processed and closed polygon points
    """
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])

    if len(points) <= 1:
        return points.copy()

    # Collect all processed points
    result_points = []
    num_points = len(points)

    # Add the first N-1 segments of the polygon
    for i in range(num_points - 1):
        left_point = points[i]
        right_point = points[i + 1]

        segment_points = _add_sampled_polygon_segment(left_point, right_point, target_distance)
        result_points.extend(segment_points)

    # Add the last polygon segment from the last point to the first point
    last_segment = _add_sampled_polygon_segment(points[-1], points[0], target_distance)
    result_points.extend(last_segment)

    # Add the first point to complete the polygon
    result_points.append(points[0])

    return np.array(result_points)


def _add_sampled_polygon_segment(
    left_point: np.ndarray,
    right_point: np.ndarray,
    target_distance: float,
) -> list[np.ndarray]:
    """
    Sample points along a line segment with the desired spacing.

    Args:
        left_point: Starting point of the segment
        right_point: Ending point of the segment
        target_distance: The desired distance between consecutive points

    Returns:
        List of sampled points along the segment (excluding the endpoint)
    """
    segment_vector = right_point - left_point
    segment_length = np.linalg.norm(segment_vector)

    # If the segment is very short, just return the left point
    if segment_length < 1e-6:
        return [left_point]

    # Calculate number of sampled points needed
    num_samples = max(1, int(segment_length / target_distance))

    # Create evenly spaced points along the segment
    sampled_points = []
    for i in range(num_samples):
        # Using actual distance-based interpolation
        distance = i * target_distance
        # Ensure we don't exceed segment length
        if distance > segment_length:
            break
        alpha = distance / segment_length
        point = left_point + alpha * segment_vector
        sampled_points.append(point)

    return sampled_points


def _prepare_points(points_to_add: np.ndarray, type: str) -> np.ndarray:
    """
    Prepare points by ensuring correct shape and dimensions, and applying
    appropriate interpolation based on the point type.

    Args:
        points_to_add: Array of points
        type: The type identifier for the points

    Returns:
        Properly formatted and interpolated points
    """
    if len(points_to_add.shape) == 1:
        points_to_add = np.expand_dims(points_to_add, axis=0)
    if points_to_add.shape[1] == 2:
        points_to_add = np.hstack([points_to_add, np.zeros((points_to_add.shape[0], 1))])

    if type in ["CROSSWALK", "SPEED_BUMP"]:
        return _add_polygon_samples(points_to_add)
    else:
        return _add_interpolated_roadgraph_samples(points_to_add)


def _plot_debug_info(
    points_to_add: np.ndarray,
    dir_points_to_add: np.ndarray,
    speed_limit: int,
    points_type_to_add: int,
) -> None:
    """
    Plot debug visualization of the points and directions.

    Args:
        points_to_add: Array of points to plot
        dir_points_to_add: Array of direction vectors
        speed_limit: Speed limit for the points
        points_type_to_add: Type of the points for coloring
    """
    ax = plt.gca()
    ax.scatter(points_to_add[:, 0], points_to_add[:, 1], s=2, c=DEBUG_COLORS[points_type_to_add])
    if speed_limit > 0:
        ax.text(
            points_to_add[0, 0],
            points_to_add[0, 1],
            f"Speed limit: {speed_limit} mph",
            fontsize=8,
            color="black",
        )
    # Uncomment to show direction arrows:
    # for i in range(len(dir_points_to_add)):
    #     ax.arrow(
    #         points_to_add[i, 0],
    #         points_to_add[i, 1],
    #         dir_points_to_add[i, 0],
    #         dir_points_to_add[i, 1],
    #         head_width=0.5,
    #         head_length=0.4,
    #         color=DEBUG_COLORS[points_type_to_add],
    #     )

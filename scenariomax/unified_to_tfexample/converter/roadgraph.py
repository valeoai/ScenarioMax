import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
        points_to_add = _prepare_points(points_to_add, points_type_to_add)

        # Check if adding these points would exceed the limit
        num_points_to_add = points_to_add.shape[0]
        if num_points + num_points_to_add >= DEFAULT_NUM_ROADMAPS:
            cropped = True
            break

        dir_points_to_add = _compute_dir_points(points_to_add)
        speed_limit = feature.get("speed_limit_mph", -1)

        if debug:
            _plot_debug_info(points_to_add, dir_points_to_add, points_type_to_add)

        # Track mean distances for each feature type
        mean_distance = np.mean(np.linalg.norm(points_to_add[1:] - points_to_add[:-1], axis=1))

        if feature_type not in mean_distances_types:
            mean_distances_types[feature_type] = mean_distance
        else:
            mean_distances_types[feature_type] = (mean_distances_types[feature_type] + mean_distance) / 2

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
            print(f"- {key} - Num types: {count_types[key]}, Mean distance: {round(value, 2)} ({value})")

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


def _prepare_points(points_to_add: np.ndarray, type: int) -> np.ndarray:
    """
    Prepare points by ensuring correct shape and dimensions.

    Args:
        points_to_add: Array of points
        type: The type identifier for the points

    Returns:
        Properly formatted and calibrated points
    """
    if len(points_to_add.shape) == 1:
        points_to_add = np.expand_dims(points_to_add, axis=0)
    if points_to_add.shape[1] == 2:
        points_to_add = np.hstack([points_to_add, np.zeros((points_to_add.shape[0], 1))])

    return _calibrate(points_to_add, type)


def _calibrate(points: np.ndarray, type: int, target_distance: float = DIST_INTERPOLATION) -> np.ndarray:
    """
    Calibrates a list of points to have a consistent distance between them.

    Args:
        points: A NumPy array of points, where each point is a 2D or 3D coordinate
        target_distance: The desired distance between consecutive points

    Returns:
        A new array of points with a consistent distance between them
    """
    points = np.array(points)
    calibrated_points = [points[0]]

    for i in range(1, len(points)):
        prev_point = calibrated_points[-1]
        curr_point = points[i]
        segment_vector = curr_point - prev_point
        distance = np.linalg.norm(segment_vector)

        while distance >= target_distance:
            unit_vector = segment_vector / distance
            new_point = prev_point + unit_vector * target_distance
            calibrated_points.append(new_point)
            prev_point = new_point
            segment_vector = curr_point - prev_point
            distance = np.linalg.norm(segment_vector)

    # Ensure the last point in the input is included
    if not np.array_equal(calibrated_points[-1], points[-1]):
        calibrated_points.append(points[-1])

    return np.array(calibrated_points)


def _detect_overpass(xyz: np.ndarray, type: np.ndarray) -> bool:
    """
    Detect if the points form an overpass.

    Args:
        points: A NumPy array of points, where each point is a 2D or 3D coordinate
        type: A NumPy array of road types

    Returns:
        Boolean indicating if the points form an overpass
    """
    # Get points that are closed in XY by less than 1m
    road_edge_points = xyz[type == 15]

    xy_distances = np.linalg.norm(
        road_edge_points[:, :2].reshape(-1, 1, 2) - road_edge_points[:, :2].reshape(1, -1, 2),
        axis=2,
    )

    # Find pairs of points (use lower triangle to avoid duplicates)
    mask = np.tril(xy_distances < 0.8, k=-1)
    paired_indices = np.argwhere(mask)

    # If no close pairs found, return False
    if len(paired_indices) == 0:
        return False

    # Calculate z-differences
    z_diffs = np.abs(road_edge_points[paired_indices[:, 0], 2] - road_edge_points[paired_indices[:, 1], 2])
    max_z_diff = np.max(z_diffs) if len(z_diffs) > 0 else 0.0

    return max_z_diff > 4.0


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
    dir_points = vectors / magnitudes

    # Append the last direction vector to maintain the same length
    dir_points = np.vstack([dir_points, dir_points[-1]])

    return dir_points


def _plot_debug_info(points_to_add: np.ndarray, dir_points_to_add: np.ndarray, points_type_to_add: int) -> None:
    """
    Plot debug visualization of the points and directions.

    Args:
        points_to_add: Array of points to plot
        dir_points_to_add: Array of direction vectors
        points_type_to_add: Type of the points for coloring
    """
    ax = plt.gca()
    ax.scatter(points_to_add[:, 0], points_to_add[:, 1], s=2, c=DEBUG_COLORS[points_type_to_add])
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

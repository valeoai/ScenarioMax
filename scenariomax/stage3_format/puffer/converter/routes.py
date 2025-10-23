"""
Compute agent routes for Puffer format.

Routes are lists of lane center IDs that:
1. Cover the ground truth trajectory
2. Extend beyond the trajectory using lane connectivity
"""

import numpy as np

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


# Constants for route computation
LANE_WIDTH_THRESHOLD = 3.5  # Meters - how close agent must be to lane
ALIGNMENT_THRESHOLD = 0.3  # Cosine similarity threshold for direction alignment
MAX_ROUTE_DEPTH = 10  # Maximum depth for route extension


def compute_agent_route(
    agent_trajectory: np.ndarray,
    agent_heading: np.ndarray,
    agent_valid: np.ndarray,
    static_map_elements: dict,
) -> list[str]:
    """
    Compute route (list of lane IDs) for an agent based on ground truth trajectory.

    The route covers the ground truth trajectory and extends beyond it using lane connectivity.

    Args:
        agent_trajectory: Agent position trajectory (N, 2) or (N, 3) array
        agent_heading: Agent heading at each timestep (N,) array
        agent_valid: Validity mask for trajectory (N,) array
        static_map_elements: Dict of static map elements (lanes, boundaries, etc.)

    Returns:
        List of lane center IDs (as strings) forming the route
    """
    if len(agent_trajectory) == 0 or not np.any(agent_valid):
        return []

    # Extract valid trajectory points
    valid_trajectory = agent_trajectory[agent_valid]
    valid_heading = agent_heading[agent_valid]

    if len(valid_trajectory) == 0:
        return []

    # Get lane centers from map (vectorized format)
    lane_data = _extract_lane_centers(static_map_elements)
    lane_ids, lane_polylines, lane_metadata = lane_data

    if len(lane_ids) == 0:
        logger.debug("No lane centers found in map")
        return []

    # Step 1: Find lanes that cover the ground truth trajectory (vectorized)
    covered_lanes = _find_lanes_covering_trajectory(
        valid_trajectory,
        valid_heading,
        lane_data,
    )

    if not covered_lanes:
        logger.debug("No lanes found covering trajectory")
        return []

    # Step 2: Extend route beyond trajectory using lane connectivity
    route = _extend_route_with_connectivity(
        covered_lanes,
        static_map_elements,
        valid_trajectory[-1],  # Last position
        valid_heading[-1],  # Last heading
    )

    return route


def _extract_lane_centers(static_map_elements: dict) -> tuple[list, np.ndarray, dict]:
    """
    Extract lane center information as numpy arrays for vectorized operations.

    Args:
        static_map_elements: Dict of static map elements

    Returns:
        Tuple of:
        - lane_ids: List of lane IDs (strings)
        - lane_polylines: Array of lane polylines, padded to max length (N_lanes, max_points, 2)
        - lane_metadata: Dict mapping lane_id to connectivity info
    """
    lane_ids = []
    lane_polylines_list = []
    lane_metadata = {}
    max_points = 0

    # First pass: collect lanes and find max polyline length
    for element_id, element_data in static_map_elements.items():
        element_type = element_data.get("type", "")

        # Only process lane centers
        if "LANE" in element_type:
            polyline = element_data.get("polyline")

            if polyline is not None and len(polyline) > 0:
                # Convert to 2D if needed
                polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline

                lane_ids.append(element_id)
                lane_polylines_list.append(polyline_2d)
                max_points = max(max_points, len(polyline_2d))

                lane_metadata[element_id] = {
                    "entry_lanes": element_data.get("entry_lanes", []),
                    "exit_lanes": element_data.get("exit_lanes", []),
                }

    if not lane_ids:
        return [], np.array([]), {}

    # Second pass: create padded array
    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float32)

    for i, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[i, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata


def _find_lanes_covering_trajectory(
    trajectory: np.ndarray,
    heading: np.ndarray,
    lane_data: tuple,
) -> list[str]:
    """
    Find ordered list of lane IDs that cover the ground truth trajectory (vectorized).

    Args:
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata) from _extract_lane_centers

    Returns:
        Ordered list of lane IDs covering the trajectory
    """
    if len(trajectory) == 0:
        return []

    lane_ids, lane_polylines, lane_metadata = lane_data

    if len(lane_ids) == 0:
        return []

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # For each trajectory point, find the closest matching lane
    trajectory_lanes = []
    last_lane_id = None

    # Vectorized computation for each trajectory point
    for pos, hdg in zip(trajectory_2d, heading):
        # Vectorized distance calculation for all lanes
        # lane_polylines: (N_lanes, max_points, 2)
        # pos: (2,)
        min_distances, closest_indices = _vectorized_point_to_polylines_distance(pos, lane_polylines)

        # Filter lanes within threshold
        valid_lanes_mask = min_distances < LANE_WIDTH_THRESHOLD

        if not np.any(valid_lanes_mask):
            continue

        # Calculate lane directions at closest points (vectorized)
        lane_directions = _get_lane_directions_at_indices(lane_polylines, closest_indices)

        # Calculate agent direction
        agent_dir = np.array([np.cos(hdg), np.sin(hdg)])

        # Vectorized alignment calculation: dot product for all lanes
        alignments = np.sum(lane_directions * agent_dir, axis=1)  # (N_lanes,)

        # Filter by alignment threshold
        valid_lanes_mask &= alignments > ALIGNMENT_THRESHOLD

        if not np.any(valid_lanes_mask):
            continue

        # Vectorized score calculation
        distance_scores = 1.0 / (1.0 + min_distances)
        scores = 0.7 * alignments + 0.3 * distance_scores

        # Mask out invalid lanes
        scores = np.where(valid_lanes_mask, scores, -np.inf)

        # Find best lane
        best_idx = np.argmax(scores)

        if scores[best_idx] > -np.inf:
            best_lane_id = lane_ids[best_idx]

            # Add lane if different from last
            if best_lane_id != last_lane_id:
                trajectory_lanes.append(best_lane_id)
                last_lane_id = best_lane_id

    return trajectory_lanes


def _vectorized_point_to_polylines_distance(point: np.ndarray, polylines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized calculation of distance from a point to multiple polylines.

    Args:
        point: 2D point (2,)
        polylines: Array of polylines (N_lanes, max_points, 2)

    Returns:
        Tuple of:
        - min_distances: Array of minimum distances for each lane (N_lanes,)
        - closest_indices: Array of closest segment indices for each lane (N_lanes,)
    """
    n_lanes, max_points, _ = polylines.shape

    # Compute distances for all segments of all lanes
    # polylines[:, :-1]: (N_lanes, max_points-1, 2) - segment starts
    # polylines[:, 1:]: (N_lanes, max_points-1, 2) - segment ends
    seg_starts = polylines[:, :-1, :]  # (N_lanes, max_points-1, 2)
    seg_ends = polylines[:, 1:, :]  # (N_lanes, max_points-1, 2)

    # Vectorized segment distance calculation
    # For each lane, for each segment, calculate distance
    seg_vecs = seg_ends - seg_starts  # (N_lanes, max_points-1, 2)
    seg_lens_sq = np.sum(seg_vecs**2, axis=2)  # (N_lanes, max_points-1)

    # Avoid division by zero
    valid_segs = seg_lens_sq > 1e-10

    # Project point onto each segment
    point_vecs = point - seg_starts  # (N_lanes, max_points-1, 2)
    t = np.sum(point_vecs * seg_vecs, axis=2) / (seg_lens_sq + 1e-10)  # (N_lanes, max_points-1)
    t = np.clip(t, 0, 1)  # Clamp to [0, 1]

    # Calculate closest point on each segment
    closest_points = seg_starts + t[:, :, np.newaxis] * seg_vecs  # (N_lanes, max_points-1, 2)

    # Calculate distances
    distances = np.linalg.norm(point - closest_points, axis=2)  # (N_lanes, max_points-1)

    # Set invalid segments to inf
    distances = np.where(valid_segs, distances, np.inf)

    # Find minimum distance and index for each lane
    min_distances = np.min(distances, axis=1)  # (N_lanes,)
    closest_indices = np.argmin(distances, axis=1)  # (N_lanes,)

    return min_distances, closest_indices


def _get_lane_directions_at_indices(polylines: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Get lane directions at specified segment indices (vectorized).

    Args:
        polylines: Array of polylines (N_lanes, max_points, 2)
        indices: Array of segment indices for each lane (N_lanes,)

    Returns:
        Array of normalized direction vectors (N_lanes, 2)
    """
    n_lanes, max_points, _ = polylines.shape

    # Create indices for gathering
    lane_indices = np.arange(n_lanes)

    # Get segment start and end points
    seg_starts = polylines[lane_indices, indices, :]  # (N_lanes, 2)

    # Handle edge cases: if index is at the end, use previous segment
    next_indices = np.minimum(indices + 1, max_points - 1)
    seg_ends = polylines[lane_indices, next_indices, :]  # (N_lanes, 2)

    # Calculate direction vectors
    directions = seg_ends - seg_starts  # (N_lanes, 2)

    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)  # (N_lanes, 1)
    directions = directions / (norms + 1e-6)  # (N_lanes, 2)

    return directions


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> tuple[float, int]:
    """
    Calculate minimum distance from a point to a polyline (considering segments).

    Args:
        point: 2D point (2,)
        polyline: 2D polyline (N, 2)

    Returns:
        Tuple of (min_distance, closest_segment_index)
    """
    if len(polyline) < 2:
        # Degenerate case: single point
        dist = np.linalg.norm(polyline[0] - point)
        return dist, 0

    min_dist = np.inf
    closest_seg_idx = 0

    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]

        # Calculate distance to line segment
        seg_dist = _point_to_segment_distance(point, p1, p2)

        if seg_dist < min_dist:
            min_dist = seg_dist
            closest_seg_idx = i

    return min_dist, closest_seg_idx


def _point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """
    Calculate distance from a point to a line segment.

    Args:
        point: 2D point (2,)
        seg_start: Segment start point (2,)
        seg_end: Segment end point (2,)

    Returns:
        Minimum distance to segment
    """
    # Vector from seg_start to seg_end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-10:
        # Degenerate segment (single point)
        return np.linalg.norm(point - seg_start)

    # Project point onto line
    # t = [(point - seg_start) · seg_vec] / |seg_vec|^2
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq

    # Clamp t to [0, 1] to stay on segment
    t = np.clip(t, 0, 1)

    # Find closest point on segment
    closest = seg_start + t * seg_vec

    # Return distance to closest point
    return np.linalg.norm(point - closest)


def _extend_route_with_connectivity(
    covered_lanes: list[str],
    static_map_elements: dict,
    last_position: np.ndarray,
    last_heading: float,
) -> list[str]:
    """
    Extend route beyond covered lanes using lane connectivity.

    Args:
        covered_lanes: List of lane IDs covering ground truth
        static_map_elements: Dict of static map elements
        last_position: Last position of agent (2,) or (3,)
        last_heading: Last heading of agent

    Returns:
        Extended route as list of lane IDs
    """
    if not covered_lanes:
        return []

    route = covered_lanes.copy()
    current_lane_id = covered_lanes[-1]

    # Extract agent direction vector
    agent_dir = np.array([np.cos(last_heading), np.sin(last_heading)])

    # Recursively extend using exit lanes
    for _ in range(MAX_ROUTE_DEPTH):
        if current_lane_id not in static_map_elements:
            break

        current_lane = static_map_elements[current_lane_id]
        exit_lanes = current_lane.get("exit_lanes", [])

        if not exit_lanes:
            break

        # Choose best exit lane based on alignment
        best_exit_lane = None
        best_alignment = -np.inf

        for exit_lane_id in exit_lanes:
            exit_lane_id_str = str(exit_lane_id)

            # Avoid loops
            if exit_lane_id_str in route:
                continue

            if exit_lane_id_str not in static_map_elements:
                continue

            exit_lane = static_map_elements[exit_lane_id_str]
            exit_polyline = exit_lane.get("polyline")

            if exit_polyline is None or len(exit_polyline) < 2:
                continue

            # Extract 2D polyline
            exit_polyline_2d = exit_polyline[:, :2] if exit_polyline.shape[1] == 3 else exit_polyline

            # Calculate direction of exit lane at start
            lane_dir = exit_polyline_2d[1] - exit_polyline_2d[0]
            lane_dir = lane_dir / (np.linalg.norm(lane_dir) + 1e-6)

            # Calculate alignment with agent direction
            alignment = np.dot(lane_dir, agent_dir)

            if alignment > best_alignment:
                best_alignment = alignment
                best_exit_lane = exit_lane_id_str

        # Add best exit lane if found and aligned
        if best_exit_lane is not None and best_alignment > ALIGNMENT_THRESHOLD:
            route.append(best_exit_lane)
            current_lane_id = best_exit_lane
        else:
            break

    return route

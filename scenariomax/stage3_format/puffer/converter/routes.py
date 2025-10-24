"""
Compute agent routes for Puffer format.

Routes are lists of lane center IDs that:
1. Cover the ground truth trajectory
2. Extend beyond the trajectory using lane connectivity
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


# Constants for route computation
LANE_WIDTH_THRESHOLD = 3.5  # Meters - how close agent must be to lane
ALIGNMENT_THRESHOLD = 0.3  # Cosine similarity threshold for direction alignment
MAX_ROUTE_DEPTH = 10  # Maximum depth for route extension
MIN_VALID_TRAJECTORY_POINTS = 10  # Minimum valid points to compute route


def compute_agent_route(
    agent_trajectory: np.ndarray,
    agent_heading: np.ndarray,
    agent_valid: np.ndarray,
    static_map_elements: dict,
    lane_data: tuple = None,
    agent_id: int | str = None,
) -> list[list[int]]:
    """
    Compute routes (lists of lane IDs) for an agent based on ground truth trajectory.

    The routes cover the ground truth trajectory and extend beyond it using lane connectivity.
    Multiple route paths are generated to explore different exit lane possibilities.

    Args:
        agent_trajectory: Agent position trajectory (N, 2) or (N, 3) array
        agent_heading: Agent heading at each timestep (N,) array
        agent_valid: Validity mask for trajectory (N,) array
        static_map_elements: Dict of static map elements (lanes, boundaries, etc.)
        lane_data: Optional precomputed lane data (lane_ids, lane_polylines, lane_metadata).
                   If None, will be computed from static_map_elements.
        agent_id: Optional agent identifier for debugging logs

    Returns:
        List of routes, where each route is a list of lane center IDs
    """
    if len(agent_trajectory) == 0 or not np.any(agent_valid):
        return []

    # Extract valid trajectory points
    valid_trajectory = agent_trajectory[agent_valid]
    valid_heading = agent_heading[agent_valid]

    if len(valid_trajectory) == 0:
        return []

    # Format agent identifier for logging
    agent_str = f"Agent {agent_id}" if agent_id is not None else "Agent"

    # Check if agent has enough valid trajectory points
    if len(valid_trajectory) < MIN_VALID_TRAJECTORY_POINTS:
        logger.debug(
            f"{agent_str}: Trajectory too short ({len(valid_trajectory)} < {MIN_VALID_TRAJECTORY_POINTS} points)",
        )
        return []

    lane_ids, lane_polylines, _ = lane_data

    if len(lane_ids) == 0:
        logger.debug(f"{agent_str}: No lane centers found in map")
        return []

    # Check if agent is mostly on lanes (not parked/off-map)
    if not _is_agent_on_lanes(valid_trajectory, lane_polylines):
        logger.debug(f"{agent_str}: Off-map or parked (not close enough to lanes)")
        return []

    # Step 1: Find current lane (root)
    root_lane = _find_root_lane(valid_trajectory, valid_heading, lane_data)

    if not root_lane:
        logger.debug(f"{agent_str}: No current lane found")
        return []

    # Step 2: Build route paths by exploring exit lanes and matching with GT trajectory
    routes = _build_route_paths_from_root(root_lane, valid_trajectory, valid_heading, static_map_elements, lane_data)

    return routes


def extract_lane_centers(static_map_elements: dict) -> tuple[list, np.ndarray, dict]:
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
        if element_type == types.LANE_SURFACE_STREET or element_type == types.LANE_FREEWAY:
            polyline = element_data.get("polyline")

            if len(polyline) > 0:
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


def _is_agent_on_lanes(trajectory: np.ndarray, lane_polylines: np.ndarray) -> bool:
    """
    Check if an agent's trajectory is mostly on lanes (not parked or off-map).

    Args:
        trajectory: Agent trajectory points (M, 2/3)
        lane_polylines: Array of lane polylines (N_lanes, max_points, 2)

    Returns:
        True if agent is mostly on lanes, False if parked or off-map
    """
    if len(trajectory) == 0 or len(lane_polylines) == 0:
        return False

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Check distance of trajectory
    if np.linalg.norm(trajectory_2d, axis=1).max() < 1e-3:
        trajectory_2d = trajectory_2d[:1]
    else:
        trajectory_2d = [trajectory_2d[0], trajectory_2d[-1]]

    # Count how many trajectory points are within LANE_WIDTH_THRESHOLD of any lane
    points_near_lanes = 0

    for pos in trajectory_2d:
        # Vectorized distance calculation for all lanes
        min_distances, _ = _point_to_polylines_distance(pos, lane_polylines)

        # Check if any lane is within threshold
        if np.any(min_distances < LANE_WIDTH_THRESHOLD):
            points_near_lanes += 1

    # Agent is on lanes if at least threshold_ratio of points are near lanes
    return points_near_lanes >= 1


def _find_root_lane(trajectory: np.ndarray, heading: np.ndarray, lane_data: tuple) -> str | None:
    """
    Find the current lane where the agent is located using the first 3 trajectory points.

    Args:
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata) from _extract_lane_centers

    Returns:
        The current lane ID (root lane) or None if no lane is found
    """
    lane_ids, lane_polylines, _ = lane_data

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Use only the first 3 points to determine current lane
    num_points_for_current_lane = min(10, len(trajectory_2d))
    first_points = trajectory_2d[:num_points_for_current_lane]
    first_headings = heading[:num_points_for_current_lane]

    # Compute lane matches for the first 3 points
    lane_vote_scores = {}  # lane_idx -> total score

    for pos, hdg in zip(first_points, first_headings):
        # Distance calculation for all lanes
        min_distances, closest_indices = _point_to_polylines_distance(pos, lane_polylines)

        # Filter lanes within threshold
        valid_lanes_mask = min_distances < LANE_WIDTH_THRESHOLD

        if not np.any(valid_lanes_mask):
            continue

        # Calculate lane directions at closest points
        lane_directions = _get_lane_directions_at_indices(lane_polylines, closest_indices)

        # Calculate agent direction
        agent_dir = np.array([np.cos(hdg), np.sin(hdg)])

        # Vectorized alignment calculation
        alignments = np.sum(lane_directions * agent_dir, axis=1)  # (N_lanes,)

        # Filter by alignment threshold
        valid_lanes_mask &= alignments > ALIGNMENT_THRESHOLD

        if not np.any(valid_lanes_mask):
            continue

        # Calculate scores
        distance_scores = 1.0 / (1.0 + min_distances)
        scores = 0.7 * alignments + 0.3 * distance_scores

        # Accumulate scores for each valid lane
        for lane_idx in np.where(valid_lanes_mask)[0]:
            if lane_idx not in lane_vote_scores:
                lane_vote_scores[lane_idx] = 0.0
            lane_vote_scores[lane_idx] += scores[lane_idx]

    if not lane_vote_scores:
        return None

    # Return the lane with the highest total score
    best_lane_idx = max(lane_vote_scores, key=lane_vote_scores.get)
    return lane_ids[best_lane_idx]


def _build_route_paths_from_root(
    root_lane: str,
    trajectory: np.ndarray,
    heading: np.ndarray,
    static_map_elements: dict,
    lane_data: tuple,
) -> list[list[str]]:
    """
    Build multiple route paths starting from the root lane.

    For each exit lane choice, determines if it follows the GT trajectory.
    Creates new route paths when beyond the GT trajectory.
    Each route path contains the GT trajectory.

    Args:
        root_lane: The current lane ID where the agent starts
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        static_map_elements: Dict of static map elements
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata)

    Returns:
        List of route paths, each containing lane IDs covering the GT trajectory
    """
    lane_ids, lane_polylines, lane_metadata = lane_data

    # Build a mapping from lane_id to lane_idx for quick lookup
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Start with the root lane
    all_routes = []

    # Use DFS/BFS to explore all paths through exit lanes
    # State: (current_route, current_trajectory_idx, current_lane_id)
    queue = [(([root_lane], 0, root_lane))]

    while queue:
        current_route, traj_idx, current_lane_id = queue.pop(0)

        # Check if we've covered the entire trajectory
        if traj_idx >= len(trajectory_2d):
            # Extend beyond GT trajectory using exit lanes
            extended_route = _extend_beyond_trajectory(current_route, static_map_elements)
            all_routes.append(extended_route)
            continue

        # Get exit lanes from current lane
        if current_lane_id not in static_map_elements:
            # No exit lanes, save current route
            all_routes.append(current_route)
            continue

        current_lane_data = static_map_elements[current_lane_id]
        exit_lanes = current_lane_data.get("exit_lanes", [])

        if not exit_lanes:
            # No exit lanes, extend beyond trajectory
            extended_route = _extend_beyond_trajectory(current_route, static_map_elements)
            all_routes.append(extended_route)
            continue

        # Score each exit lane based on how well it matches the remaining GT trajectory
        exit_lane_scores = []
        for exit_lane_id in exit_lanes:
            if exit_lane_id not in lane_id_to_idx:
                continue

            score, next_traj_idx = _score_lane_against_trajectory(
                exit_lane_id, lane_id_to_idx, lane_polylines, trajectory_2d, heading, traj_idx
            )

            exit_lane_scores.append((exit_lane_id, score, next_traj_idx))

        if not exit_lane_scores:
            # No valid exit lanes, save current route
            all_routes.append(current_route)
            continue

        # Sort by score (descending) and select the best matching exit lanes
        exit_lane_scores.sort(key=lambda x: x[1], reverse=True)

        # Add paths for all exit lanes with positive scores (following GT)
        # This creates multiple route possibilities
        for exit_lane_id, score, next_traj_idx in exit_lane_scores:
            if score > 0:  # Only follow lanes that match GT
                new_route = current_route + [exit_lane_id]
                queue.append((new_route, next_traj_idx, exit_lane_id))
            else:
                # Beyond GT trajectory, create new route path
                new_route = current_route + [exit_lane_id]
                extended_route = _extend_beyond_trajectory(new_route, static_map_elements)
                all_routes.append(extended_route)

    # If no routes were found, return just the root lane
    if not all_routes:
        all_routes = [[root_lane]]

    return all_routes


def _score_lane_against_trajectory(
    lane_id: str,
    lane_id_to_idx: dict,
    lane_polylines: np.ndarray,
    trajectory: np.ndarray,
    heading: np.ndarray,
    start_traj_idx: int,
) -> tuple[float, int]:
    """
    Score how well a lane matches the remaining ground truth trajectory.

    Args:
        lane_id: Lane ID to score
        lane_id_to_idx: Mapping from lane_id to lane_idx
        lane_polylines: Array of lane polylines (N_lanes, max_points, 2)
        trajectory: Trajectory points (M, 2)
        heading: Headings (M,)
        start_traj_idx: Index in trajectory to start matching from

    Returns:
        Tuple of (score, next_trajectory_idx) where score > 0 means lane follows GT
    """
    if lane_id not in lane_id_to_idx:
        return 0.0, start_traj_idx

    lane_idx = lane_id_to_idx[lane_id]
    lane_polyline = lane_polylines[lane_idx]

    # Check trajectory points that fall within this lane
    total_score = 0.0
    points_matched = 0
    next_traj_idx = start_traj_idx

    for traj_idx in range(start_traj_idx, len(trajectory)):
        pos = trajectory[traj_idx]
        hdg = heading[traj_idx]

        # Calculate distance to lane
        # Get single lane polyline
        lane_poly_single = lane_polyline[np.newaxis, :, :]  # (1, max_points, 2)
        min_distances, closest_indices = _point_to_polylines_distance(pos, lane_poly_single)
        min_dist = min_distances[0]
        closest_idx = closest_indices[0]

        # Check if point is close to lane
        if min_dist > LANE_WIDTH_THRESHOLD:
            break  # Point is too far from lane

        # Check alignment
        lane_directions = _get_lane_directions_at_indices(lane_poly_single, np.array([closest_idx]))
        lane_dir = lane_directions[0]
        agent_dir = np.array([np.cos(hdg), np.sin(hdg)])
        alignment = np.dot(lane_dir, agent_dir)

        if alignment < ALIGNMENT_THRESHOLD:
            break  # Direction mismatch

        # Calculate score for this point
        distance_score = 1.0 / (1.0 + min_dist)
        point_score = 0.7 * alignment + 0.3 * distance_score

        total_score += point_score
        points_matched += 1
        next_traj_idx = traj_idx + 1

    if points_matched == 0:
        return 0.0, start_traj_idx

    # Average score
    avg_score = total_score / points_matched

    return avg_score, next_traj_idx


def _extend_beyond_trajectory(route: list[str], static_map_elements: dict) -> list[str]:
    """
    Extend route beyond the GT trajectory using exit lanes.

    Simply follows the first exit lane at each step.

    Args:
        route: Current route
        static_map_elements: Dict of static map elements

    Returns:
        Extended route
    """
    extended_route = route.copy()
    current_lane_id = route[-1] if route else None

    if not current_lane_id:
        return extended_route

    # Extend using exit lanes
    for _ in range(MAX_ROUTE_DEPTH):
        if current_lane_id not in static_map_elements:
            break

        current_lane = static_map_elements[current_lane_id]
        exit_lanes = current_lane.get("exit_lanes", [])

        if not exit_lanes:
            break

        # Take first exit lane
        next_lane_id = exit_lanes[0]
        extended_route.append(next_lane_id)
        current_lane_id = next_lane_id

    return extended_route


def _point_to_polylines_distance(point: np.ndarray, polylines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

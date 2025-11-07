"""
Compute agent routes for Puffer format.

Routes are lists of lane center IDs that cover the ground truth trajectory
and explore different possible paths through the road network.

Algorithm Overview:
------------------
The route computation uses a 4-step graph-based approach:

1. Validation: Check if agent has sufficient trajectory points and is on lanes (not parked)
2. Root Lane Selection: Find the current lane using the first N trajectory points
3. Graph Building: Build full reachability graph from root lane (no pruning)
4. Metadata Computation: Score each edge against ground truth trajectory
5. Path Extraction: Extract top N paths using GT coverage metric

Multiple route paths are generated to represent different possible paths through
exit lanes (e.g., at highway interchanges or intersections). This graph-based
approach enables flexible path selection with pluggable metrics.

The algorithm uses vectorized operations for performance and scores lane matches
based on both spatial proximity (distance) and directional alignment (heading).
"""

from collections import deque

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


# Constants for route computation
# Physical thresholds
LANE_WIDTH_THRESHOLD = 4.0  # Meters - maximum distance from lane center to consider agent "on lane"
ALIGNMENT_THRESHOLD = 0.3  # Cosine similarity threshold for direction alignment
# Value of 0.3 corresponds to ~72.5° angle deviation (arccos(0.3) ≈ 72.5°)
# Allows moderate heading mismatch while filtering out wrong-way or perpendicular lanes
# Range: [-1, 1] where 1 = perfect alignment, 0 = perpendicular, -1 = opposite direction

# Algorithm parameters
MAX_GRAPH_DEPTH = 10  # Maximum depth when building reachability graph from root lane
ROOT_LANE_POINTS = 3  # Number of initial trajectory points used to determine current lane
MAX_PATH_LENGTH = 10  # Maximum number of lanes per route path
MAX_ROUTES = 10  # Maximum number of route paths to generate per agent

# Score calculation weights
# These weights determine the relative importance of direction vs. distance in lane matching
# The scoring formula is: score = ALIGNMENT_WEIGHT * alignment + DISTANCE_WEIGHT * distance_score
# where alignment ∈ [-1, 1] and distance_score = 1/(1+distance) ∈ (0, 1]
ALIGNMENT_WEIGHT = 0.7  # Weight for directional alignment (heading match)
DISTANCE_WEIGHT = 0.3  # Weight for spatial proximity
# Combined weights sum to 1.0 for interpretable normalized scoring


def compute_agent_route(
    agent_trajectory: np.ndarray,
    agent_heading: np.ndarray,
    agent_valid: np.ndarray,
    static_map_elements: dict,
    lane_data: tuple,
    agent_id: int | str = None,
    min_route_valid_points: int = 0,
) -> list[list[int]]:
    """
    Compute routes (lists of lane IDs) for an agent based on ground truth trajectory.

    Algorithm:
    1. Find root lane using first N trajectory points
    2. Build full reachability graph from root (no pruning)
    3. Compute GT matching metadata for each edge
    4. Extract top N paths using GT coverage metric

    Multiple route paths are generated to explore different exit lane possibilities.

    Args:
        agent_trajectory: Agent position trajectory (N, 2) or (N, 3) array
        agent_heading: Agent heading at each timestep (N,) array
        agent_valid: Validity mask for trajectory (N,) array
        static_map_elements: Dict of static map elements (lanes, boundaries, etc.)
        lane_data: Precomputed lane data (lane_ids, lane_polylines, lane_metadata).
                   Must be provided - use extract_lane_centers() to generate.
        agent_id: Optional agent identifier for debugging logs
        min_route_valid_points: Minimum valid trajectory points required for route computation (0 = no filtering)

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

    # Check if trajectory has sufficient valid points
    if min_route_valid_points > 0 and len(valid_trajectory) < min_route_valid_points:
        logger.debug(
            f"{agent_str}: Insufficient valid points ({len(valid_trajectory)} < {min_route_valid_points})",
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

    # Step 2: Build reachability graph from root lane (no pruning)
    graph = _build_graph(root_lane, static_map_elements)

    if not graph:
        logger.debug(f"{agent_str}: No reachable lanes from root {root_lane}")
        return [[root_lane]]

    # Step 3: Compute GT matching metadata for graph edges
    metadata = _compute_gt_metadata(graph, root_lane, valid_trajectory, valid_heading, lane_data)

    # Step 4: Extract top N paths using GT coverage metric
    routes = extract_top_n_paths(graph, root_lane, metadata, n=MAX_ROUTES)

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


def _build_graph(
    root_lane: int | str,
    static_map_elements: dict,
    max_depth: int = MAX_GRAPH_DEPTH,
) -> dict[int | str, list]:
    """
    Build complete reachability graph from root lane with no pruning.

    Explores all exit lanes up to max_depth using BFS with cycle detection.
    Returns adjacency list representation of the lane connectivity graph.

    Args:
        root_lane: Starting lane ID
        static_map_elements: Dict of static map elements
        max_depth: Maximum depth to explore from root

    Returns:
        Dict mapping lane_id to list of exit lane IDs: {lane_id: [exit_ids]}
    """
    graph = {}
    visited_lanes = {root_lane}
    queue = deque([(root_lane, 0)])

    while queue:
        lane_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        if lane_id not in static_map_elements:
            continue

        exit_lanes = static_map_elements[lane_id].get("exit_lanes", [])
        graph[lane_id] = []

        for exit_id in exit_lanes:
            if exit_id in static_map_elements:
                graph[lane_id].append(exit_id)

                if exit_id not in visited_lanes:
                    visited_lanes.add(exit_id)
                    queue.append((exit_id, depth + 1))

    return graph


def _compute_gt_metadata(
    graph: dict,
    root_lane: int | str,
    trajectory: np.ndarray,
    heading: np.ndarray,
    lane_data: tuple,
    max_depth: int = MAX_GRAPH_DEPTH,
) -> dict[tuple, tuple]:
    """
    Compute GT trajectory matching metadata for each edge in the graph.

    For each edge (from_lane, to_lane), calculates how well the to_lane
    matches the remaining ground truth trajectory.

    Args:
        graph: Adjacency list from _build_graph
        root_lane: Starting lane ID
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata)
        max_depth: Maximum depth to compute metadata for

    Returns:
        Dict mapping (from_lane, to_lane) to (score, next_traj_idx, traj_start_idx)
    """
    lane_ids, lane_polylines, _ = lane_data
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    metadata = {}
    queue = deque([(root_lane, 0, 0)])
    visited_states = set()

    while queue:
        lane_id, depth, traj_idx = queue.popleft()

        if (lane_id, traj_idx) in visited_states or depth >= max_depth:
            continue
        visited_states.add((lane_id, traj_idx))

        for exit_id in graph.get(lane_id, []):
            score, next_traj_idx = _score_lane_against_trajectory(
                exit_id,
                lane_id_to_idx,
                lane_polylines,
                trajectory_2d,
                heading,
                traj_idx,
            )
            metadata[(lane_id, exit_id)] = (score, next_traj_idx, traj_idx)
            queue.append((exit_id, depth + 1, next_traj_idx))

    return metadata


def _gt_coverage_metric(path: list, metadata: dict) -> float:
    """
    Calculate GT coverage score for a path.

    Sums the GT matching scores along all edges in the path.

    Args:
        path: List of lane IDs
        metadata: Edge metadata from _compute_gt_metadata

    Returns:
        Total score (sum of edge scores)
    """
    total_score = 0.0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge in metadata:
            score, _, _ = metadata[edge]
            total_score += score
    return total_score


def extract_top_n_paths(
    graph: dict,
    root_lane: int | str,
    metadata: dict,
    n: int = MAX_ROUTES,
    max_length: int = MAX_PATH_LENGTH,
) -> list[list]:
    """
    Extract top N paths from graph using GT coverage metric.

    Enumerates paths via DFS with per-path cycle detection, scores each
    path using _gt_coverage_metric, and returns the best N paths.

    Args:
        graph: Adjacency list from _build_graph
        root_lane: Starting lane ID
        metadata: Edge metadata from _compute_gt_metadata
        n: Number of top paths to return
        max_length: Maximum path length in number of lanes

    Returns:
        List of N best paths, each path is a list of lane IDs
    """
    complete_paths = []
    queue = [([root_lane], {root_lane})]

    while queue:
        path, visited = queue.pop(0)
        current = path[-1]

        if len(path) >= max_length or current not in graph:
            complete_paths.append(path)
            continue

        exits = graph.get(current, [])
        if not exits:
            complete_paths.append(path)
            continue

        has_valid_exit = False
        for exit_id in exits:
            if exit_id not in visited:
                has_valid_exit = True
                queue.append((path + [exit_id], visited | {exit_id}))

        if not has_valid_exit:
            complete_paths.append(path)

    if not complete_paths:
        return [[root_lane]]

    scored_paths = [(_gt_coverage_metric(path, metadata), path) for path in complete_paths]
    scored_paths.sort(reverse=True)

    return [path for _, path in scored_paths[:n]]


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

    # Check if agent has meaningful trajectory (not stationary)
    # Use displacement between first and last point
    trajectory_displacement = np.linalg.norm(trajectory_2d[-1] - trajectory_2d[0])

    # Stationary threshold: 0.1 meters (10 cm)
    # This accounts for GPS/sensor noise and small movements (e.g., rolling at stop sign)
    # Agents moving less than 10cm over their entire trajectory are considered stationary
    STATIONARY_THRESHOLD = 0.1  # meters

    if trajectory_displacement < STATIONARY_THRESHOLD:
        # Agent is stationary (parked, waiting at light, etc.), only check start position
        trajectory_sample = trajectory_2d[:1]
        _lane_width_threshold = 1.0  # Tighter threshold for parked agents
    else:
        # Agent is moving, check start and end positions to verify on-lane status
        trajectory_sample = np.array([trajectory_2d[0], trajectory_2d[-1]])
        _lane_width_threshold = LANE_WIDTH_THRESHOLD

    # Vectorized: Calculate distances for all sample points at once
    min_distances = _points_to_polylines_distance_batch(trajectory_sample, lane_polylines)

    # Check if any sampled point is near any lane
    return np.any(min_distances < _lane_width_threshold)


def _find_root_lane(trajectory: np.ndarray, heading: np.ndarray, lane_data: tuple) -> int | None:
    """
    Find the current lane where the agent is located using the first N trajectory points.

    Args:
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata) from extract_lane_centers

    Returns:
        The current lane ID (root lane) or None if no lane is found
    """
    lane_ids, lane_polylines, _ = lane_data

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Use only the first N points to determine current lane
    num_points_for_current_lane = min(ROOT_LANE_POINTS, len(trajectory_2d))
    first_points = trajectory_2d[:num_points_for_current_lane]
    first_headings = heading[:num_points_for_current_lane]

    # Vectorized: Calculate distances and directions for all points at once
    # Shape: (num_points, num_lanes)
    min_distances_all, closest_indices_all = _points_to_polylines_distance_batch_with_indices(
        first_points,
        lane_polylines,
    )

    # Shape: (num_points, num_lanes, 2)
    lane_directions_all = _get_lane_directions_at_indices_batch(lane_polylines, closest_indices_all)

    # Calculate agent directions for all points: (num_points, 2)
    agent_dirs = np.stack([np.cos(first_headings), np.sin(first_headings)], axis=1)

    # Vectorized alignment calculation: (num_points, num_lanes)
    # Broadcasting: (num_points, 1, 2) * (num_points, num_lanes, 2) -> sum over last axis
    alignments_all = np.sum(lane_directions_all * agent_dirs[:, np.newaxis, :], axis=2)

    # Create validity masks: (num_points, num_lanes)
    distance_valid = min_distances_all < LANE_WIDTH_THRESHOLD
    alignment_valid = alignments_all > ALIGNMENT_THRESHOLD
    valid_mask = distance_valid & alignment_valid

    # Calculate scores for all valid combinations: (num_points, num_lanes)
    distance_scores_all = 1.0 / (1.0 + min_distances_all)
    scores_all = ALIGNMENT_WEIGHT * alignments_all + DISTANCE_WEIGHT * distance_scores_all

    # Apply validity mask and sum scores across all points: (num_lanes,)
    scores_all_masked = np.where(valid_mask, scores_all, 0.0)
    lane_total_scores = np.sum(scores_all_masked, axis=0)

    # Find the lane with the highest total score
    if np.max(lane_total_scores) == 0:
        return None

    best_lane_idx = np.argmax(lane_total_scores)
    return lane_ids[best_lane_idx]


def _score_lane_against_trajectory(
    lane_id: int,
    lane_id_to_idx: dict,
    lane_polylines: np.ndarray,
    trajectory: np.ndarray,
    heading: np.ndarray,
    start_traj_idx: int,
) -> tuple[float, int]:
    """
    Score how well a lane matches the remaining ground truth trajectory.

    Args:
        lane_id: Lane ID to score (integer)
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
    lane_polyline = lane_polylines[lane_idx : lane_idx + 1]  # Keep as (1, max_points, 2) for vectorization

    # Get remaining trajectory points
    remaining_trajectory = trajectory[start_traj_idx:]
    remaining_heading = heading[start_traj_idx:]

    if len(remaining_trajectory) == 0:
        return 0.0, start_traj_idx

    # Vectorized: Calculate distances for all remaining trajectory points at once
    min_distances, closest_indices = _points_to_polylines_distance_batch_with_indices(
        remaining_trajectory,
        lane_polyline,
    )
    # Squeeze to 1D since we only have 1 lane
    min_distances = min_distances[:, 0]
    closest_indices = closest_indices[:, 0]

    # Vectorized: Calculate lane directions for all points
    lane_directions = _get_lane_directions_at_indices_batch(lane_polyline, closest_indices.reshape(-1, 1))[
        :,
        0,
        :,
    ]  # Shape: (N_points, 2)

    # Vectorized: Calculate agent directions
    agent_dirs = np.stack([np.cos(remaining_heading), np.sin(remaining_heading)], axis=1)

    # Vectorized: Calculate alignments
    alignments = np.sum(lane_directions * agent_dirs, axis=1)

    # Find first point that violates constraints
    distance_valid = min_distances <= LANE_WIDTH_THRESHOLD
    alignment_valid = alignments >= ALIGNMENT_THRESHOLD
    valid_mask = distance_valid & alignment_valid

    # Find the first invalid point (if any)
    if not np.any(valid_mask):
        return 0.0, start_traj_idx

    # Find where validity breaks (first False)
    valid_until = np.argmax(~valid_mask) if not np.all(valid_mask) else len(valid_mask)
    if valid_until == 0:
        return 0.0, start_traj_idx

    # Calculate scores for valid points only
    distance_scores = 1.0 / (1.0 + min_distances[:valid_until])
    point_scores = ALIGNMENT_WEIGHT * alignments[:valid_until] + DISTANCE_WEIGHT * distance_scores

    # Calculate average score
    avg_score = np.mean(point_scores)
    next_traj_idx = start_traj_idx + valid_until

    return float(avg_score), next_traj_idx


def _points_to_polylines_distance_batch(points: np.ndarray, polylines: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of minimum distances from multiple points to multiple polylines.

    Optimized for batch processing of points with reduced memory allocations.

    Args:
        points: 2D points array (N_points, 2)
        polylines: Array of polylines (N_lanes, max_points, 2) - padded with [0, 0]

    Returns:
        Array of minimum distances for each point-lane pair (N_points, N_lanes)
    """
    n_points = len(points)
    n_lanes = len(polylines)
    max_segments = polylines.shape[1] - 1

    # Extract segment endpoints: (N_lanes, max_segments, 2)
    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

    # Detect valid segments (exclude padding and transitions to padding)
    # A segment is valid only if BOTH endpoints are non-zero
    # Shape: (N_lanes, max_segments)
    starts_nonzero = np.any(seg_starts != 0, axis=2)  # True if start point is not [0, 0]
    ends_nonzero = np.any(seg_ends != 0, axis=2)  # True if end point is not [0, 0]
    valid_segs = starts_nonzero & ends_nonzero  # Both must be non-zero

    # Compute segment vectors and squared lengths
    seg_vecs = seg_ends - seg_starts  # (N_lanes, max_segments, 2)
    seg_lens_sq = np.einsum("ijk,ijk->ij", seg_vecs, seg_vecs)  # (N_lanes, max_segments)

    # Additional check: filter zero-length valid segments (degenerate polylines)
    valid_segs = valid_segs & (seg_lens_sq > 1e-10)
    seg_lens_sq_safe = seg_lens_sq + 1e-10  # Avoid division by zero

    # Reshape for broadcasting
    seg_starts_bc = seg_starts.reshape(1, n_lanes, max_segments, 2)  # (1, N_lanes, max_segments, 2)
    seg_vecs_bc = seg_vecs.reshape(1, n_lanes, max_segments, 2)  # (1, N_lanes, max_segments, 2)
    seg_lens_sq_bc = seg_lens_sq_safe.reshape(1, n_lanes, max_segments)  # (1, N_lanes, max_segments)
    valid_segs_bc = valid_segs.reshape(1, n_lanes, max_segments)  # (1, N_lanes, max_segments)
    points_bc = points.reshape(n_points, 1, 1, 2)  # (N_points, 1, 1, 2)

    # Project each point onto each segment
    # t = (point - seg_start) · seg_vec / |seg_vec|²
    # Shape: (N_points, N_lanes, max_segments)
    point_to_start = points_bc - seg_starts_bc  # (N_points, N_lanes, max_segments, 2)
    t = np.einsum("ijkl,jkl->ijk", point_to_start, seg_vecs) / seg_lens_sq_bc
    t = np.clip(t, 0, 1, out=t)  # Clamp to [0, 1] for segment bounds

    # Find closest point on segment: closest = seg_start + t * seg_vec
    # Shape: (N_points, N_lanes, max_segments, 2)
    t_expanded = t[..., np.newaxis]  # (N_points, N_lanes, max_segments, 1)
    closest_on_seg = seg_starts_bc + t_expanded * seg_vecs_bc

    # Compute squared distance from point to closest point on segment
    # Shape: (N_points, N_lanes, max_segments)
    diff = points_bc - closest_on_seg
    distances_sq = np.einsum("ijkl,ijkl->ijk", diff, diff)

    # Mask invalid segments with infinite distance
    distances_sq = np.where(valid_segs_bc, distances_sq, np.inf)

    # Find minimum distance across all segments for each point-lane pair
    min_distances_sq = np.min(distances_sq, axis=2)  # (N_points, N_lanes)
    min_distances = np.sqrt(min_distances_sq)

    return min_distances


def _points_to_polylines_distance_batch_with_indices(
    points: np.ndarray,
    polylines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized calculation of minimum distances and closest segment indices from multiple points to polylines.

    Optimized version that reduces memory allocations and uses squared distances where possible.

    Args:
        points: 2D points array (N_points, 2)
        polylines: Array of polylines (N_lanes, max_points, 2) - padded with [0, 0]

    Returns:
        Tuple of:
        - min_distances: Array of minimum distances (N_points, N_lanes)
        - closest_indices: Array of closest segment indices (N_points, N_lanes)
    """
    n_points = len(points)
    n_lanes = len(polylines)
    max_segments = polylines.shape[1] - 1

    # Extract segment endpoints: (N_lanes, max_segments, 2)
    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

    # Detect valid segments (exclude padding and transitions to padding)
    # A segment is valid only if BOTH endpoints are non-zero
    # Shape: (N_lanes, max_segments)
    starts_nonzero = np.any(seg_starts != 0, axis=2)  # True if start point is not [0, 0]
    ends_nonzero = np.any(seg_ends != 0, axis=2)  # True if end point is not [0, 0]
    valid_segs = starts_nonzero & ends_nonzero  # Both must be non-zero

    # Compute segment vectors and squared lengths
    seg_vecs = seg_ends - seg_starts  # (N_lanes, max_segments, 2)
    seg_lens_sq = np.einsum("ijk,ijk->ij", seg_vecs, seg_vecs)  # (N_lanes, max_segments)

    # Additional check: filter zero-length valid segments (degenerate polylines)
    valid_segs = valid_segs & (seg_lens_sq > 1e-10)
    seg_lens_sq_safe = seg_lens_sq + 1e-10  # Avoid division by zero

    # Reshape for broadcasting
    seg_starts_bc = seg_starts.reshape(1, n_lanes, max_segments, 2)  # (1, N_lanes, max_segments, 2)
    seg_vecs_bc = seg_vecs.reshape(1, n_lanes, max_segments, 2)  # (1, N_lanes, max_segments, 2)
    seg_lens_sq_bc = seg_lens_sq_safe.reshape(1, n_lanes, max_segments)  # (1, N_lanes, max_segments)
    valid_segs_bc = valid_segs.reshape(1, n_lanes, max_segments)  # (1, N_lanes, max_segments)
    points_bc = points.reshape(n_points, 1, 1, 2)  # (N_points, 1, 1, 2)

    # Project each point onto each segment
    # t = (point - seg_start) · seg_vec / |seg_vec|²
    # Shape: (N_points, N_lanes, max_segments)
    point_to_start = points_bc - seg_starts_bc  # (N_points, N_lanes, max_segments, 2)
    t = np.einsum("ijkl,jkl->ijk", point_to_start, seg_vecs) / seg_lens_sq_bc
    t = np.clip(t, 0, 1, out=t)  # Clamp to [0, 1] for segment bounds

    # Find closest point on segment: closest = seg_start + t * seg_vec
    # Shape: (N_points, N_lanes, max_segments, 2)
    t_expanded = t[..., np.newaxis]  # (N_points, N_lanes, max_segments, 1)
    closest_on_seg = seg_starts_bc + t_expanded * seg_vecs_bc

    # Compute squared distance from point to closest point on segment
    # Shape: (N_points, N_lanes, max_segments)
    diff = points_bc - closest_on_seg
    distances_sq = np.einsum("ijkl,ijkl->ijk", diff, diff)

    # Mask invalid segments with infinite distance
    distances_sq = np.where(valid_segs_bc, distances_sq, np.inf)

    # Find minimum distance and closest segment index for each point-lane pair
    closest_indices = np.argmin(distances_sq, axis=2).astype(np.int32)  # (N_points, N_lanes)
    min_distances_sq = np.min(distances_sq, axis=2)  # (N_points, N_lanes)

    # Take sqrt at the end
    min_distances = np.sqrt(min_distances_sq)

    return min_distances, closest_indices


def _get_lane_directions_at_indices_batch(polylines: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Get lane directions at specified segment indices for multiple points (fully vectorized).

    Args:
        polylines: Array of polylines (N_lanes, max_points, 2)
        indices: Array of segment indices (N_points, N_lanes)

    Returns:
        Array of normalized direction vectors (N_points, N_lanes, 2)
    """
    n_points, n_lanes = indices.shape
    max_points = polylines.shape[1]

    # Create meshgrid for lane indices (point_idx not needed due to numpy broadcasting)
    lane_idx = np.arange(n_lanes)[np.newaxis, :]  # (1, N_lanes)

    # Get segment start and end points
    seg_starts = polylines[lane_idx, indices, :]  # (N_points, N_lanes, 2)

    # Handle edge cases: if index is at the end, use previous segment
    next_indices = np.minimum(indices + 1, max_points - 1)
    seg_ends = polylines[lane_idx, next_indices, :]  # (N_points, N_lanes, 2)

    # Calculate direction vectors
    directions = seg_ends - seg_starts  # (N_points, N_lanes, 2)

    # Normalize
    norms = np.linalg.norm(directions, axis=2, keepdims=True)  # (N_points, N_lanes, 1)
    directions = directions / (norms + 1e-6)

    return directions

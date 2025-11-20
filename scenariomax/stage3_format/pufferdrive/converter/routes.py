"""
Compute agent routes for Puffer format.

Routes are lists of lane center IDs that cover the ground truth trajectory
and explore different possible paths through the road network.

Algorithm Overview:
------------------
The route computation uses a 3-step graph-based approach:

1. Validation: Check if agent has sufficient trajectory points and is on lanes (not parked)
2. Root Lane Selection: Find the current lane using sample trajectory points
3. Graph Building: Build full reachability graph from root lane (no pruning)
4. Path Extraction: Enumerate all paths, score each using geometric distance to GT trajectory

Multiple route paths are generated to represent different possible paths through
exit lanes (e.g., at highway interchanges or intersections).

Geometric Scoring:
-----------------
Each route path is scored by:
1. Concatenating lane polylines into a single route polyline
2. Computing distance from each GT trajectory point to the route polyline
3. Calculating coverage (% of trajectory within LANE_WIDTH_THRESHOLD)
4. Calculating distance_score (1 / (1 + avg_distance) for covered points)
5. Final score = coverage × distance_score

Routes are ranked by score (higher is better), with an optional heading alignment
filter to reject routes where the agent travels in the wrong direction.
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

# Root lane selection weights
# These weights are used only for finding the initial root lane (current lane)
# The scoring formula is: score = ALIGNMENT_WEIGHT * alignment + DISTANCE_WEIGHT * distance_score
# where alignment ∈ [-1, 1] and distance_score = 1/(1+distance) ∈ (0, 1]
ALIGNMENT_WEIGHT = 0.7  # Weight for directional alignment (heading match)
DISTANCE_WEIGHT = 0.3  # Weight for spatial proximity
# Combined weights sum to 1.0 for interpretable normalized scoring
# Note: Route scoring uses geometric distance (coverage × distance_score), not these weights


def compute_agent_route(
    agent_data: tuple,
    static_map_elements: dict,
    lane_data: tuple,
    min_route_valid_points: int = 0,
    max_routes: int = 10,
    route_check_timestep: int = 0,
) -> list[list[int]]:
    """
    Compute routes (lists of lane IDs) for an agent based on ground truth trajectory.

    Algorithm:
    1. Find root lane using sample trajectory points
    2. Build full reachability graph from root (no pruning)
    3. Enumerate all paths and score each using geometric distance to GT trajectory

    Multiple route paths are generated to explore different exit lane possibilities.

    Args:
        agent_data: Tuple of (agent_id, position, heading, valid, length, width)
        static_map_elements: Dict of static map elements (lanes, boundaries, etc.)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        min_route_valid_points: Minimum valid trajectory points required (0 = no filtering)
        max_routes: Maximum number of route paths to generate (default: 10)
        route_check_timestep: Timestep to check if agent is offroad (default: 0)

    Returns:
        List of routes, where each route is a list of lane center IDs
    """
    agent_id, agent_trajectory, agent_heading, agent_valid, agent_length, agent_width = agent_data

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

    lane_ids, lane_polylines, _, _ = lane_data

    if len(lane_ids) == 0:
        logger.debug(f"{agent_str}: No lane centers found in map")
        return []

    # Check if agent is offroad at check timestep (bbox crosses road edge OR >5m from lane)
    offroad_agent_data = (agent_id, agent_trajectory, agent_heading, agent_valid, agent_length, agent_width)
    if _is_offroad_at_init(offroad_agent_data, static_map_elements, lane_polylines, route_check_timestep):
        logger.debug(f"{agent_str}: Off-road at timestep {route_check_timestep}")
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

    # Step 3: Extract top N paths using geometric GT coverage metric
    routes = extract_top_n_paths(graph, root_lane, lane_data, valid_trajectory, valid_heading, n=max_routes)

    return routes


def extract_lane_centers(static_map_elements: dict) -> tuple[list, np.ndarray, dict, np.ndarray]:
    """
    Extract lane center information as numpy arrays for vectorized operations.

    Args:
        static_map_elements: Dict of static map elements

    Returns:
        Tuple of:
        - lane_ids: List of lane IDs (strings)
        - lane_polylines: Array of lane polylines, padded to max length (N_lanes, max_points, 2)
        - lane_metadata: Dict mapping lane_id to connectivity info
        - lane_lengths: Array of actual polyline lengths (N_lanes,) for each lane
    """
    lane_ids = []
    lane_polylines_list = []
    lane_lengths_list = []
    lane_metadata = {}
    max_points = 0

    # First pass: collect lanes and find max polyline length
    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        # Only process lane centers
        if element_type == types.LANE_SURFACE_STREET or element_type == types.LANE_FREEWAY:
            polyline = element_data["polyline"]

            if len(polyline) > 0:
                # Convert to 2D if needed
                polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline

                lane_ids.append(element_id)
                lane_polylines_list.append(polyline_2d)
                lane_lengths_list.append(len(polyline_2d))
                max_points = max(max_points, len(polyline_2d))

                lane_metadata[element_id] = {
                    "entry_lanes": element_data["entry_lanes"],
                    "exit_lanes": element_data["exit_lanes"],
                }

    if not lane_ids:
        return [], np.array([]), {}, np.array([])

    # Second pass: create padded array
    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float32)
    lane_lengths = np.array(lane_lengths_list, dtype=np.int32)

    for i, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[i, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata, lane_lengths


def _build_route_polyline(route_path: list, lane_data: tuple) -> np.ndarray:
    """
    Build a single continuous polyline from a route path by concatenating lane polylines.

    Args:
        route_path: List of lane IDs forming the route
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)

    Returns:
        Concatenated polyline array (N_points, 2) representing the full route
    """
    lane_ids, lane_polylines, _, lane_lengths = lane_data
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

    route_polyline_segments = []
    for lane_id in route_path:
        if lane_id not in lane_id_to_idx:
            continue

        idx = lane_id_to_idx[lane_id]
        length = lane_lengths[idx]
        polyline_valid = lane_polylines[idx, :length, :]  # (length, 2)

        if length > 0:
            route_polyline_segments.append(polyline_valid)

    if not route_polyline_segments:
        return np.array([]).reshape(0, 2)

    # Concatenate all segments into single polyline
    route_polyline = np.vstack(route_polyline_segments)

    return route_polyline


def _check_route_heading_alignment(
    route_polyline: np.ndarray,
    trajectory: np.ndarray,
    heading: np.ndarray,
    min_alignment_ratio: float = 0.7,
) -> bool:
    """
    Check if trajectory heading generally aligns with route direction (binary filter).

    Samples points from trajectory and checks if agent is traveling in the same
    direction as the closest route segment.

    Args:
        route_polyline: Route polyline (N_points, 2)
        trajectory: Trajectory points (M, 2)
        heading: Trajectory headings (M,)
        min_alignment_ratio: Minimum fraction of samples that must align (default 0.7)

    Returns:
        True if route direction generally matches trajectory heading, False otherwise
    """
    if len(route_polyline) < 2 or len(trajectory) == 0:
        return False

    # Sample 10 points evenly from trajectory
    n_samples = min(10, len(trajectory))
    sample_indices = np.linspace(0, len(trajectory) - 1, n_samples, dtype=int)
    sample_positions = trajectory[sample_indices]
    sample_headings = heading[sample_indices]

    # Reshape route polyline for distance calculation
    route_polyline_batch = route_polyline[np.newaxis, :, :]  # (1, N_points, 2)

    # Find closest route segment for each sample
    _, closest_indices = _points_to_polylines_distance(sample_positions, route_polyline_batch)
    closest_indices = closest_indices[:, 0]  # Squeeze to 1D

    # Get route directions at closest segments
    route_directions = _get_lane_directions_at_indices_batch(route_polyline_batch, closest_indices.reshape(-1, 1))[
        :, 0, :
    ]  # (n_samples, 2)

    # Calculate agent directions
    agent_dirs = np.stack([np.cos(sample_headings), np.sin(sample_headings)], axis=1)

    # Calculate alignment (dot product)
    alignments = np.sum(route_directions * agent_dirs, axis=1)

    # Check if enough samples are aligned
    aligned_count = np.sum(alignments > ALIGNMENT_THRESHOLD)
    alignment_ratio = aligned_count / n_samples

    return alignment_ratio >= min_alignment_ratio


def _score_route_geometric(
    route_path: list,
    lane_data: tuple,
    trajectory: np.ndarray,
    heading: np.ndarray,
) -> float:
    """
    Score route using geometric distance from trajectory to full route polyline.

    Uses multiplicative scoring: coverage × distance_score, where:
    - coverage: Percentage of trajectory points within LANE_WIDTH_THRESHOLD of route
    - distance_score: 1 / (1 + avg_distance) for covered points

    Args:
        route_path: List of lane IDs forming the route
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        trajectory: Trajectory points (M, 2/3)
        heading: Trajectory headings (M,)

    Returns:
        Score value where higher is better (0.0 if route doesn't match trajectory)
    """
    # Ensure trajectory is 2D
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Build route polyline
    route_polyline = _build_route_polyline(route_path, lane_data)

    if len(route_polyline) < 2:
        return 0.0

    # Apply heading alignment filter
    if not _check_route_heading_alignment(route_polyline, trajectory_2d, heading):
        return 0.0

    # Reshape route polyline for distance calculation
    route_polyline_batch = route_polyline[np.newaxis, :, :]  # (1, N_points, 2)

    # Compute distances from all trajectory points to route polyline
    min_distances = _points_to_polylines_distance(trajectory_2d, route_polyline_batch, return_indices=False)
    min_distances = min_distances[:, 0]  # Squeeze to 1D (N_traj_points,)

    # Compute coverage: percentage of trajectory within threshold
    coverage_mask = min_distances < LANE_WIDTH_THRESHOLD
    coverage_ratio = np.sum(coverage_mask) / len(trajectory_2d)

    if coverage_ratio == 0:
        return 0.0

    # Compute average distance for covered points only
    covered_distances = min_distances[coverage_mask]
    avg_distance = np.mean(covered_distances)
    distance_score = 1.0 / (1.0 + avg_distance)

    # Multiplicative score: coverage × distance_score
    final_score = coverage_ratio * distance_score

    return final_score


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

        exit_lanes = static_map_elements[lane_id]["exit_lanes"]
        graph[lane_id] = []

        for exit_id in exit_lanes:
            if exit_id in static_map_elements:
                graph[lane_id].append(exit_id)

                if exit_id not in visited_lanes:
                    visited_lanes.add(exit_id)
                    queue.append((exit_id, depth + 1))

    return graph


def _gt_coverage_metric(
    path: list,
    lane_data: tuple,
    trajectory: np.ndarray,
    heading: np.ndarray,
) -> float:
    """
    Calculate GT coverage score for a path using geometric distance.

    Delegates to _score_route_geometric which computes coverage × distance_score.

    Args:
        path: List of lane IDs
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata)
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)

    Returns:
        Score value where higher is better (0.0 if route doesn't match trajectory)
    """
    return _score_route_geometric(path, lane_data, trajectory, heading)


def extract_top_n_paths(
    graph: dict,
    root_lane: int | str,
    lane_data: tuple,
    trajectory: np.ndarray,
    heading: np.ndarray,
    n: int = MAX_ROUTES,
    max_length: int = MAX_PATH_LENGTH,
) -> list[list]:
    """
    Extract top N paths from graph using geometric GT coverage metric.

    Enumerates paths via DFS with per-path cycle detection, scores each
    path using geometric distance from trajectory to route polyline, and
    returns the best N paths.

    Args:
        graph: Adjacency list from _build_graph
        root_lane: Starting lane ID
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
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

    scored_paths = [(_gt_coverage_metric(path, lane_data, trajectory, heading), path) for path in complete_paths]
    scored_paths.sort(reverse=True)

    return [path for _, path in scored_paths[:n]]


def _find_root_lane(trajectory: np.ndarray, heading: np.ndarray, lane_data: tuple) -> int | None:
    """
    Find the current lane where the agent is located using strategic sample points.

    Uses 1st, 3rd, 5th, middle, and last valid points for better direction estimation.

    Args:
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths) from extract_lane_centers

    Returns:
        The current lane ID (root lane) or None if no lane is found
    """
    lane_ids, lane_polylines, _, _ = lane_data

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Use strategic sample points: 1st, 3rd, 5th, middle, last
    traj_len = len(trajectory_2d)
    sample_indices = []

    # Add indices if they exist and are unique
    for idx in [0, 2, 4, traj_len // 2, -1]:
        # Normalize negative indices
        actual_idx = idx if idx >= 0 else traj_len + idx
        if 0 <= actual_idx < traj_len and actual_idx not in sample_indices:
            sample_indices.append(actual_idx)

    # Extract sample points and headings
    first_points = trajectory_2d[sample_indices]
    first_headings = heading[sample_indices]

    # Vectorized: Calculate distances and directions for all points at once
    # Shape: (num_points, num_lanes)
    min_distances_all, closest_indices_all = _points_to_polylines_distance(first_points, lane_polylines)

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


def _points_to_polylines_distance(
    points: np.ndarray,
    polylines: np.ndarray,
    return_indices: bool = True,
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

    # Find minimum distance and optionally closest segment index
    if return_indices:
        closest_indices = np.argmin(distances_sq, axis=2).astype(np.int32)  # (N_points, N_lanes)
        min_distances_sq = np.min(distances_sq, axis=2)  # (N_points, N_lanes)
        min_distances = np.sqrt(min_distances_sq)
        return min_distances, closest_indices
    else:
        min_distances_sq = np.min(distances_sq, axis=2)  # (N_points, N_lanes)
        min_distances = np.sqrt(min_distances_sq)
        return min_distances


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

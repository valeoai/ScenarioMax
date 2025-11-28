"""
Compute agent routes for Puffer format.

Routes are lists of lane center IDs that cover the ground truth trajectory
by selecting the best root lane and greedily extending it.

Algorithm Overview:
------------------
The route computation uses a 3-step approach:

1. Root Lane Candidates: Find 1-3 candidate root lanes using sample trajectory points
   - Only candidates with score >= ROOT_LANE_MIN_SCORE are considered
   - Sorted by score (alignment + distance)
2. Greedy Extension: For each candidate, iteratively pick exit lane that best covers trajectory
   - Stop when no valid exits OR max length reached
3. Best Route Selection: Score all candidate routes and return the best one

Geometric Scoring:
-----------------
Each candidate route is scored by:
1. Concatenating lane polylines into a single route polyline
2. Computing distance from each GT trajectory point to the route polyline
3. Calculating coverage (% of trajectory within LANE_WIDTH_THRESHOLD)
4. Calculating distance_score (1 / (1 + avg_distance) for covered points)
5. Final score = coverage × distance_score

Routes with heading misalignment (agent traveling wrong direction) are rejected.
"""

import numpy as np

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)


# Constants for route computation
# Physical thresholds
LANE_WIDTH_THRESHOLD = 6.0  # Meters - maximum distance from lane center to consider agent "on lane"
ALIGNMENT_THRESHOLD = 0.3  # Cosine similarity threshold for direction alignment
# Value of 0.3 corresponds to ~72.5° angle deviation (arccos(0.3) ≈ 72.5°)
# Allows moderate heading mismatch while filtering out wrong-way or perpendicular lanes
# Range: [-1, 1] where 1 = perfect alignment, 0 = perpendicular, -1 = opposite direction

# Algorithm parameters
MAX_PATH_LENGTH = 10  # Maximum number of lanes per route path
MAX_ROOT_CANDIDATES = 3  # Maximum number of root lane candidates to try
ROOT_LANE_MIN_SCORE = 0.3  # Minimum score for root lane candidate to be considered

# Root lane selection weights
# These weights are used only for finding the initial root lane (current lane)
# The scoring formula is: score = ALIGNMENT_WEIGHT * alignment + DISTANCE_WEIGHT * distance_score
# where alignment ∈ [-1, 1] and distance_score = 1/(1+distance) ∈ (0, 1]
ALIGNMENT_WEIGHT = 0.7  # Weight for directional alignment (heading match)
DISTANCE_WEIGHT = 0.3  # Weight for spatial proximity
# Combined weights sum to 1.0 for interpretable normalized scoring
# Note: Route scoring uses geometric distance (coverage × distance_score), not these weights

# Controllable vehicle filtering constants
OFFROAD_DISTANCE_THRESHOLD = 5.0  # Meters - max distance from closest lane to be considered on-road
ROAD_EDGE_TYPES = {
    types.ROAD_EDGE_BOUNDARY,
    types.ROAD_EDGE_MEDIAN,
    types.ROAD_EDGE_SIDEWALK,
}


def compute_agent_route(
    agent_data: tuple,
    static_map_elements: dict,
    lane_data: tuple,
    min_route_valid_points: int = 0,
    route_check_timestep: int = 0,
) -> list[list[int]]:
    """
    Compute single route (list of lane IDs) for an agent based on ground truth trajectory.

    Algorithm:
    1. Find 1-3 root lane candidates using sample trajectory points
    2. For each candidate, greedily extend route by picking best exit lane
    3. Score each complete route and return the best one

    Args:
        agent_data: Tuple of (agent_id, position, heading, valid, length, width)
        static_map_elements: Dict of static map elements (lanes, boundaries, etc.)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        min_route_valid_points: Minimum valid trajectory points required (0 = no filtering)
        route_check_timestep: Timestep to check if agent is offroad (default: 0)

    Returns:
        List containing single best route, where route is a list of lane center IDs
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

    lane_ids, lane_polylines, _, lane_lengths = lane_data

    if len(lane_ids) == 0:
        logger.debug(f"{agent_str}: No lane centers found in map")
        return []

    # Check if agent is offroad at check timestep (bbox crosses road edge OR >5m from lane)
    offroad_agent_data = (agent_id, agent_trajectory, agent_heading, agent_valid, agent_length, agent_width)
    if _is_offroad_at_init(
        offroad_agent_data,
        static_map_elements,
        lane_polylines,
        lane_lengths,
        route_check_timestep,
    ):
        logger.debug(f"{agent_str}: Off-road at timestep {route_check_timestep}")
        return []

    # Step 1: Find 1-3 root lane candidates
    root_candidates = _find_root_lane_candidates(valid_trajectory, valid_heading, lane_data)

    if not root_candidates:
        logger.debug(f"{agent_str}: No current lane found")
        return []

    # Step 2: For each candidate, compute route and score it
    best_route = None
    best_score = 0.0

    for root_lane, root_score in root_candidates:
        # Compute greedy route from this root (returns route and final score)
        candidate_route, route_score = _compute_route(
            root_lane,
            static_map_elements,
            lane_data,
            valid_trajectory,
            valid_heading,
        )

        # Update best if this is better
        if route_score > best_score:
            best_score = route_score
            best_route = candidate_route

    # Return best route
    if best_route is None:
        logger.debug(f"{agent_str}: No valid route found from candidates")
        return []

    return [best_route]


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
    direction as the closest route segment. Uses two thresholds:
    - ALIGNMENT_THRESHOLD (0.3 cosine similarity): Per-point heading alignment check
    - min_alignment_ratio (0.7 default): Fraction of points that must pass alignment check

    Args:
        route_polyline: Route polyline (N_points, 2)
        trajectory: Trajectory points (M, 2)
        heading: Trajectory headings (M,)
        min_alignment_ratio: Minimum fraction of samples that must align with route direction
            (default 0.7 = 70% of points)

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
    route_lengths = np.array([len(route_polyline)], dtype=np.int32)

    # Find closest route segment for each sample
    _, closest_indices = _points_to_polylines_distance(
        sample_positions,
        route_polyline_batch,
        polyline_lengths=route_lengths,
    )
    closest_indices = closest_indices[:, 0]  # Squeeze to 1D

    # Get route directions at closest segments
    route_directions = _get_lane_directions_at_indices_batch(route_polyline_batch, closest_indices.reshape(-1, 1))[
        :,
        0,
        :,
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

    return _score_route_polyline(route_polyline, trajectory_2d, heading)


def _score_route_polyline(
    route_polyline: np.ndarray,
    trajectory_2d: np.ndarray,
    heading: np.ndarray,
) -> float:
    """Score a pre-built route polyline against a trajectory."""
    if len(route_polyline) < 2 or len(trajectory_2d) == 0:
        return 0.0

    # Apply heading alignment filter
    if not _check_route_heading_alignment(route_polyline, trajectory_2d, heading):
        return 0.0

    # Reshape route polyline for distance calculation
    route_polyline_batch = route_polyline[np.newaxis, :, :]  # (1, N_points, 2)
    route_lengths = np.array([len(route_polyline)], dtype=np.int32)

    # Compute distances from all trajectory points to route polyline
    min_distances = _points_to_polylines_distance(
        trajectory_2d,
        route_polyline_batch,
        polyline_lengths=route_lengths,
        return_indices=False,
    )
    min_distances = min_distances[:, 0]  # Squeeze to 1D (N_traj_points,)

    # Compute coverage: percentage of trajectory within threshold
    coverage_mask = min_distances < LANE_WIDTH_THRESHOLD
    coverage_ratio = np.sum(coverage_mask) / len(trajectory_2d)

    if coverage_ratio == 0:
        return 0.0

    # Compute average distance for covered points only
    covered_distances = min_distances[coverage_mask]
    avg_distance = np.mean(covered_distances)

    # Fail fast if invalid data (e.g., degenerate polylines producing NaN/Inf)
    if not np.isfinite(avg_distance):
        raise ValueError(
            f"Invalid route distance: {avg_distance}. "
            "This may indicate degenerate polylines (zero-length segments) in the route.",
        )

    distance_score = 1.0 / (1.0 + avg_distance)

    # Multiplicative score: coverage × distance_score
    final_score = coverage_ratio * distance_score

    return final_score


def _compute_route(
    root_lane: int | str,
    static_map_elements: dict,
    lane_data: tuple,
    trajectory: np.ndarray,
    heading: np.ndarray,
    max_length: int = MAX_PATH_LENGTH,
) -> tuple[list, float]:
    """
    Compute single route using greedy lane selection.

    Starting from root lane, iteratively picks exit lane that best covers
    trajectory when added to route. Stops when no valid exits OR max length reached.

    Args:
        root_lane: Starting lane ID
        static_map_elements: Dict of static map elements
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        max_length: Maximum number of lanes in route (default: 10)

    Returns:
        Tuple containing:
        - List of lane IDs forming the route
        - Final geometric score for the constructed route
    """
    lane_ids, lane_polylines, _, lane_lengths = lane_data
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

    def _lane_polyline(lane_id: int | str) -> np.ndarray:
        idx = lane_id_to_idx.get(lane_id)
        if idx is None:
            return np.zeros((0, 2), dtype=np.float32)
        length = lane_lengths[idx]
        return lane_polylines[idx, :length, :]

    def _append_polyline(base: np.ndarray, addition: np.ndarray) -> np.ndarray:
        if len(addition) == 0:
            return base.copy()
        if len(base) == 0:
            return addition.copy()
        if np.allclose(base[-1], addition[0]):
            return np.vstack((base, addition[1:]))
        return np.vstack((base, addition))

    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    route = [root_lane]
    current_lane = root_lane
    visited = {root_lane}
    route_polyline = _lane_polyline(root_lane)
    current_score = _score_route_polyline(route_polyline, trajectory_2d, heading)

    for _ in range(max_length - 1):  # Already have root lane
        # Get exit lanes
        if current_lane not in static_map_elements:
            break

        exit_lanes = static_map_elements[current_lane]["exit_lanes"]

        if not exit_lanes:
            break

        # Find best exit lane
        best_exit = None
        best_exit_score = 0.0
        best_exit_polyline = None

        for exit_id in exit_lanes:
            # Skip already visited lanes (cycle prevention)
            if exit_id in visited:
                continue

            # Skip lanes not in map
            if exit_id not in static_map_elements:
                continue

            # Score candidate route with this exit
            exit_polyline = _lane_polyline(exit_id)
            if len(exit_polyline) == 0:
                continue

            candidate_polyline = _append_polyline(route_polyline, exit_polyline)
            score = _score_route_polyline(candidate_polyline, trajectory_2d, heading)

            if score > best_exit_score:
                best_exit_score = score
                best_exit = exit_id
                best_exit_polyline = candidate_polyline

        # Stop if no valid exit found
        if best_exit is None or best_exit_polyline is None:
            break

        # Add best exit to route
        route.append(best_exit)
        visited.add(best_exit)
        current_lane = best_exit
        route_polyline = best_exit_polyline
        current_score = best_exit_score

    return route, current_score


def _find_root_lane_candidates(
    trajectory: np.ndarray,
    heading: np.ndarray,
    lane_data: tuple,
    max_candidates: int = MAX_ROOT_CANDIDATES,
    min_score: float = ROOT_LANE_MIN_SCORE,
) -> list[tuple[int | str, float]]:
    """
    Find top 1-3 root lane candidates using strategic sample points.

    Uses 1st, 3rd, 5th, middle, and last valid points for better direction estimation.
    Returns candidates with score >= min_score, sorted by score (best first).

    Args:
        trajectory: Valid trajectory points (M, 2/3)
        heading: Valid headings (M,)
        lane_data: Tuple of (lane_ids, lane_polylines, lane_metadata, lane_lengths)
        max_candidates: Maximum number of candidates to return (default: 3)
        min_score: Minimum score threshold for candidates (default: 0.3)

    Returns:
        List of (lane_id, score) tuples, sorted by score descending (1-3 candidates)
    """
    lane_ids, lane_polylines, _, lane_lengths = lane_data

    # Extract 2D positions
    trajectory_2d = trajectory[:, :2] if trajectory.shape[1] == 3 else trajectory

    # Use strategic sample points: 1st, 3rd, 5th, middle, last
    traj_len = len(trajectory_2d)
    sample_indices = []

    # Add indices if they exist and are unique
    for idx in [0, 2, 4, 6]:
        # Normalize negative indices
        actual_idx = idx if idx >= 0 else traj_len + idx
        if 0 <= actual_idx < traj_len and actual_idx not in sample_indices:
            sample_indices.append(actual_idx)

    if not sample_indices:
        return []

    # Extract sample points and headings
    first_points = trajectory_2d[sample_indices]
    first_headings = heading[sample_indices]

    # Deduplicate nearly identical points to avoid overweighting stationary agents
    if len(first_points) > 1:
        keep_mask = np.ones(len(first_points), dtype=bool)
        unique_points = []
        for idx, point in enumerate(first_points):
            if any(np.linalg.norm(point - seen) < 1e-3 for seen in unique_points):
                keep_mask[idx] = False
            else:
                unique_points.append(point)

        first_points = first_points[keep_mask]
        first_headings = first_headings[keep_mask]

    if len(first_points) == 0:
        return []

    # Vectorized: Calculate distances and directions for all points at once
    # Shape: (num_points, num_lanes)
    min_distances_all, closest_indices_all = _points_to_polylines_distance(
        first_points,
        lane_polylines,
        polyline_lengths=lane_lengths,
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

    # Find lanes with score >= min_score
    valid_lane_indices = np.where(lane_total_scores >= min_score)[0]

    if len(valid_lane_indices) == 0:
        return []

    # Get scores for valid lanes
    valid_scores = lane_total_scores[valid_lane_indices]

    # Sort by score (descending)
    sorted_indices = np.argsort(valid_scores)[::-1]

    # Take top max_candidates
    top_indices = sorted_indices[:max_candidates]

    # Build result list
    candidates = []
    for idx in top_indices:
        lane_idx = valid_lane_indices[idx]
        lane_id = lane_ids[lane_idx]
        score = lane_total_scores[lane_idx]
        candidates.append((lane_id, score))

    return candidates


def _points_to_polylines_distance(
    points: np.ndarray,
    polylines: np.ndarray,
    polyline_lengths: np.ndarray | None = None,
    return_indices: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Vectorized calculation of minimum distances and closest segment indices from multiple points to polylines.

    Optimized version that reduces memory allocations and uses squared distances where possible.

    Args:
        points: 2D points array (N_points, 2)
        polylines: Array of polylines (N_lanes, max_points, 2) - padded with [0, 0]
        polyline_lengths: Optional array with the number of valid points per polyline (N_lanes,)

    Returns:
        Tuple of:
        - min_distances: Array of minimum distances (N_points, N_lanes)
        - closest_indices: Array of closest segment indices (N_points, N_lanes)
    """
    n_points = len(points)
    n_lanes = len(polylines)
    max_segments = polylines.shape[1] - 1

    if n_points == 0 or n_lanes == 0 or max_segments <= 0:
        empty_distances = np.zeros((n_points, n_lanes), dtype=np.float32)
        empty_indices = np.zeros((n_points, n_lanes), dtype=np.int32)
        return (empty_distances, empty_indices) if return_indices else empty_distances

    # Extract segment endpoints: (N_lanes, max_segments, 2)
    seg_starts = polylines[:, :-1, :]
    seg_ends = polylines[:, 1:, :]

    if polyline_lengths is not None:
        polyline_lengths = np.asarray(polyline_lengths, dtype=np.int32)
        if len(polyline_lengths) != n_lanes:
            raise ValueError("polyline_lengths must match number of polylines")
        seg_counts = np.clip(polyline_lengths - 1, 0, max_segments)
        segment_indices = np.arange(max_segments)[np.newaxis, :]
        valid_segs = segment_indices < seg_counts[:, np.newaxis]
    else:
        # Detect valid segments (exclude padding and transitions to padding)
        # A segment is valid only if BOTH endpoints are non-zero
        starts_nonzero = np.any(seg_starts != 0, axis=2)
        ends_nonzero = np.any(seg_ends != 0, axis=2)
        valid_segs = starts_nonzero & ends_nonzero

    # Compute segment vectors and squared lengths
    seg_vecs = seg_ends - seg_starts  # (N_lanes, max_segments, 2)
    seg_lens_sq = np.einsum("ijk,ijk->ij", seg_vecs, seg_vecs)  # (N_lanes, max_segments)

    # Additional check: filter zero-length valid segments (degenerate polylines)
    valid_segs = valid_segs & (seg_lens_sq > 1e-10)
    seg_lens_sq_safe = seg_lens_sq + 1e-10  # Avoid division by zero

    # Reshape for broadcasting
    seg_starts_bc = seg_starts.reshape(1, n_lanes, max_segments, 2)
    seg_vecs_bc = seg_vecs.reshape(1, n_lanes, max_segments, 2)
    seg_lens_sq_bc = seg_lens_sq_safe.reshape(1, n_lanes, max_segments)
    valid_segs_bc = valid_segs.reshape(1, n_lanes, max_segments)
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


def _is_offroad_at_init(
    agent_data: tuple,
    static_map_elements: dict,
    lane_polylines: np.ndarray,
    lane_lengths: np.ndarray,
    route_check_timestep: int = 0,
) -> bool:
    """
    Check if vehicle is offroad at specified timestep.

    Offroad = bounding box crosses road edge OR center >5m from closest lane.

    Args:
        agent_data: Tuple of (agent_id, position, heading, valid, length, width)
        static_map_elements: Dict of static map elements (lanes, boundaries, road edges)
        lane_polylines: Precomputed lane polylines array (N_lanes, max_points, 2)
        lane_lengths: Number of valid points in each lane polyline (N_lanes,)
        route_check_timestep: Timestep to check (default: 0)

    Returns:
        True if vehicle is offroad, False otherwise
    """
    _, positions, headings, valid, lengths, widths = agent_data

    if route_check_timestep >= len(positions) or not valid[route_check_timestep]:
        return True

    position = positions[route_check_timestep, :2]  # (2,) xy
    heading = headings[route_check_timestep]
    length = lengths[route_check_timestep]
    width = widths[route_check_timestep]

    # Is stationary vehicle?
    valid_positions = positions[valid]
    if len(valid_positions) >= 2:
        displacement = np.linalg.norm(valid_positions[-1, :2] - valid_positions[0, :2])
    else:
        displacement = 0.0
    has_movement = displacement > 0.5
    OFFROAD_DISTANCE_THRESHOLD_LOCAL = OFFROAD_DISTANCE_THRESHOLD if has_movement else 1.0

    # Check 1: Distance from closest lane > threshold
    if len(lane_polylines) > 0:
        position_2d = position.reshape(1, 2)
        min_distances = _points_to_polylines_distance(
            position_2d,
            lane_polylines,
            polyline_lengths=lane_lengths,
            return_indices=False,
        )
        min_dist_to_lane = np.min(min_distances)
        if min_dist_to_lane > OFFROAD_DISTANCE_THRESHOLD_LOCAL:
            return True

    # Check 2: Bounding box crosses road edges
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_len = length / 2
    half_width = width / 2

    # Corners in local frame (front-right, front-left, rear-left, rear-right)
    local_corners = np.array(
        [
            [half_len, -half_width],
            [half_len, half_width],
            [-half_len, half_width],
            [-half_len, -half_width],
        ],
    )

    # Rotate and translate to global frame
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = local_corners @ rotation_matrix.T + position  # (4, 2)

    # Compute bounding box for filtering (min/max of corners)
    bbox_min = np.min(corners, axis=0)
    bbox_max = np.max(corners, axis=0)
    # Expand bbox slightly for edge detection
    bbox_margin = max(half_len, half_width)
    bbox_min -= bbox_margin
    bbox_max += bbox_margin

    # Check road edges with spatial filtering
    for element_id, element in static_map_elements.items():
        element_type = element["type"]
        if element_type not in ROAD_EDGE_TYPES:
            continue

        polyline = element["polyline"]
        if polyline is None or len(polyline) < 2:
            continue

        polyline_2d = polyline[:, :2]

        # Quick bbox filter: skip edges entirely outside vehicle bbox
        edge_min = np.min(polyline_2d, axis=0)
        edge_max = np.max(polyline_2d, axis=0)
        if edge_max[0] < bbox_min[0] or edge_min[0] > bbox_max[0]:
            continue
        if edge_max[1] < bbox_min[1] or edge_min[1] > bbox_max[1]:
            continue

        # Check segments within this edge
        for i in range(len(polyline_2d) - 1):
            seg_start = polyline_2d[i]
            seg_end = polyline_2d[i + 1]

            # Skip segments entirely outside bbox
            seg_min = np.minimum(seg_start, seg_end)
            seg_max = np.maximum(seg_start, seg_end)
            if seg_max[0] < bbox_min[0] or seg_min[0] > bbox_max[0]:
                continue
            if seg_max[1] < bbox_min[1] or seg_min[1] > bbox_max[1]:
                continue

            if _segment_intersects_polygon(seg_start, seg_end, corners):
                return True

    return False


def _segment_intersects_polygon(seg_start: np.ndarray, seg_end: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if line segment intersects with polygon edges.

    Args:
        seg_start: Segment start point (2,)
        seg_end: Segment end point (2,)
        polygon: Polygon vertices (N, 2)

    Returns:
        True if segment intersects polygon, False otherwise
    """
    # Check each edge of polygon
    n_vertices = len(polygon)
    for i in range(n_vertices):
        poly_start = polygon[i]
        poly_end = polygon[(i + 1) % n_vertices]

        if _segments_intersect(seg_start, seg_end, poly_start, poly_end):
            return True

    return False


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    Check if two line segments intersect using cross product method.

    Args:
        p1, p2: First segment endpoints (2,)
        p3, p4: Second segment endpoints (2,)

    Returns:
        True if segments intersect, False otherwise
    """
    d1 = p2 - p1
    d2 = p4 - p3

    # Cross product to determine orientation
    def cross_2d(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    denom = cross_2d(d1, d2)

    # Parallel segments
    if abs(denom) < 1e-10:
        # Check if collinear
        if abs(cross_2d(p3 - p1, d1)) > 1e-10 or abs(cross_2d(p4 - p1, d1)) > 1e-10:
            return False

        # Collinear overlap test using bounding boxes
        def ranges_overlap(a1, a2, b1, b2) -> bool:
            return max(min(a1, a2), min(b1, b2)) <= min(max(a1, a2), max(b1, b2)) + 1e-10

        return ranges_overlap(p1[0], p2[0], p3[0], p4[0]) and ranges_overlap(p1[1], p2[1], p3[1], p4[1])

    t = cross_2d(p3 - p1, d2) / denom
    u = cross_2d(p3 - p1, d1) / denom

    # Check if intersection point is within both segments
    return 0 <= t <= 1 and 0 <= u <= 1

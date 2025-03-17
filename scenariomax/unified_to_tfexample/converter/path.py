import numpy as np

from scenariomax.unified_to_tfexample.constants import NUM_PATHS, NUM_POINTS_PER_PATH
from scenariomax.unified_to_tfexample.converter.datatypes import MultiAgentPathSamples, PathSamples
from scenariomax.unified_to_tfexample.converter.utils import get_object_heading, get_object_trajectory


LANE_WIDTH = 2.5
YAW_THRESHOLD = 0.5


def get_scenario_paths(scenario, state, roadgraph_samples, compute_other_paths, debug=False):
    """
    Compute paths for the SDC and all valid vehicles in the scene.

    Args:
        scenario: The scenario object containing map features
        sdc_trajectory: Trajectory for the SDC
        sdc_heading: Heading for the SDC
        roadgraph_samples: Road graph samples
        debug: Whether to display debug information
        agents: Dictionary of agents with their trajectories and headings

    Returns:
        Dictionary mapping agent IDs to their MultiAgentPathSamples, with SDC having ID 'sdc'
    """
    all_paths = MultiAgentPathSamples()

    # First compute paths for the SDC (backward compatibility)
    sdc_trajectory = get_object_trajectory(state)
    sdc_heading = get_object_heading(state)
    sdc_paths = _compute_agent_paths(scenario, sdc_trajectory, sdc_heading, roadgraph_samples, debug)
    all_paths.add_data(0, sdc_paths)

    if debug:
        import matplotlib.pyplot as plt

        ax = plt.gca()

        for i in range(all_paths.valid[0].shape[0]):
            mask = all_paths.valid[0, i] == 1
            if np.any(mask):
                on_route = all_paths.on_route[0, i]
                color = "blue" if on_route else "green"
                ax.plot(all_paths.xyz[0, i, mask, 0], all_paths.xyz[0, i, mask, 1], ".", color=color, alpha=0.3)

    # If agents were provided, compute paths for each valid vehicle
    if compute_other_paths:
        for i in range(1, state.num_objects):
            if not state.tracks_to_predict[i]:
                continue

            object_trajectory = get_object_trajectory(state, i)
            object_heading = get_object_heading(state, i)

            agent_paths = _compute_agent_paths(scenario, object_trajectory, object_heading, roadgraph_samples, debug)
            all_paths.add_data(i, agent_paths)

            if debug:
                position = np.array([state.past_x[i, 0], state.past_y[i, 0]])
                length = state.past_length[i, 0]
                width = state.past_width[i, 0]
                heading = state.past_bbox_yaw[i, 0]

                ax.add_patch(
                    plt.Rectangle(
                        position - np.array([length / 2, width / 2]),
                        length,
                        width,
                        angle=np.degrees(heading),
                        rotation_point=tuple(position),
                        edgecolor="green",
                        facecolor="none",
                    ),
                )
                ax.text(position[0], position[1], str(i))  # , color=color)

                mask = all_paths.valid[i, 0] == 1
                if np.any(mask):
                    ax.plot(all_paths.xyz[i, 0, mask, 0], all_paths.xyz[i, 0, mask, 1], ".", color="skyblue", alpha=0.3)

        return all_paths
    else:
        return sdc_paths


def _compute_agent_paths(scenario, trajectory, heading, roadgraph_samples, debug=False):
    """Extract of the original get_scenario_paths function to compute paths for a single agent"""
    path_samples = PathSamples()

    # Filter for lane centers only (type 2 and 1)
    lane_mask = np.logical_or(roadgraph_samples.type == 1, roadgraph_samples.type == 2)
    lanes_ids = roadgraph_samples.id[lane_mask]
    lanes_xyz = roadgraph_samples.xyz[lane_mask]
    lanes_dir = roadgraph_samples.dir[lane_mask]

    # Calculate distance to the agent
    distances_to_agent = np.linalg.norm(lanes_xyz - trajectory[0], axis=1)

    # Find lanes that are close to the agent
    close_lanes_mask = distances_to_agent < LANE_WIDTH

    if not np.any(close_lanes_mask):
        # If no lanes are close enough, try with a larger threshold
        close_lanes_mask = distances_to_agent < (LANE_WIDTH * 2)
        if not np.any(close_lanes_mask):
            return path_samples  # No suitable lanes found

    closest_points_ids = lanes_ids[close_lanes_mask]
    closest_points_dir = lanes_dir[close_lanes_mask]

    # Get unique lane IDs with index of first appearance
    unique_indices = {}
    for i, lane_id in enumerate(closest_points_ids):
        if lane_id not in unique_indices:
            unique_indices[lane_id] = i

    # Calculate lane suitability scores based on alignment and distance
    root_ids = []
    lane_scores = {}

    for lane_id, idx in unique_indices.items():
        # Get direction vector for the lane
        lane_vector = closest_points_dir[idx, :2]
        agent_vector = np.array([np.cos(heading[0]), np.sin(heading[0])])

        # Calculate alignment score (how well the lane aligns with agent direction)
        alignment_score = np.dot(lane_vector, agent_vector)

        # Only consider lanes that generally align with the agent direction
        if alignment_score > YAW_THRESHOLD:
            # Distance score (closer is better)
            distance_score = 1.0 / (1.0 + distances_to_agent[idx])

            # Final score combines alignment and distance
            total_score = (alignment_score * 0.7) + (distance_score * 0.3)

            lane_scores[str(lane_id)] = total_score
            root_ids.append(str(lane_id))

    # Sort root_ids by score in descending order
    root_ids = sorted(root_ids, key=lambda x: lane_scores.get(x, 0), reverse=True)

    # if debug:
    #     print(f"Root IDs with scores: {[(id, lane_scores.get(id, 0)) for id in root_ids]}")

    paths = _create_sdc_tree_path(root_ids, scenario.map_features)

    # if debug:
    #     print(f"Number of paths: {len(paths)}")

    if len(paths) == 0:
        return path_samples

    xyz, valid, ids, arc_length = _gather_all_paths(paths, trajectory[0], roadgraph_samples)

    agent_lane, idx_closest_paths = _sort_paths_and_get_ego_lane(trajectory, xyz, valid)
    len_agent_path = len(agent_lane)

    # Fill the path_samples
    for i, idx in enumerate(idx_closest_paths):
        path_samples.xyz[i] = xyz[idx]
        path_samples.valid[i] = valid[idx]
        path_samples.id[i] = ids[idx]
        path_samples.arc_length[i] = arc_length[idx]

        # Check if agent_path is contained in the path
        if np.all(np.equal(path_samples.xyz[i, :len_agent_path], agent_lane)):
            path_samples.on_route[i] = 1
        else:
            path_samples.on_route[i] = 0

    return path_samples


def _sort_paths_and_get_ego_lane(sdc_trajectory, xyz, valid):
    closest_path = []

    for i, path in enumerate(xyz):
        dist = 0
        _path = path[valid[i] == 1]

        for j in range(len(sdc_trajectory)):
            dist += np.linalg.norm(_path - sdc_trajectory[j], axis=1).min()

        closest_path.append(dist)

    idx_closest_paths = np.argsort(closest_path)[:NUM_PATHS]
    ego_lane = xyz[idx_closest_paths[0]]

    # Crop to the last point of the SDC trajectory
    last_sdc_pos = sdc_trajectory[-1]
    distances = np.linalg.norm(ego_lane - last_sdc_pos, axis=1)
    idx_closest_point = np.argmin(distances)
    ego_lane = ego_lane[: idx_closest_point + 1]

    return ego_lane, idx_closest_paths


def _gather_all_paths(paths, sdc_pos, roadgraph_samples):
    xyz = np.full((len(paths), NUM_POINTS_PER_PATH, 3), -1.0)
    valid = np.full((len(paths), NUM_POINTS_PER_PATH), 0)
    ids = np.full((len(paths), NUM_POINTS_PER_PATH), -1)
    arc_length = np.full((len(paths), NUM_POINTS_PER_PATH), 0.0)

    for idx, path in enumerate(paths):
        points, points_id = _gather_points_from_path(path, roadgraph_samples)

        if len(points) == 0:
            continue

        # Path starts at the closest point to the SDC
        distances = np.linalg.norm(points - sdc_pos, axis=1)
        idx_closest_point = np.argmin(distances)
        points = points[idx_closest_point:]
        points_id = points_id[idx_closest_point:]

        # Crop to NUM_POINTS_PER_PATH
        if points.shape[0] > NUM_POINTS_PER_PATH:
            points = points[:NUM_POINTS_PER_PATH]
            points_id = points_id[:NUM_POINTS_PER_PATH]

        xyz[idx, : points.shape[0]] = points
        valid[idx, : points.shape[0]] = 1
        ids[idx, : points.shape[0]] = points_id
        arc_lengths = np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))
        arc_lengths = np.concatenate([[0], arc_lengths])
        arc_length[idx, : points.shape[0]] = arc_lengths

    return xyz, valid, ids, arc_length


def _create_sdc_tree_path(root_ids, map_features):
    paths = []
    seen_paths = set()  # To track and avoid duplicate paths

    for root_id in root_ids:
        tree_path = _make_tree_path(map_features, root_id)
        new_paths = _retrieve_path_from_tree(tree_path)

        for path in new_paths:
            # Convert path to tuple for hashing
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                paths.append(path)
                seen_paths.add(path_tuple)

    filtered_paths = []
    paths.sort(key=len, reverse=True)  # Sort by length (longest first)

    for path in paths:
        path_set = set(path)
        # Check if this path is already contained in a longer path we're keeping
        is_subset = False
        for accepted_path in filtered_paths:
            if path_set.issubset(set(accepted_path)):
                is_subset = True
                break

        if not is_subset:
            filtered_paths.append(path)

    return filtered_paths


def _gather_points_from_path(path, roadgraph_samples):
    points = []
    points_id = []

    for str_id in path:
        _id = int(str_id)

        if _id in points_id:
            break

        points_to_add = roadgraph_samples.xyz[roadgraph_samples.id == _id][1:]
        points.extend(points_to_add)
        points_id.extend([str_id] * len(points_to_add))

    points = np.array(points)
    points_id = np.array(points_id)

    if len(points) == 0:
        return points, points_id

    # Add z coordinate
    if points.shape[1] == 2:
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)

    return points, points_id


def _make_tree_path(map_features, _id, depth=0):
    if isinstance(_id, int):
        _id = str(_id)

    if _id not in map_features or depth > 10:
        return {"id": _id, "children": []}

    root_lane = map_features[_id]
    root_lane["id"] = _id

    tree = {"id": _id, "children": []}

    if "exit_lanes" in root_lane:
        for child_id in root_lane["exit_lanes"]:
            tree["children"].append(_make_tree_path(map_features, child_id, depth + 1))

    return tree


def _retrieve_path_from_tree(tree):
    """
    Retrieve all the paths from a tree structure.
    A path is a list of lanes from the root to the leaf.
    The number of paths is equal to the number of leaves in the tree.

    The result should be a list of paths, where each path is a list of lanes.
    """

    if len(tree["children"]) == 0:
        return [[tree["id"]]]

    paths = []

    for child in tree["children"]:
        child_paths = _retrieve_path_from_tree(child)

        for child_path in child_paths:
            paths.append([tree["id"]] + child_path)

    return paths

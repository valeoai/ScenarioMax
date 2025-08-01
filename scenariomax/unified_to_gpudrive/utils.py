import numpy as np
import trimesh


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def ensure_scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.size == 1 else value


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32 | np.float64):
        return float(obj)
    elif isinstance(obj, np.int32 | np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


def filter_small_segments(segments, min_length=1e-6):
    """Filter out segments that are too short."""
    valid_segments = []
    for segment in segments:
        start, end = segment
        length = np.linalg.norm(np.array(end) - np.array(start))
        if length >= min_length:
            valid_segments.append(segment)
    return valid_segments


def generate_mesh(segments, height=2.0, width=0.2):
    """Generate mesh for collision detection like GPUDrive template."""
    if not segments:
        # Return empty mesh if no segments
        return trimesh.Trimesh()

    segments = np.array(segments, dtype=np.float64)
    if len(segments) == 0:
        return trimesh.Trimesh()

    starts, ends = segments[:, 0, :], segments[:, 1, :]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)

    # Filter out zero-length segments to avoid division by zero
    valid_mask = lengths.flatten() > 1e-10
    if not np.any(valid_mask):
        return trimesh.Trimesh()

    starts = starts[valid_mask]
    ends = ends[valid_mask]
    directions = directions[valid_mask]
    lengths = lengths[valid_mask]

    unit_directions = directions / lengths

    # Create the base box mesh with the height along the z-axis
    base_box = trimesh.creation.box(extents=[1.0, width, height])
    base_box.apply_translation([0.5, 0, 0])  # Align box's origin to its start
    z_axis = np.array([0, 0, 1])
    angles = np.arctan2(unit_directions[:, 1], unit_directions[:, 0])  # Rotation in the XY plane

    rectangles = []
    lengths = lengths.flatten()

    for i, (start, length, angle) in enumerate(zip(starts, lengths, angles)):
        # Copy the base box and scale to match segment length
        scaled_box = base_box.copy()
        scaled_box.apply_scale([length, 1.0, 1.0])

        # Apply rotation around the z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, z_axis)
        scaled_box.apply_transform(rotation_matrix)

        # Translate the box to the segment's starting point
        scaled_box.apply_translation(start)

        rectangles.append(scaled_box)

    if not rectangles:
        return trimesh.Trimesh()

    # Concatenate all boxes into a single mesh
    mesh = trimesh.util.concatenate(rectangles)
    return mesh


def mark_colliding_agents(objects, agent_collision_manager, road_collision_manager, trajectory_collision_manager):
    # Check collisions between all init agent positions
    _, agent_collision_pairs = agent_collision_manager.in_collision_internal(return_names=True)

    # Check collisions between init agent positions and road edges
    _, road_collision_pairs = agent_collision_manager.in_collision_other(road_collision_manager, return_names=True)

    # Check trajectory collisions with road edges
    _, trajectory_collision_pairs = trajectory_collision_manager.in_collision_other(
        road_collision_manager,
        return_names=True,
    )

    # Create sets of colliding agent IDs
    colliding_agents = set()

    # Add agents that collide with each other at first step
    for agent1, agent2 in agent_collision_pairs:
        colliding_agents.add(agent1)
        colliding_agents.add(agent2)

    # Add agents that collide with road edges
    road_colliding_agents = {agent_id for agent_id, _ in road_collision_pairs}
    colliding_agents.update(road_colliding_agents)

    # Add agents whose trajectories collide with road edges
    trajectory_colliding_agents = {agent_id for agent_id, _ in trajectory_collision_pairs}
    colliding_agents.update(trajectory_colliding_agents)

    # Update mark_as_expert based on initial collisions
    for index, obj in enumerate(objects):
        if obj["type"] in ["vehicle", "cyclist"]:
            if str(obj["id"]) in colliding_agents:
                objects[index]["mark_as_expert"] = True
            else:
                objects[index]["mark_as_expert"] = False

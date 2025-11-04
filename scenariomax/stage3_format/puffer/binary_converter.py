"""
Convert Puffer dict format to binary format matching datatypes.h structure.

This module provides in-memory conversion from Puffer dict (as returned by
convert_to_puffer.py) to compact binary format for C/GPU simulators.

Binary Format Structure:
========================

Header (12 bytes):
    - num_agents (int32)
    - num_road_elements (int32)
    - num_traffic_controls (int32)

For each DynamicAgent:
    - id (int32)
    - type (int32)
    - trajectory_length (int32)
    - Arrays (transposed for cache efficiency):
        - log_trajectory_x[trajectory_length] (float32[])
        - log_trajectory_y[trajectory_length] (float32[])
        - log_trajectory_z[trajectory_length] (float32[])
        - log_heading[trajectory_length] (float32[])
        - log_velocity_x[trajectory_length] (float32[])
        - log_velocity_y[trajectory_length] (float32[])
        - length[trajectory_length] (float32[])
        - width[trajectory_length] (float32[])
        - height[trajectory_length] (float32[])
        - log_valid[trajectory_length] (int32[])
    - num_route_ints (int32) - total number of route integers
    - routes (int32[]) - flattened: [route1_len, id1, id2, ..., route2_len, id1, ...]
    - goal_position_x, y, z (float32, float32, float32)
    - mark_as_expert (int32)

For each RoadMapElement:
    - id (int32)
    - type (int32)
    - segment_length (int32)
    - Arrays (transposed):
        - x[segment_length] (float32[])
        - y[segment_length] (float32[])
        - z[segment_length] (float32[])
        - dir_x[segment_length] (float32[])
        - dir_y[segment_length] (float32[])
        - dir_z[segment_length] (float32[])
    - entry (int32) - first entry lane or -1
    - exit (int32) - first exit lane or -1
    - speed_limit (float32) - in m/s

For each TrafficControlElement:
    - id (int32)
    - type (int32)
    - state_length (int32)
    - x, y, z (float32, float32, float32)
    - states[state_length] (int32[])
    - controlled_lane (int32)

Metadata (200 bytes):
    - scenario_id (char[128]) - null-padded UTF-8 string
    - dataset_name (char[64]) - null-padded UTF-8 string
    - length (int32) - number of timesteps
    - ego_id (int32) - ego vehicle ID
"""

import struct

import numpy as np


def puffer_dict_to_binary(puffer_dict: dict) -> bytes:
    """
    Convert Puffer dict to binary format matching datatypes.h.

    Args:
        puffer_dict: Puffer scenario dict with keys:
            - dynamic_agents: list of agent dicts
            - road_map_elements: list of road dicts
            - traffic_control_elements: list of traffic dicts

    Returns:
        Binary data as bytes
    """
    # Extract data
    dynamic_agents = puffer_dict.get('dynamic_agents', [])
    road_map_elements = puffer_dict.get('road_map_elements', [])
    traffic_control_elements = puffer_dict.get('traffic_control_elements', [])

    num_agents = len(dynamic_agents)
    num_roads = len(road_map_elements)
    num_traffic = len(traffic_control_elements)

    # Build binary in memory
    buffer = bytearray()

    # Write header
    buffer.extend(struct.pack('iii', num_agents, num_roads, num_traffic))

    # ========================================================================
    # Write DynamicAgents
    # ========================================================================
    for agent in dynamic_agents:
        # ID and type
        agent_id = int(agent.get('id', 0))
        agent_type = int(agent.get('type', 1))
        buffer.extend(struct.pack('ii', agent_id, agent_type))

        # Get states
        states = agent.get('states', {})
        xyz = np.array(states.get('xyz', []))
        velocity = np.array(states.get('velocity', []))
        heading = np.array(states.get('heading', []))
        valid = np.array(states.get('valid', []))
        width = np.array(states.get('width', []))
        length = np.array(states.get('length', []))
        height = np.array(states.get('height', []))

        trajectory_length = len(xyz)
        buffer.extend(struct.pack('i', trajectory_length))

        # Write trajectory arrays - TRANSPOSED: all X, then Y, then Z
        for i in range(3):  # x, y, z
            for j in range(trajectory_length):
                buffer.extend(struct.pack('f', float(xyz[j, i])))

        # Write heading array
        for j in range(trajectory_length):
            buffer.extend(struct.pack('f', float(heading[j])))

        # Write velocity arrays - TRANSPOSED
        for i in range(2):  # x, y
            for j in range(trajectory_length):
                buffer.extend(struct.pack('f', float(velocity[j, i])))

        # Write dimension arrays
        # Handle both scalar and array formats
        if isinstance(length, (int, float)):
            length = np.full(trajectory_length, float(length))
        if isinstance(width, (int, float)):
            width = np.full(trajectory_length, float(width))
        if isinstance(height, (int, float)):
            height = np.full(trajectory_length, float(height))

        for j in range(trajectory_length):
            buffer.extend(struct.pack('f', float(length[j]) if j < len(length) else 0.0))
        for j in range(trajectory_length):
            buffer.extend(struct.pack('f', float(width[j]) if j < len(width) else 0.0))
        for j in range(trajectory_length):
            buffer.extend(struct.pack('f', float(height[j]) if j < len(height) else 0.0))

        # Write valid array
        for j in range(trajectory_length):
            buffer.extend(struct.pack('i', int(valid[j])))

        # Write routes (flatten all routes into single array)
        routes = agent.get('routes', [])
        if routes:
            # Flatten routes: [[1,2,3], [4,5]] -> [3, 1,2,3, 2, 4,5]
            # Format: for each route, write length then route IDs
            flattened = []
            for route in routes:
                flattened.append(len(route))  # route length
                flattened.extend(route)       # route IDs

            total_route_ints = len(flattened)
        else:
            total_route_ints = 0
            flattened = []

        buffer.extend(struct.pack('i', total_route_ints))  # total number of ints
        for route_int in flattened:
            buffer.extend(struct.pack('i', int(route_int)))

        # Calculate goal position from last valid position
        goal_x, goal_y, goal_z = 0.0, 0.0, 0.0
        if len(valid) > 0:
            valid_indices = np.where(valid > 0)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                goal_x = float(xyz[last_valid_idx, 0])
                goal_y = float(xyz[last_valid_idx, 1])
                goal_z = float(xyz[last_valid_idx, 2])

        buffer.extend(struct.pack('fff', goal_x, goal_y, goal_z))

        # Write mark_as_expert: 1 if routes defined, 0 otherwise
        mark_as_expert = 0 if (routes and len(routes) > 0) else 1
        buffer.extend(struct.pack('i', mark_as_expert))

    # ========================================================================
    # Write RoadMapElements
    # ========================================================================
    for road in road_map_elements:
        # ID and type
        road_id = int(road.get('id', 0))
        road_type = int(road.get('type', 0))
        buffer.extend(struct.pack('ii', road_id, road_type))

        # Get geometry
        xyz = np.array(road.get('xyz', []))
        dir_xyz = np.array(road.get('dir_xyz', []))

        segment_length = len(xyz)
        buffer.extend(struct.pack('i', segment_length))

        # Write geometry arrays - TRANSPOSED
        for i in range(3):  # x, y, z
            for j in range(segment_length):
                buffer.extend(struct.pack('f', float(xyz[j, i])))

        # Write direction arrays - TRANSPOSED
        for i in range(3):  # x, y, z
            for j in range(segment_length):
                buffer.extend(struct.pack('f', float(dir_xyz[j, i])))

        # Entry and exit (take first element or -1)
        entry_lanes = road.get('entry', [])
        exit_lanes = road.get('exit', [])

        entry = int(entry_lanes[0]) if entry_lanes and len(entry_lanes) > 0 else -1
        exit_val = int(exit_lanes[0]) if exit_lanes and len(exit_lanes) > 0 else -1

        buffer.extend(struct.pack('ii', entry, exit_val))

        # Speed limit (convert from mph to m/s if available)
        speed_limit = road.get('speed_limit', 0.0)
        buffer.extend(struct.pack('f', speed_limit))

    # ========================================================================
    # Write TrafficControlElements
    # ========================================================================
    for traffic in traffic_control_elements:
        # ID and type
        traffic_id = int(traffic.get('id', 0))
        traffic_type = int(traffic.get('type', 11))
        buffer.extend(struct.pack('ii', traffic_id, traffic_type))

        # State length
        states = traffic.get('states', [])
        state_length = len(states)
        buffer.extend(struct.pack('i', state_length))

        # Position
        xyz = traffic.get('xyz', [0.0, 0.0, 0.0])
        if isinstance(xyz, list):
            xyz = np.array(xyz)

        x = float(xyz[0]) if len(xyz) > 0 else 0.0
        y = float(xyz[1]) if len(xyz) > 1 else 0.0
        z = float(xyz[2]) if len(xyz) > 2 else 0.0

        buffer.extend(struct.pack('fff', x, y, z))

        # States array
        for state in states:
            buffer.extend(struct.pack('i', int(state)))

        # Controlled lane
        controlled_lane = int(traffic.get('controlled_lane', -1))
        buffer.extend(struct.pack('i', controlled_lane))

    # ========================================================================
    # Write Metadata
    # ========================================================================
    metadata = puffer_dict.get('metadata', {})

    # scenario_id (fixed 128 bytes, null-padded)
    scenario_id = puffer_dict.get('scenario_id', '')[:128]
    scenario_id_bytes = scenario_id.encode('utf-8').ljust(128, b'\0')
    buffer.extend(scenario_id_bytes)

    # dataset_name (fixed 64 bytes, null-padded)
    dataset_name = metadata.get('dataset_name', '')[:64]
    dataset_name_bytes = dataset_name.encode('utf-8').ljust(64, b'\0')
    buffer.extend(dataset_name_bytes)

    # length (int - number of timesteps)
    length = int(metadata.get('length', 0))
    buffer.extend(struct.pack('i', length))

    # ego_id (int)
    ego_id = int(metadata.get('ego_id', -1))
    buffer.extend(struct.pack('i', ego_id))

    return bytes(buffer)

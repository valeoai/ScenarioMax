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
    - Arrays:
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
    - routes_length (int32) - total number of route integers
    - routes (int32[]) - flattened: [id1, id2, ...] # Only first route for now
    - goal_position_x, y, z (float32, float32, float32)
    - mark_as_expert (int32) - 1 if no routes (expert/uncontrollable), 0 if routes exist (controllable)

For each RoadMapElement:
    - id (int32)
    - type (int32)
    - segment_length (int32)
    - Arrays (transposed):
        - x[segment_length] (float32[])
        - y[segment_length] (float32[])
        - z[segment_length] (float32[])
    - num_entry (int32) - number of entry lanes
    - entry_lanes (int32[]) - list of entry lane IDs
    - num_exit (int32) - number of exit lanes
    - exit_lanes (int32) - list of exit lane IDs
    - speed_limit (float32) - in m/s

For each TrafficControlElement:
    - id (int32)
    - type (int32)
    - x, y, z (float32, float32, float32)
    - state_length (int32)
    - states[state_length] (int32[])
    - num_controlled_lanes (int32)
    - controlled_lanes[num_controlled_lanes] (int32[])

Metadata (variable length):
    - scenario_id (char[128]) - null-padded UTF-8 string
    - map_id (int32) - map identifier assigned during postprocessing
    - dataset_name (char[64]) - null-padded UTF-8 string
    - length (int32) - number of timesteps
    - sdc_index (int32) - ego vehicle ID
    - num_objects_of_interest (int32)
    - objects_of_interest[num_objects_of_interest] (int32[])
    - num_tracks_to_predict (int32)
    - tracks_to_predict[num_tracks_to_predict] (int32[])
"""

import struct

import numpy as np


def puffer_dict_to_binary(puffer_dict: dict, map_id: int = 0) -> bytes:  # noqa: C901
    """
    Convert Puffer dict to binary format matching datatypes.h.

    Args:
        puffer_dict: Puffer scenario dict with keys:
            - dynamic_agents: list of agent dicts
            - road_map_elements: list of road dicts
            - traffic_control_elements: list of traffic dicts
        map_id: Map identifier (e.g., 0, 1, 125786) assigned during postprocessing

    Returns:
        Binary data as bytes
    """
    # Extract data
    dynamic_agents = puffer_dict["dynamic_agents"]
    road_map_elements = puffer_dict["road_map_elements"]
    traffic_control_elements = puffer_dict["traffic_control_elements"]

    num_agents = len(dynamic_agents)
    num_roads = len(road_map_elements)
    num_traffic = len(traffic_control_elements)

    # Build binary in memory
    buffer = bytearray()

    # Write header
    buffer.extend(struct.pack("iii", num_agents, num_roads, num_traffic))

    # ========================================================================
    # Write DynamicAgents
    # ========================================================================
    for agent in dynamic_agents:
        # ID and type
        agent_id = int(agent["id"])
        agent_type = int(agent["type"])
        buffer.extend(struct.pack("ii", agent_id, agent_type))

        # Get states
        states = agent["states"]
        xyz = np.array(states["xyz"])
        velocity = np.array(states["velocity"])
        heading = np.array(states["heading"])
        valid = np.array(states["valid"])
        width = np.array(states["width"])
        length = np.array(states["length"])
        height = np.array(states["height"])

        trajectory_length = len(xyz)
        buffer.extend(struct.pack("i", trajectory_length))

        # Write trajectory arrays - TRANSPOSED: all X, then Y, then Z
        for i in range(3):  # x, y, z
            for j in range(trajectory_length):
                buffer.extend(struct.pack("f", float(xyz[j, i])))

        # Write heading array
        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(heading[j])))

        # Write velocity arrays - TRANSPOSED
        for i in range(2):  # x, y
            for j in range(trajectory_length):
                buffer.extend(struct.pack("f", float(velocity[j, i])))

        # Write dimension arrays
        # Handle both scalar and array formats
        if isinstance(length, (int, float)):
            length = np.full(trajectory_length, float(length))
        if isinstance(width, (int, float)):
            width = np.full(trajectory_length, float(width))
        if isinstance(height, (int, float)):
            height = np.full(trajectory_length, float(height))

        # Validate array lengths match trajectory_length
        if len(length) != trajectory_length:
            raise ValueError(
                f"Agent {agent_id}: length array has {len(length)} elements "
                f"but trajectory has {trajectory_length} timesteps"
            )
        if len(width) != trajectory_length:
            raise ValueError(
                f"Agent {agent_id}: width array has {len(width)} elements "
                f"but trajectory has {trajectory_length} timesteps"
            )
        if len(height) != trajectory_length:
            raise ValueError(
                f"Agent {agent_id}: height array has {len(height)} elements "
                f"but trajectory has {trajectory_length} timesteps"
            )

        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(length[j])))
        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(width[j])))
        for j in range(trajectory_length):
            buffer.extend(struct.pack("f", float(height[j])))

        # Write valid array
        for j in range(trajectory_length):
            buffer.extend(struct.pack("i", int(valid[j])))

        # Write routes (flatten all routes into single array)
        routes = agent["routes"]
        if routes:
            # Option 1: First route only
            first_route = routes[0]
            flattened = first_route
            total_route_ints = len(first_route)

            # Option 2: All route IDs flattened
            # # Flatten routes: [[1,2,3], [4,5]] -> [3, 1,2,3, 2, 4,5]
            # # Format: for each route, write length then route IDs
            # flattened = []
            # for route in routes:
            #     flattened.append(len(route))  # route length
            #     flattened.extend(route)  # route IDs

            # total_route_ints = len(flattened)
        else:
            total_route_ints = 0
            flattened = []

        buffer.extend(struct.pack("i", total_route_ints))  # total number of ints
        for route_int in flattened:
            buffer.extend(struct.pack("i", int(route_int)))

        # Calculate goal position from last valid position
        goal_x, goal_y, goal_z = 0.0, 0.0, 0.0
        if len(valid) > 0:
            valid_indices = np.where(valid > 0)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                goal_x = float(xyz[last_valid_idx, 0])
                goal_y = float(xyz[last_valid_idx, 1])
                goal_z = float(xyz[last_valid_idx, 2])

        buffer.extend(struct.pack("fff", goal_x, goal_y, goal_z))

        # Write mark_as_expert: 1 if NO routes (expert/uncontrollable), 0 if routes exist (controllable)
        mark_as_expert = 0 if (routes and len(routes) > 0) else 1
        buffer.extend(struct.pack("i", mark_as_expert))

    # ========================================================================
    # Write RoadMapElements
    # ========================================================================
    for road in road_map_elements:
        # ID and type
        road_id = int(road["id"])
        road_type = int(road["type"])
        buffer.extend(struct.pack("ii", road_id, road_type))

        # Get geometry
        xyz = np.array(road["xyz"])

        segment_length = len(xyz)
        buffer.extend(struct.pack("i", segment_length))

        # Write geometry arrays - TRANSPOSED
        for i in range(3):  # x, y, z
            for j in range(segment_length):
                buffer.extend(struct.pack("f", float(xyz[j, i])))

        # If lane type (0-10 range), write entry/exit lanes and speed limit
        # Lane types: 0=unknown, 1=freeway, 2=surface_street, 3=bike_lane
        if road_type <= 10:  # All lane types are in range 0-10
            entry_lanes = road["entry_lanes"]
            exit_lanes = road["exit_lanes"]

            num_entry = len(entry_lanes)
            num_exit = len(exit_lanes)

            buffer.extend(struct.pack("i", num_entry))
            for lane_id in entry_lanes:
                buffer.extend(struct.pack("i", int(lane_id)))

            buffer.extend(struct.pack("i", num_exit))
            for lane_id in exit_lanes:
                buffer.extend(struct.pack("i", int(lane_id)))

            # Speed limit (convert from mph to m/s if available)
            speed_limit = road["speed_limit"]
            buffer.extend(struct.pack("f", speed_limit))

    # ========================================================================
    # Write TrafficControlElements
    # ========================================================================
    for element in traffic_control_elements:
        # ID and type
        traffic_id = int(element["id"])
        traffic_type = int(element["type"])
        buffer.extend(struct.pack("ii", traffic_id, traffic_type))

        # Position
        xyz = element["xyz"]
        if isinstance(xyz, list):
            xyz = np.array(xyz)

        x = float(xyz[0]) if len(xyz) > 0 else 0.0
        y = float(xyz[1]) if len(xyz) > 1 else 0.0
        z = float(xyz[2]) if len(xyz) > 2 else 0.0

        buffer.extend(struct.pack("fff", x, y, z))

        # States
        states = element["states"]
        state_length = len(states)
        buffer.extend(struct.pack("i", state_length))
        for state in states:
            buffer.extend(struct.pack("i", int(state)))

        # Controlled lanes
        controlled_lanes = element["controlled_lanes"]
        controlled_lanes_length = len(controlled_lanes)
        buffer.extend(struct.pack("i", controlled_lanes_length))
        for lane in controlled_lanes:
            buffer.extend(struct.pack("i", int(lane)))

    # ========================================================================
    # Write Metadata
    # ========================================================================
    metadata = puffer_dict["metadata"]

    # scenario_id (fixed 128 bytes, null-padded)
    scenario_id = puffer_dict["scenario_id"][:128]
    scenario_id_bytes = scenario_id.encode("utf-8").ljust(128, b"\0")
    buffer.extend(scenario_id_bytes)

    # map_id (int32)
    buffer.extend(struct.pack("i", int(map_id)))

    # dataset_name (fixed 64 bytes, null-padded)
    dataset_name = metadata["dataset_name"][:64]
    dataset_name_bytes = dataset_name.encode("utf-8").ljust(64, b"\0")
    buffer.extend(dataset_name_bytes)

    # length (int - number of timesteps)
    length = int(metadata["scenario_length"])
    buffer.extend(struct.pack("i", length))

    # sdc_index (int)
    sdc_index = int(metadata["sdc_index"])
    buffer.extend(struct.pack("i", sdc_index))

    # objects_of_interest
    objects_of_interest = metadata["objects_of_interests"]
    num_oi = len(objects_of_interest)
    buffer.extend(struct.pack("i", num_oi))
    for oi in objects_of_interest:
        buffer.extend(struct.pack("i", int(oi)))

    # tracks_to_predict
    tracks_to_predict = metadata["tracks_to_predict"]
    num_ttp = len(tracks_to_predict)
    buffer.extend(struct.pack("i", num_ttp))
    for ttp in tracks_to_predict:
        buffer.extend(struct.pack("i", int(ttp)))

    return bytes(buffer)

import numpy as np

from scenariomax.unified_to_tfexample import constants, exceptions
from scenariomax.unified_to_tfexample.converter import datatypes


def get_distance(sdc_track, other_track):
    """
    Calculate the distance at the first and last time step between the SDC and other tracks.
    """
    first_distance = np.linalg.norm(sdc_track["states"]["position"][0] - other_track["states"]["position"][0])
    last_distance = np.linalg.norm(sdc_track["states"]["position"][-1] - other_track["states"]["position"][-1])

    return (first_distance + last_distance) / 2.0


def get_state(scenario, multiagent, roadgraph_samples, debug):
    state = datatypes.State()
    sdc_id = scenario.metadata["ego_id"]

    swing_index = constants.NUM_TS_PAST
    scenario_length = constants.NUM_TS_ALL

    # Change dict order by moving SDC to the first key
    scenario.dynamic_agents = {sdc_id: scenario.dynamic_agents.pop(sdc_id), **scenario.dynamic_agents}

    # Sort the tracks by the average distance to the SDC at every 10 steps
    scenario.dynamic_agents = dict(
        sorted(
            scenario.dynamic_agents.items(),
            key=lambda item: get_distance(scenario.dynamic_agents[sdc_id], item[1]),
        ),
    )

    state.is_sdc[0] = 1

    if debug:
        import matplotlib.pyplot as plt

        ax = plt.gca()

        # Plot the agents
        for i, object_id in enumerate(scenario.dynamic_agents):
            color = "red" if i == 0 else "blue"

            if np.all(scenario.dynamic_agents[object_id]["states"]["velocity"] == 0.0):
                color = "pink"

            # if object_id in scenario.metadata["tracks_to_predict"]:
            #     color = "orange"

            # if object_id in scenario.metadata["objects_of_interest"]:
            #     color = "green"

            idx_valid = np.argmax(scenario.dynamic_agents[object_id]["states"]["valid"])

            position = scenario.dynamic_agents[object_id]["states"]["position"][idx_valid][:2]
            heading = scenario.dynamic_agents[object_id]["states"]["heading"][idx_valid]
            length = scenario.dynamic_agents[object_id]["states"]["length"][idx_valid].squeeze()
            width = scenario.dynamic_agents[object_id]["states"]["width"][idx_valid].squeeze()

            ax.add_patch(
                plt.Rectangle(
                    position - np.array([length / 2, width / 2]),
                    length,
                    width,
                    angle=np.degrees(heading),
                    rotation_point=tuple(position),
                    edgecolor=color,
                    facecolor="none",
                ),
            )
            ax.text(position[0], position[1], str(i), color=color)

    for i, object_id in enumerate(scenario.dynamic_agents):
        if i > constants.DEFAULT_NUM_OBJECTS - 1:
            break

        object = scenario.dynamic_agents[object_id]

        object_type = object["type"]
        waymo_type = _type_to_int(object_type)

        state.type[i] = waymo_type
        state.id[i] = i

        state_positions = object["states"]["position"][:scenario_length]
        state_headings = object["states"]["heading"][:scenario_length]

        state.past_x[i] = state_positions[:swing_index, :1].flatten()
        state.past_y[i] = state_positions[:swing_index, 1:2].flatten()
        state.past_z[i] = state_positions[:swing_index, 2:3].flatten()
        state.past_bbox_yaw[i] = state_headings[:swing_index].flatten()

        state.current_x[i] = state_positions[swing_index, :1]
        state.current_y[i] = state_positions[swing_index, 1:2]
        state.current_z[i] = state_positions[swing_index, 2:3]
        state.current_bbox_yaw[i] = state_headings[swing_index]

        state.future_x[i] = state_positions[swing_index + 1 :, :1].flatten()
        state.future_y[i] = state_positions[swing_index + 1 :, 1:2].flatten()
        state.future_z[i] = state_positions[swing_index + 1 :, 2:3].flatten()
        state.future_bbox_yaw[i][:scenario_length] = state_headings[swing_index + 1 :].flatten()

        # Vehicle dimensions length, width, height
        default_length, default_width, default_height = _get_default_vehicle_dimensions(object_type)

        if "length" in object["states"]:
            state_length = object["states"]["length"][:scenario_length]
        else:
            state_length = np.full((scenario_length,), default_length, dtype=np.float64)

        state.past_length[i] = state_length[:swing_index].flatten()
        state.current_length[i] = state_length[swing_index]
        state.future_length[i] = state_length[swing_index + 1 :].flatten()

        if "width" in object["states"]:
            state_width = object["states"]["width"][:scenario_length]
        else:
            state_width = np.full((scenario_length,), default_width, dtype=np.float64)

        state.past_width[i] = state_width[:swing_index].flatten()
        state.current_width[i] = state_width[swing_index]
        state.future_width[i] = state_width[swing_index + 1 :].flatten()

        if "height" in object["states"]:
            state_height = object["states"]["height"][:scenario_length]
        else:
            state_height = np.full((scenario_length,), default_height, dtype=np.float64)

        state.past_height[i] = state_height[:swing_index].flatten()
        state.current_height[i] = state_height[swing_index].flatten()
        state.future_height[i] = state_height[swing_index + 1 :].flatten()

        # Velocities
        state_velocity = object["states"]["velocity"][:scenario_length]

        state.past_velocity_x[i] = state_velocity[:swing_index, :1].flatten()
        state.past_velocity_y[i] = state_velocity[:swing_index, 1:2].flatten()
        state.current_velocity_x[i] = state_velocity[swing_index, :1]
        state.current_velocity_y[i] = state_velocity[swing_index, 1:2]
        state.future_velocity_x[i][:scenario_length] = state_velocity[swing_index + 1 :, :1].flatten()
        state.future_velocity_y[i][:scenario_length] = state_velocity[swing_index + 1 :, 1:2].flatten()

        state_valids = object["states"]["valid"][:scenario_length]
        state.past_valid[i] = state_valids[:swing_index].astype(np.int64).flatten()
        state.current_valid[i] = state_valids[swing_index].astype(np.int64)
        state.future_valid[i][:scenario_length] = state_valids[swing_index + 1 :].astype(np.int64).flatten()

        # Interests
        if np.sum(state_valids) >= constants.NUM_TS_ALL * 0.8 and object_type == "VEHICLE":
            roadgraph_xy = roadgraph_samples.xyz[..., :2]
            lane_points = (roadgraph_samples.type == 2) | (roadgraph_samples.type == 1)
            lane_points = np.expand_dims(lane_points, axis=1)
            roadgraph_xy = np.where(lane_points, roadgraph_xy, np.inf)

            has_moved = np.linalg.norm(state_positions[0][:2] - state_positions[-1][:2]) > 2
            is_on_lane = np.any(np.linalg.norm(roadgraph_xy - state_positions[-1][:2], axis=1) < 2.0)
            if has_moved and is_on_lane:
                state.tracks_to_predict[i] = 1

    if not multiagent:
        return state

    if np.sum(state.tracks_to_predict) < 4:
        raise exceptions.NotEnoughValidObjectsException()

    # Sort objects to put those with tracks_to_predict=1 first (after SDC)
    if np.sum(state.tracks_to_predict) > 0:
        # Get the indices of tracked and untracked objects (excluding SDC at 0)
        tracked = [
            i
            for i in range(1, min(constants.DEFAULT_NUM_OBJECTS, len(scenario.dynamic_agents)))
            if state.tracks_to_predict[i] == 1
        ]
        untracked = [
            i
            for i in range(1, min(constants.DEFAULT_NUM_OBJECTS, len(scenario.dynamic_agents)))
            if state.tracks_to_predict[i] == 0
        ]

        # Create the new ordering: SDC at 0, then tracked objects, then untracked ones
        new_order = [0] + tracked + untracked

        # For every array in the state object, apply the reordering
        for attr_name in vars(state):
            attr = getattr(state, attr_name)
            if isinstance(attr, np.ndarray) and attr.shape[0] > 1:
                # Create a copy of the attribute array
                temp = attr.copy()

                # Reapply the values according to the new order
                for new_idx, old_idx in enumerate(new_order):
                    if old_idx < attr.shape[0] and new_idx < attr.shape[0]:
                        attr[new_idx] = temp[old_idx]

    return state


def _type_to_int(type_str):
    mapping = {
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "CYCLIST": 3,
    }
    return mapping.get(type_str, 4)


def _get_default_vehicle_dimensions(type_str):
    default_length = 4.6
    default_width = 1.8
    default_height = 1.8

    if type_str == "TRAFFIC_BARRIER":
        default_length = 1
        default_width = 0.2
        default_height = 1
    elif type_str == "TRAFFIC_CONE":
        default_length = 0.5
        default_width = 0.3
        default_height = 0.5
    elif type_str == "VEHICLE":
        default_length = 4.6
        default_width = 1.8
        default_height = 1.8
    elif type_str == "PEDESTRIAN":
        default_length = 0.5
        default_width = 0.8
        default_height = 1.8
    elif type_str == "CYCLIST":
        default_length = 1.5
        default_width = 0.5
        default_height = 1.2

    return default_length, default_width, default_height

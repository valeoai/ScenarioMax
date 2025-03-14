import numpy as np

from scenariomax.unified_to_tfrecord.constants import DEFAULT_NUM_OBJECTS, NUM_TS_ALL, NUM_TS_PAST
from scenariomax.unified_to_tfrecord.converter.datatypes import State


def get_state(scenario, debug):
    state = State()
    sdc_id = scenario.metadata["sdc_id"]

    swing_index = NUM_TS_PAST
    scenario_length = NUM_TS_ALL

    # Change dict order by moving SDC to the first key
    scenario.tracks = {sdc_id: scenario.tracks.pop(sdc_id), **scenario.tracks}
    scenario.metadata["object_summary"] = {
        sdc_id: scenario.metadata["object_summary"].pop(sdc_id),
        **scenario.metadata["object_summary"],
    }

    # Sort the tracks by the distance to the SDC
    scenario.tracks = dict(
        sorted(
            scenario.tracks.items(),
            key=lambda item: np.linalg.norm(
                scenario.tracks[sdc_id]["state"]["position"][swing_index + 10]
                - item[1]["state"]["position"][swing_index + 10]
            ),
        )
    )

    state.is_sdc[0] = 1

    # if debug:
    #     ax = plt.gca()

    #     # Plot the agents
    #     for i, object_id in enumerate(scenario.tracks):
    #         color = "red" if i == 0 else "blue"

    #         if np.all(scenario.tracks[object_id]["state"]["velocity"] == 0.0):
    #             color = "pink"

    #         for j in range(scenario_length):
    #             position = scenario.tracks[object_id]["state"]["position"][j]
    #             if position[0] > 0 and position[1] > 0:
    #                 ax.scatter(position[0], position[1], s=10, marker="o", color=color)

    for i, object_id in enumerate(scenario.tracks):
        if i > DEFAULT_NUM_OBJECTS - 1:
            break

        object = scenario.tracks[object_id]

        object_type = scenario.metadata["object_summary"][object_id]["type"]
        waymo_type = _type_to_int(object_type)

        state.type[i] = waymo_type
        state.id[i] = i

        state_positions = object["state"]["position"][:scenario_length]
        state_headings = object["state"]["heading"][:scenario_length]

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

        if "length" in object["state"]:
            state_length = object["state"]["length"][:scenario_length]
        else:
            state_length = np.full((scenario_length,), default_length, dtype=np.float64)

        state.past_length[i] = state_length[:swing_index].flatten()
        state.current_length[i] = state_length[swing_index]
        state.future_length[i] = state_length[swing_index + 1 :].flatten()

        if "width" in object["state"]:
            state_width = object["state"]["width"][:scenario_length]
        else:
            state_width = np.full((scenario_length,), default_width, dtype=np.float64)

        state.past_width[i] = state_width[:swing_index].flatten()
        state.current_width[i] = state_width[swing_index]
        state.future_width[i] = state_width[swing_index + 1 :].flatten()

        if "height" in object["state"]:
            state_height = object["state"]["height"][:scenario_length]
        else:
            state_height = np.full((scenario_length,), default_height, dtype=np.float64)

        state.past_height[i] = state_height[:swing_index].flatten()
        state.current_height[i] = state_height[swing_index].flatten()
        state.future_height[i] = state_height[swing_index + 1 :].flatten()

        # Velocities
        state_velocity = object["state"]["velocity"][:scenario_length]

        state.past_velocity_x[i] = state_velocity[:swing_index, :1].flatten()
        state.past_velocity_y[i] = state_velocity[:swing_index, 1:2].flatten()
        state.current_velocity_x[i] = state_velocity[swing_index, :1]
        state.current_velocity_y[i] = state_velocity[swing_index, 1:2]
        state.future_velocity_x[i][:scenario_length] = state_velocity[swing_index + 1 :, :1].flatten()
        state.future_velocity_y[i][:scenario_length] = state_velocity[swing_index + 1 :, 1:2].flatten()

        state_valids = object["state"]["valid"][:scenario_length]
        state.past_valid[i] = state_valids[:swing_index].astype(np.int64).flatten()
        state.current_valid[i] = state_valids[swing_index].astype(np.int64)
        state.future_valid[i][:scenario_length] = state_valids[swing_index + 1 :].astype(np.int64).flatten()

    return state


def _type_to_int(type_str):
    if type_str == "VEHICLE":
        return 1
    elif type_str == "PEDESTRIAN":
        return 2
    elif type_str == "CYCLIST":
        return 3
    else:
        return 4


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

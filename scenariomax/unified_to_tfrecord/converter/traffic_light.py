import numpy as np

from scenariomax.unified_to_tfrecord.constants import DEFAULT_NUM_LIGHT_POSITIONS, NUM_TS_ALL, NUM_TS_PAST
from scenariomax.unified_to_tfrecord.converter.datatypes import TrafficLightState


TRAFFIC_LIGHT_OFFSET = 25


def from_traffic_light_state_to_int(traffic_light_state):
    mapping = {
        "TRAFFIC_LIGHT_RED": 4,
        "LANE_STATE_ARROW_STOP": 1,
        "LANE_STATE_STOP": 4,
        "LANE_STATE_FLASHING_STOP": 7,
        "LANE_STATE_CAUTION": 5,
        "LANE_STATE_ARROW_CAUTION": 2,
        "LANE_STATE_FLASHING_CAUTION": 8,
        "TRAFFIC_LIGHT_GREEN": 6,
        "LANE_STATE_GO": 6,
        "LANE_STATE_ARROW_GO": 3,
        "TRAFFIC_LIGHT_UNKNOWN": 0,
        "LANE_STATE_UNKNOWN": 0,
    }

    return mapping.get(traffic_light_state, 0)


def get_traffic_lights_state(scenario):
    traffic_lights_state = TrafficLightState()
    i = 0

    swing_index = NUM_TS_PAST

    for k in scenario["dynamic_map_states"]:
        if i >= DEFAULT_NUM_LIGHT_POSITIONS:
            break

        traffic_lights_state.current_x[i] = scenario["dynamic_map_states"][k]["stop_point"][0]
        traffic_lights_state.current_y[i] = scenario["dynamic_map_states"][k]["stop_point"][1]
        traffic_lights_state.current_z[i] = (
            scenario["dynamic_map_states"][k]["stop_point"][2]
            if len(scenario["dynamic_map_states"][k]["stop_point"]) > 2
            else -1.0
        )
        traffic_lights_state.current_id[i] = scenario["dynamic_map_states"][k]["lane"]

        states = scenario["dynamic_map_states"][k]["state"]["object_state"][:NUM_TS_ALL]
        states_int = [from_traffic_light_state_to_int(state) for state in states]

        traffic_lights_state.past_state[:, i] = states_int[:swing_index]
        traffic_lights_state.current_state[i] = states_int[swing_index]
        traffic_lights_state.future_state[:, i] = states_int[swing_index + 1 :]

        i += 1

    indices_tl = np.where(traffic_lights_state.current_id != -1)
    traffic_lights_state.current_valid[indices_tl] = 1

    traffic_lights_state.past_x[:] = traffic_lights_state.current_x
    traffic_lights_state.past_y[:] = traffic_lights_state.current_y
    traffic_lights_state.past_z[:] = traffic_lights_state.current_z
    traffic_lights_state.past_valid[:] = traffic_lights_state.current_valid
    traffic_lights_state.past_id[:] = traffic_lights_state.current_id

    traffic_lights_state.future_x[:] = traffic_lights_state.current_x
    traffic_lights_state.future_y[:] = traffic_lights_state.current_y
    traffic_lights_state.future_z[:] = traffic_lights_state.current_z
    traffic_lights_state.future_valid[:] = traffic_lights_state.current_valid
    traffic_lights_state.future_id[:] = traffic_lights_state.current_id

    return traffic_lights_state

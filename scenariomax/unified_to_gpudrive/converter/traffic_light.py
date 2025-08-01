from collections import defaultdict


# Traffic light state mapping
TRAFFIC_LIGHT_STATE_MAPPING = {
    # Waymo protobuf lane states (matching GPUDrive template)
    "LANE_STATE_UNKNOWN": "unknown",
    "LANE_STATE_ARROW_STOP": "arrow_stop",
    "LANE_STATE_ARROW_CAUTION": "arrow_caution",
    "LANE_STATE_ARROW_GO": "arrow_go",
    "LANE_STATE_STOP": "stop",
    "LANE_STATE_CAUTION": "caution",
    "LANE_STATE_GO": "go",
    "LANE_STATE_FLASHING_STOP": "flashing_stop",
    "LANE_STATE_FLASHING_CAUTION": "flashing_caution",
    "TRAFFIC_LIGHT_UNKNOWN": "unknown",
    "TRAFFIC_LIGHT_ARROW_RED": "arrow_stop",
    "TRAFFIC_LIGHT_ARROW_YELLOW": "arrow_caution",
    "TRAFFIC_LIGHT_ARROW_GREEN": "arrow_go",
    "TRAFFIC_LIGHT_RED": "stop",
    "TRAFFIC_LIGHT_YELLOW": "caution",
    "TRAFFIC_LIGHT_GREEN": "go",
    "TRAFFIC_LIGHT_FLASHING_RED": "flashing_stop",
    "TRAFFIC_LIGHT_FLASHING_YELLOW": "flashing_caution",
}


def convert_traffic_lights(scenario_net_tl_states):
    tl_dict = defaultdict(lambda: {"state": [], "x": [], "y": [], "z": [], "time_index": [], "lane_id": []})
    for i, (lane_id, tl_state) in enumerate(scenario_net_tl_states.items()):
        position = tl_state["position"]
        x, y = position[:2]
        z = position[2] if len(position) > 2 else 0.0
        light_states = tl_state["states"]

        for j, state in enumerate(light_states):
            tl_dict[lane_id]["state"].append(TRAFFIC_LIGHT_STATE_MAPPING.get(state, "unknown"))
            tl_dict[lane_id]["x"].append(x)
            tl_dict[lane_id]["y"].append(y)
            tl_dict[lane_id]["z"].append(z)
            tl_dict[lane_id]["time_index"].append(j)
            tl_dict[lane_id]["lane_id"].append(lane_id)

    return tl_dict

from scenariomax.core import types


def get_lane_type(waymo_type):
    lane_mapping = {
        0: types.LANE_UNKNOWN,
        1: types.LANE_FREEWAY,
        2: types.LANE_SURFACE_STREET,
        3: types.LANE_BIKE_LANE,
    }

    return lane_mapping.get(waymo_type, types.LANE_UNKNOWN)


def get_road_line_type(waymo_type):
    road_line_mapping = {
        0: types.ROAD_LINE_UNKNOWN,
        1: types.ROAD_LINE_BROKEN_SINGLE_WHITE,
        2: types.ROAD_LINE_SOLID_SINGLE_WHITE,
        3: types.ROAD_LINE_SOLID_DOUBLE_WHITE,
        4: types.ROAD_LINE_BROKEN_SINGLE_YELLOW,
        5: types.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
        6: types.ROAD_LINE_SOLID_SINGLE_YELLOW,
        7: types.ROAD_LINE_SOLID_DOUBLE_YELLOW,
        8: types.ROAD_LINE_PASSING_DOUBLE_YELLOW,
    }

    return road_line_mapping.get(waymo_type, types.ROAD_LINE_UNKNOWN)


def get_road_edge_type(waymo_type):
    road_edge_mapping = {
        0: types.ROAD_EDGE_UNKNOWN,
        1: types.ROAD_EDGE_BOUNDARY,
        2: types.ROAD_EDGE_MEDIAN,
    }

    return road_edge_mapping.get(waymo_type, types.ROAD_EDGE_UNKNOWN)


def get_agent_type(waymo_type):
    agent_mapping = {
        0: types.OTHER,
        1: types.VEHICLE,
        2: types.PEDESTRIAN,
        3: types.CYCLIST,
        4: types.OTHER,
    }

    return agent_mapping.get(waymo_type, types.OTHER)


def get_traffic_light_state(waymo_state):
    traffic_light_state_mapping = {
        0: types.TRAFFIC_LIGHT_UNKNOWN,
        1: types.TRAFFIC_LIGHT_ARROW_RED,
        2: types.TRAFFIC_LIGHT_ARROW_YELLOW,
        3: types.TRAFFIC_LIGHT_ARROW_GREEN,
        4: types.TRAFFIC_LIGHT_RED,
        5: types.TRAFFIC_LIGHT_YELLOW,
        6: types.TRAFFIC_LIGHT_GREEN,
        7: types.TRAFFIC_LIGHT_FLASHING_RED,
        8: types.TRAFFIC_LIGHT_FLASHING_YELLOW,
    }

    return traffic_light_state_mapping.get(waymo_state, types.TRAFFIC_LIGHT_UNKNOWN)

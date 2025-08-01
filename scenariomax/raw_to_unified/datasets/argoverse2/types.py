from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneMarkType, LaneType

from scenariomax.core import types as scenario_type


def get_traffic_obj_type(av2_obj_type):
    if av2_obj_type == ObjectType.VEHICLE or av2_obj_type == ObjectType.BUS:
        return scenario_type.VEHICLE
    # elif av2_obj_type == ObjectType.MOTORCYCLIST:
    #     return scenario_type.MOTORCYCLIST
    elif av2_obj_type == ObjectType.PEDESTRIAN:
        return scenario_type.PEDESTRIAN
    elif av2_obj_type == ObjectType.CYCLIST:
        return scenario_type.CYCLIST
    # elif av2_obj_type == ObjectType.BUS:
    #     return scenario_type.BUS
    # elif av2_obj_type == ObjectType.STATIC:
    #     return scenario_type.STATIC
    # elif av2_obj_type == ObjectType.CONSTRUCTION:
    #     return scenario_type.CONSTRUCTION
    # elif av2_obj_type == ObjectType.BACKGROUND:
    #     return scenario_type.BACKGROUND
    # elif av2_obj_type == ObjectType.RIDERLESS_BICYCLE:
    #     return scenario_type.RIDERLESS_BICYCLE
    # elif av2_obj_type == ObjectType.UNKNOWN:
    #     return scenario_type.UNKNOWN
    else:
        return scenario_type.OTHER


def get_lane_type(av2_lane_type):
    if av2_lane_type == LaneType.VEHICLE or av2_lane_type == LaneType.BUS:
        return scenario_type.LANE_SURFACE_STREET
    elif av2_lane_type == LaneType.BIKE:
        return scenario_type.LANE_BIKE_LANE
    else:
        raise ValueError(f"Unknown nuplan lane type: {av2_lane_type}")


def get_lane_mark_type(av2_mark_type):
    conversion_dict = {
        LaneMarkType.DOUBLE_SOLID_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        LaneMarkType.DOUBLE_SOLID_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        LaneMarkType.SOLID_YELLOW: "ROAD_LINE_SOLID_SINGLE_YELLOW",
        LaneMarkType.SOLID_WHITE: "ROAD_LINE_SOLID_SINGLE_WHITE",
        LaneMarkType.DASHED_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.DASHED_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.DASH_SOLID_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        LaneMarkType.DASH_SOLID_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        LaneMarkType.DOUBLE_DASH_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.DOUBLE_DASH_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.SOLID_DASH_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.SOLID_DASH_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.SOLID_BLUE: "UNKNOWN_LINE",
        LaneMarkType.NONE: "UNKNOWN_LINE",
        LaneMarkType.UNKNOWN: "UNKNOWN_LINE",
    }

    return conversion_dict.get(av2_mark_type, "UNKNOWN_LINE")

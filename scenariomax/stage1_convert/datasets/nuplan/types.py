import os

import nuplan
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from scenariomax.core import types


NuPlanEgoType = TrackedObjectType.EGO

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)


def get_road_line_type(nuplan_type):
    """Map nuPlan boundary types to unified road line types."""
    road_line_mapping = {
        0: types.ROAD_LINE_BROKEN_SINGLE_WHITE,
        1: types.ROAD_LINE_SOLID_SINGLE_WHITE,  # Added mapping for solid white
        2: types.ROAD_LINE_SOLID_SINGLE_WHITE,
        3: types.ROAD_LINE_UNKNOWN,
        4: types.ROAD_LINE_SOLID_SINGLE_YELLOW,  # Added yellow line support
        5: types.ROAD_LINE_BROKEN_SINGLE_YELLOW,  # Added broken yellow
        6: types.ROAD_LINE_SOLID_DOUBLE_WHITE,  # Added double lines
        7: types.ROAD_LINE_SOLID_DOUBLE_YELLOW,
    }

    return road_line_mapping.get(nuplan_type, types.ROAD_LINE_UNKNOWN)


def get_agent_type(nuplan_type):
    """Map nuPlan tracked object types to unified agent types."""
    agent_mapping = {
        TrackedObjectType.VEHICLE: types.VEHICLE,
        TrackedObjectType.PEDESTRIAN: types.PEDESTRIAN,
        TrackedObjectType.BICYCLE: types.CYCLIST,
        TrackedObjectType.TRAFFIC_CONE: None,  # Not supported in unified format
        TrackedObjectType.BARRIER: None,  # Not supported in unified format
        TrackedObjectType.CZONE_SIGN: None,  # Not supported in unified format
        TrackedObjectType.GENERIC_OBJECT: None,  # Not supported in unified format
    }

    return agent_mapping.get(nuplan_type, types.OTHER)


def get_traffic_light_state(status):
    traffic_light_state_mapping = {
        TrafficLightStatusType.RED: types.TRAFFIC_LIGHT_RED,
        TrafficLightStatusType.YELLOW: types.TRAFFIC_LIGHT_YELLOW,
        TrafficLightStatusType.GREEN: types.TRAFFIC_LIGHT_GREEN,
        TrafficLightStatusType.UNKNOWN: types.TRAFFIC_LIGHT_UNKNOWN,
    }

    return traffic_light_state_mapping.get(status, types.TRAFFIC_LIGHT_UNKNOWN)

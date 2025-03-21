import os

from scenariomax.raw_to_unified.type import ScenarioType


try:
    import nuplan
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

    NuPlanEgoType = TrackedObjectType.EGO

    NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
except ImportError as e:
    raise RuntimeError("NuPlan package not found. Please install NuPlan to use this module.") from e


def get_line_type(nuplan_type):
    if nuplan_type == 2:
        return ScenarioType.LINE_SOLID_SINGLE_WHITE
    elif nuplan_type == 0:
        return ScenarioType.LINE_BROKEN_SINGLE_WHITE
    elif nuplan_type == 3:
        return ScenarioType.LINE_UNKNOWN
    else:
        raise ValueError(f"Unknown line type: {nuplan_type}")


def get_traffic_obj_type(nuplan_type):
    if nuplan_type == TrackedObjectType.VEHICLE:
        return ScenarioType.VEHICLE
    elif nuplan_type == TrackedObjectType.TRAFFIC_CONE:
        return ScenarioType.TRAFFIC_CONE
    elif nuplan_type == TrackedObjectType.PEDESTRIAN:
        return ScenarioType.PEDESTRIAN
    elif nuplan_type == TrackedObjectType.BICYCLE:
        return ScenarioType.CYCLIST
    elif nuplan_type == TrackedObjectType.BARRIER:
        return ScenarioType.TRAFFIC_BARRIER
    elif nuplan_type == TrackedObjectType.EGO:
        raise ValueError("Ego should not be in detected resukts")
    else:
        return None


def set_light_status(status):
    if status == TrafficLightStatusType.GREEN:
        return ScenarioType.LIGHT_GREEN
    elif status == TrafficLightStatusType.RED:
        return ScenarioType.LIGHT_RED
    elif status == TrafficLightStatusType.YELLOW:
        return ScenarioType.LIGHT_YELLOW
    elif status == TrafficLightStatusType.UNKNOWN:
        return ScenarioType.LIGHT_UNKNOWN

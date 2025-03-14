import logging
import math
import os

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import nuplan
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    NuPlanEgoType = TrackedObjectType.EGO

    NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
except ImportError as e:
    raise RuntimeError(e)


def get_center_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = get_center_vector(points, center)

    return points


def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([get_center_vector([pose.x, pose.y], nuplan_center) for pose in path])

    return points


def set_light_position(scenario, lane_id, center, target_position=8):
    lane = scenario.map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)

    assert lane is not None, f"Can not find lane: {lane_id}"

    path = lane.baseline_path.discrete_path
    acc_length = 0
    point = [path[0].x, path[0].y]

    for k, point in enumerate(path[1:], start=1):
        previous_p = path[k - 1]
        acc_length += np.linalg.norm([point.x - previous_p.x, point.y - previous_p.y])
        if acc_length > target_position:
            break

    return [point.x - center[0], point.y - center[1]]

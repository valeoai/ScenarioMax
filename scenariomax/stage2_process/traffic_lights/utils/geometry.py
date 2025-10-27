import math

import numpy as np

from .generic import Direction, Pt, UnionFind


def distance_between_points(pt1: Pt, pt2: Pt) -> float:
    """
    Return the distance between two points pt1 and pt2
    """
    if any([pt1.z is None, pt2.z is None]):
        return ((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2) ** 0.5
    return ((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2 + (pt1.z - pt2.z) ** 2) ** 0.5


def two_lines_parallel(line1, line2, LINE_PARALLEL_THRESHOLD: float = 15) -> bool:
    """
    Calculate the angle of two lines represented by line1 and line2,
    and return True if they are nearly parallel (and in the same direction)

    'LINE_PARALLEL_THRESHOLD' unit is degree
    """

    vec1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    vec2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

    mag_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    mag_vec2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    cos_angle = dot_product / (mag_vec1 * mag_vec2)
    cos_angle = max(min(cos_angle, 1), -1)

    angle = math.acos(cos_angle) * (180 / math.pi)

    return angle < LINE_PARALLEL_THRESHOLD


def polyline_length(polyline: list[Pt]) -> float:
    """
    Return the length of this lane.
    """
    distance_list = [distance_between_points(polyline[i - 1], polyline[i]) for i in range(1, len(polyline))]
    return sum(distance_list)


def find_polyline_nearest_point(polyline: list[Pt], reference_point: Pt) -> int:
    """
    Find the point to be truncated at. The point is on polyline, and nearest to the reference_point.
    """

    # feature is to be truncated by the nearest point of reference_point
    distances = [distance_between_points(pt, reference_point) for pt in polyline]
    return distances.index(min(distances))


def real_neighbor_type(
    polyline1: list[Pt],
    polyline2: list[Pt],
    POINT_CLOSE_THRESHOLD: float = 5,
    LENGTH_DIFFERENCE_THRESHOLD: float = 3,
) -> str:
    """
    Return the type of neighborship between feature1 and feature2.

       feature1 and feature2 should be already real neighbors. Feature1 should be longer than feature2.
    """
    start_point_close = distance_between_points(polyline1[0], polyline2[0]) < POINT_CLOSE_THRESHOLD
    end_point_close = distance_between_points(polyline1[-1], polyline2[-1]) < POINT_CLOSE_THRESHOLD
    polyline1_length = polyline_length(polyline1)
    polyline2_length = polyline_length(polyline2)
    polyline1_longer = polyline2_length + LENGTH_DIFFERENCE_THRESHOLD < polyline1_length

    if start_point_close and end_point_close:
        return "complete"
    elif start_point_close and (not end_point_close) and polyline1_longer:
        return "side-start"
    elif (not start_point_close) and end_point_close and polyline1_longer:
        return "side-end"
    else:
        return "other"


def calculate_turning_angle(points):
    """in radians"""

    vectors = np.diff(points, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

    total_turning_angle = np.sum(angle_diffs)

    return total_turning_angle


def classify_direction(shape) -> Direction:
    total_turn_angle = calculate_turning_angle(shape)
    if np.abs(total_turn_angle) < np.pi / 6:
        return Direction.S
    elif total_turn_angle > 0:
        return Direction.L
    else:
        return Direction.R


def points_to_vector(pt_start: Pt, pt_end: Pt) -> np.ndarray:
    return np.array([pt_end.x - pt_start.x, pt_end.y - pt_start.y])


def vector_heading(vec, unit: str = "radian") -> float:
    """unit = 'degree' | 'radian'"""

    if unit == "radian":
        return np.arctan2(vec[1], vec[0])
    elif unit == "degree":
        return np.rad2deg(np.arctan2(vec[1], vec[0]))


def angle_of_two_vectors(vec1, vec2, unit: str = "radian") -> float:
    """unit = 'degree' | 'radian'"""
    # vec1, vec2 = (x,y, [z])
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        raise ValueError("One of the vectors is a zero vector.")

    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    if unit == "radian":
        return np.arccos(cos_theta)  # [0, pi]
    elif unit == "degree":
        return np.degrees(np.arccos(cos_theta))  # [0, 180]
    else:
        raise Exception()


def angle_of_twoheadings(angle1, angle2, unit: str = "radian"):
    """unit ='degree' | 'radian'"""

    if unit == "degree":
        angle1, angle2 = np.deg2rad(angle1), np.deg2rad(angle2)

    diff = np.abs(angle1 - angle2)

    while diff > 2 * np.pi:
        diff -= np.pi * 2
    if diff > np.pi:
        diff = 2 * np.pi - diff

    if unit == "radian":
        return diff  # [0, pi]
    elif unit == "degree":
        return np.rad2deg(diff)  # [0, 180]


def group_vectors_by_angles(lane_vectors: list[np.ndarray], ANGLE_CRITERIA: float = np.pi / 6) -> list[list[int]]:
    uf = UnionFind(len(lane_vectors))  # uf[i]对应node.incoming_SUMO_edges[i]
    for i in range(len(lane_vectors)):
        for j in range(len(lane_vectors)):
            if angle_of_two_vectors(lane_vectors[i], lane_vectors[j]) < ANGLE_CRITERIA:
                uf.union(i, j)
    return uf.form_groups()

import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from scenariomax.core import types


# simplified representation
class TLS(Enum):
    ABSENT = -1
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


# direction
class Direction(Enum):
    L = 0
    S = 1
    R = 2


class Pt:
    def __init__(self, x, y, z=None) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # 路径压缩
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

    def form_groups(self) -> list[list[int]]:
        """
        Given a UnionFind object uf, which represents the edge grouping result, returns a list of groups of edges that
        are meant to be joined into single junctions. The criteria is _is_connection_group().

        For example,
        [ [1,2,3],
          [4],
          [5,6]]
        """

        root_to_elements: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in root_to_elements:
                root_to_elements[root] = []
            root_to_elements[root].append(i)

        groups = [elements for elements in root_to_elements.values()]
        return groups


# Import after class definitions to avoid circular imports
from .geometry import (  # noqa: E402
    angle_of_two_vectors,
    angle_of_twoheadings,
    group_vectors_by_angles,
    points_to_vector,
    vector_heading,
)
from .intersection import ApproachingLane, VehicleState  # noqa: E402


def group_lanes_into_ways(approaching_lanes: list[ApproachingLane]) -> list[list[ApproachingLane]]:
    """
    Groups approaching lanes into ways based on their heading directions.

    This function performs the following steps:
    1. Groups lanes based on their heading directions by forming vectors from the start to the end points of each lane.
    2. Adjusts the order of the groups to ensure that opposite directions are correctly
    paired for 3-way and 4-way intersections.

    Args:
        approaching_lanes (List[ApproachingLane]): A list of ApproachingLane objects representing
        the lanes approaching an intersection.

    Returns:
        List[List[ApproachingLane]]: A list of lists, where each sublist contains ApproachingLane objects
        grouped by their heading directions.
    """

    def _form_lane_vector(lane: ApproachingLane) -> np.ndarray:
        start_point = lane.shape[-min(50, len(lane.shape))]
        end_point = lane.shape[-1]
        return points_to_vector(start_point, end_point)

    # I. group edges based on heading direction
    lane_vectors = [_form_lane_vector(lane) for lane in approaching_lanes]
    groups = group_vectors_by_angles(lane_vectors)
    ways: list[list[ApproachingLane]] = [
        [approaching_lanes[i] for i in group] for group in groups
    ]  # [[], [lane0, lane1, lane2, lane3...], ...]

    # II. adjust the order of how approaches are stored in the list
    # if it is a 3-way intersection, we ensure that way0 & way1 are the two in the opposite directions
    if len(ways) == 3:
        angle01 = angle_of_two_vectors(_form_lane_vector(ways[1][0]), _form_lane_vector(ways[0][0]))
        angle02 = angle_of_two_vectors(_form_lane_vector(ways[2][0]), _form_lane_vector(ways[0][0]))
        angle12 = angle_of_two_vectors(_form_lane_vector(ways[2][0]), _form_lane_vector(ways[1][0]))
        if angle02 > angle01 and angle02 > angle12:  # 0 2方向相对
            ways = [ways[0], ways[2], ways[1]]
        elif angle12 > angle01 and angle12 > angle02:  # 1 2方向相对
            ways = [ways[1], ways[2], ways[0]]
    # if it is a 4-way intersection, we ensure that way0&way2 are opposites, way1&way3 are opposites
    elif len(ways) == 4:
        ways.sort(key=lambda way: vector_heading(_form_lane_vector(way[0])), reverse=False)

    return ways


def has_unprotected_left_turns(tls_state: list[dict[tuple, TLS]]) -> bool:
    """
    Determines if there are unprotected left turns in the given traffic light state.
    Args:
        tls_state (List[Dict[tuple, TLS]]): A list of dictionaries representing the traffic light states.
            Each dictionary maps a tuple (representing a direction) to a TLS object.
    Returns:
        bool: True if there are unprotected left turns, False otherwise.
    Notes:
        - The function checks pairs of opposite directions to determine if there are unprotected left turns.
        - If the traffic light state has 3 entries, it considers the pair (0, 1) as opposite.
        - If the traffic light state has 4 entries, it considers the pairs (0, 2) and (1, 3) as opposite.
        - An unprotected left turn is identified if both directions in a pair have a green light
        for left and straight directions simultaneously.
    """

    # find out pairs of ways that are opposite to each other
    # in order to find out whether there is unprotected left turns
    if len(tls_state) == 3:
        opposite_way_pairs = [(0, 1)]
    elif len(tls_state) == 4:
        opposite_way_pairs = [(0, 2), (1, 3)]
    else:
        opposite_way_pairs = []

    for i, j in opposite_way_pairs:
        phase_i_L = next((phase for phase in tls_state[i] if Direction.L in phase), None)
        phase_j_L = next((phase for phase in tls_state[j] if Direction.S in phase), None)
        if phase_i_L and phase_j_L and tls_state[i][phase_i_L] == tls_state[j][phase_j_L] == TLS.GREEN:
            return True
    return False


def assign_veh_states_to_lane(
    tracks: dict,
    lane_center_matrix: np.ndarray,
    row_to_lane_id: dict = None,
    start_step: int = 0,
    end_step: int = 91,
    DISTANCE_CRITERIA: float = 4,
    ANGLE_CRITERIA: float = np.pi / 12,
    ACCELERATION_MAXLIMIT: float = 10,
) -> dict[str, list[dict[int, VehicleState]]]:
    """
    Assigns vehicle states to lanes based on distance, angle, and acceleration criteria.
    Args:
        tracks (list): List of vehicle tracks, where each track contains historical states of a vehicle.
        lane_center_matrix (np.ndarray): A matrix representing the center points of lanes.
        row_to_lane_id (dict, optional): A mapping from row indices to lane IDs. Defaults to None.
        start_step (int, optional): The starting time step for processing. Defaults to 0.
        end_step (int, optional): The ending time step for processing. Defaults to 91.
        DISTANCE_CRITERIA (float, optional): The maximum allowable distance between a vehicle and a lane center point.
        Defaults to 4.
        ANGLE_CRITERIA (float, optional): The maximum allowable angle difference between a vehicle's heading
        and a lane's direction. Defaults to np.pi / 12.
        ACCELERATION_MAXLIMIT (float, optional): The maximum allowable acceleration for a vehicle. Defaults to 10.
    Returns:
        Dict[str, List[Dict[int, VehicleState]]]: A dictionary where keys are lane IDs
        and values are lists of dictionaries. Each dictionary contains vehicle states indexed by vehicle IDs.
    """

    def _filter_criteria(row: int, heading) -> bool:
        # 1. distance criteria
        distance_criteria = min_distances[row] < DISTANCE_CRITERIA

        # 2. angle criteria
        if min_dis_col[row] == 0:
            lane_start_point = lane_center_matrix[row][min_dis_col[row]]
            lane_end_point = lane_center_matrix[row][min_dis_col[row] + 1]
        else:
            lane_start_point = lane_center_matrix[row][min_dis_col[row] - 1]
            lane_end_point = lane_center_matrix[row][min_dis_col[row]]

        lane_vector = lane_end_point - lane_start_point
        lane_angle = vector_heading(lane_vector, unit="radian")
        angle_abs_diff = angle_of_twoheadings(heading, lane_angle, unit="radian")
        angle_criteria = angle_abs_diff < ANGLE_CRITERIA

        return distance_criteria and angle_criteria

    # a container for sotring results
    veh_assignment_by_row: list[list[dict[int, VehicleState]]] = [
        [{} for _ in range(start_step, end_step)] for _ in range(lane_center_matrix.shape[0])
    ]

    for _id, track in tracks.items():  # for each vehicle
        if track["type"] != types.VEHICLE:
            continue

        position = track["states"]["position"]
        valid = track["states"]["valid"]
        velocity = track["states"]["velocity"]
        heading = track["states"]["heading"]

        for tt in range(start_step, end_step):  # for each historical state of this vehicle
            if not valid[tt]:
                continue

            # 1. calculate distances point-2-point
            veh_position: np.ndarray = np.array([position[tt][0], position[tt][1], position[tt][2]])
            distances = np.linalg.norm(lane_center_matrix - veh_position, axis=2)
            min_distances = np.min(distances, axis=1)
            min_dis_col = np.argmin(distances, axis=1)  # distance to each feature

            # 2. filter out all possible lanes that the vehicle is at
            candidate_rows = [row for row in range(min_distances.shape[0]) if _filter_criteria(row, heading[tt])]
            if not candidate_rows:
                continue

            # 3. determine the exact lanes the vehicle is at
            best_row = candidate_rows[np.argmin([distances[row, min_dis_col[row]] for row in candidate_rows])]
            best_col = min_dis_col[best_row]

            # 4. the speed of the vehicle
            absolute_speed = np.linalg.norm([velocity[tt][0], velocity[tt][1]])

            # 5. the acceleration of the vehicle
            acceleration = 0
            j = tt - 5
            while j >= 0:
                if valid[j]:
                    absolute_speed_last_step = np.linalg.norm([velocity[j][0], velocity[j][1]])
                    acceleration = (absolute_speed - absolute_speed_last_step) / ((tt - j) / 10)
                    break
                j -= 1
            if abs(acceleration) > ACCELERATION_MAXLIMIT:
                acceleration = 0
            veh_assignment_by_row[best_row][tt][int(_id)] = VehicleState(
                int(_id),
                int(best_col),
                float(absolute_speed),
                float(acceleration),
            )

    if not row_to_lane_id:
        row_to_lane_id = {i: i for i in range(len(veh_assignment_by_row))}
    veh_assignment_by_lane_id: dict[str, list[dict[int, VehicleState]]] = {
        str(row_to_lane_id[row]): veh_assignment_by_row[row] for row in range(len(veh_assignment_by_row))
    }

    return veh_assignment_by_lane_id


def save_veh_states_assignment(veh_assignment: dict[int, list[dict[int, VehicleState]]], file_path: Path) -> None:
    """
    Save the vehicle states assignment to a specified file path in JSON format.
    Args:
        veh_assignment (Dict[int, List[Dict[int, VehicleState]]]): A dictionary where the keys are lane IDs
        and the values are lists of dictionaries. Each dictionary contains vehicle IDs
        as keys and VehicleState objects as values.
        file_path (Path): The file path where the vehicle states assignment data will be saved.
    Returns:
        None
    """

    file_path = Path(file_path)
    base_dir = os.path.dirname(file_path)
    os.makedirs(base_dir, exist_ok=True)

    veh_assignment_output = copy.deepcopy(veh_assignment)
    for lane_id in veh_assignment:
        for t, veh_states in enumerate(veh_assignment[lane_id]):
            for veh_id, veh_state in veh_states.items():
                veh_assignment_output[lane_id][t][veh_id] = {
                    "object_id": veh_state.object_id,
                    "lane_pos_idx": veh_state.lane_pos_idx,
                    "speed": veh_state.speed,
                    "acceleration": veh_state.acceleration,
                }
    with open(file_path, "w") as file:
        json.dump(veh_assignment_output, file, indent=4)
    print(f"[File Output] Saved vehicle states assignment data: {file_path} ...")


def load_veh_states_assignment(file_path: Path) -> dict[Any, list[dict[int, VehicleState]]]:
    """
    Loads vehicle states assignment data from a JSON file and converts it into a dictionary
    with lane IDs as keys and lists of dictionaries containing vehicle states as values.
    Args:
        file_path (Path): The path to the JSON file containing the vehicle states assignment data.
    Returns:
        Dict[Any, List[Dict[int, VehicleState]]]: A dictionary where each key is a lane ID and each value
        is a list of dictionaries. Each dictionary contains vehicle IDs as keys and VehicleState objects as values.
    """

    print(f"[File Input] Loading vehicle states assignment data: {file_path} ...")
    with open(file_path) as file:
        veh_assignment = json.load(file)

    veh_assignment_loaded = copy.deepcopy(veh_assignment)
    for lane_id in veh_assignment:
        for t, veh_states in enumerate(veh_assignment[lane_id]):
            for veh_id, veh_state in veh_states.items():
                veh_assignment_loaded[lane_id][t][veh_id] = VehicleState(
                    veh_state["object_id"],
                    veh_state["lane_pos_idx"],
                    veh_state["speed"],
                    veh_state["acceleration"],
                )

    return veh_assignment_loaded

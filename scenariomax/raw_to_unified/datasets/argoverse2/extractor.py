from typing import Final

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scenariomax import logger_utils
from scenariomax.core import description
from scenariomax.core import types as scenario_type
from scenariomax.raw_to_unified.datasets import utils as converter_utils
from scenariomax.raw_to_unified.datasets.argoverse2 import types as argoverse2_type


logger = logger_utils.get_logger(__name__)


_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_ESTIMATED_PEDESTRIAN_LENGTH_M: Final[float] = 0.5
_ESTIMATED_PEDESTRIAN_WIDTH_M: Final[float] = 0.5
_ESTIMATED_BUS_LENGTH_M: Final[float] = 12.0
_ESTIMATED_BUS_WIDTH_M: Final[float] = 2.5

_HIGHWAY_SPEED_LIMIT_MPH: Final[float] = 85.0


def convert_av2_scenario(scenario, version):
    _scenario = description.ScenarioDescription()

    _scenario[description.ID] = scenario.scenario_id
    _scenario[description.VERSION] = version

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.

    track_length = scenario.timestamps_ns.shape[0]

    tracks, category = extract_tracks(scenario.tracks, scenario.focal_track_id, track_length)

    _scenario[description.LENGTH] = track_length
    _scenario[description.TRACKS] = tracks
    _scenario[description.DYNAMIC_MAP_STATES] = {}
    map_features = extract_map_features(scenario.static_map)
    _scenario[description.MAP_FEATURES] = map_features
    _scenario[description.METADATA] = {}
    _scenario[description.METADATA][description.ID] = _scenario[description.ID]
    _scenario[description.METADATA][description.TIMESTEP] = np.array(list(range(track_length))) / 10
    _scenario[description.METADATA][description.SDC_ID] = "AV"
    _scenario[description.METADATA]["dataset"] = "av2"
    _scenario[description.METADATA]["scenario_id"] = scenario.scenario_id
    _scenario[description.METADATA]["source_file"] = scenario.scenario_id
    _scenario[description.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    _scenario[description.METADATA]["current_time_index"] = 49

    # obj id
    obj_keys = list(tracks.keys())
    _scenario[description.METADATA]["objects_of_interest"] = [
        obj_keys[idx] for idx, cat in enumerate(category) if cat == 2
    ]
    _scenario[description.METADATA]["sdc_track_index"] = obj_keys.index("AV")

    track_index = [obj_keys.index(scenario.focal_track_id)]
    track_id = [scenario.focal_track_id]
    track_difficulty = [0]
    track_obj_type = [tracks[id]["type"] for id in track_id]
    _scenario[description.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": track_difficulty[count],
            "object_type": track_obj_type[count],
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return _scenario


def extract_tracks(tracks, sdc_idx, track_length):
    ret = {}

    def _object_state_template(object_id):
        return {
            "type": None,
            "state": {  # Never add extra dim if the value is scalar.
                "position": np.zeros([track_length, 3], dtype=np.float32),
                "length": np.zeros([track_length], dtype=np.float32),
                "width": np.zeros([track_length], dtype=np.float32),
                "height": np.zeros([track_length], dtype=np.float32),
                "heading": np.zeros([track_length], dtype=np.float32),
                "velocity": np.zeros([track_length, 2], dtype=np.float32),
                "valid": np.zeros([track_length], dtype=bool),
            },
            "metadata": {"track_length": track_length, "type": None, "object_id": object_id, "dataset": "av2"},
        }

    track_category = []

    for obj in tracks:
        object_id = obj.track_id
        track_category.append(obj.category.value)
        obj_state = _object_state_template(object_id)
        # Transform it to Waymo type string
        obj_state["type"] = argoverse2_type.get_traffic_obj_type(obj.object_type)
        if obj_state["type"] == scenario_type.VEHICLE:
            length = _ESTIMATED_VEHICLE_LENGTH_M
            width = _ESTIMATED_VEHICLE_WIDTH_M
        elif obj_state["type"] == scenario_type.PEDESTRIAN:
            length = _ESTIMATED_PEDESTRIAN_LENGTH_M
            width = _ESTIMATED_PEDESTRIAN_WIDTH_M
        elif obj_state["type"] == scenario_type.CYCLIST:
            length = _ESTIMATED_CYCLIST_LENGTH_M
            width = _ESTIMATED_CYCLIST_WIDTH_M
        # elif obj_state["type"] == scenario_type.BUS:
        #     length = _ESTIMATED_BUS_LENGTH_M
        #     width = _ESTIMATED_BUS_WIDTH_M
        else:
            length = 1
            width = 1

        for _, state in enumerate(obj.object_states):
            step_count = state.timestep
            obj_state["state"]["position"][step_count][0] = state.position[0]
            obj_state["state"]["position"][step_count][1] = state.position[1]
            obj_state["state"]["position"][step_count][2] = 0

            # l = [state.length for state in obj.states]
            # w = [state.width for state in obj.states]
            # h = [state.height for state in obj.states]
            # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
            obj_state["state"]["length"][step_count] = length
            obj_state["state"]["width"][step_count] = width
            obj_state["state"]["height"][step_count] = 1

            # heading = [state.heading for state in obj.states]
            obj_state["state"]["heading"][step_count] = state.heading

            obj_state["state"]["velocity"][step_count][0] = state.velocity[0]
            obj_state["state"]["velocity"][step_count][1] = state.velocity[1]

            obj_state["state"]["valid"][step_count] = True

        obj_state["metadata"]["type"] = obj_state["type"]

        ret[object_id] = obj_state

    return ret, track_category


def extract_lane_mark(lane_mark):
    line = {}
    line["type"] = argoverse2_type.get_lane_mark_type(lane_mark.mark_type)
    line["polyline"] = lane_mark.polyline.astype(np.float32)

    return line


def extract_map_features(map_features):
    ret = {}
    vector_lane_segments = map_features.get_scenario_lane_segments()
    vector_drivable_areas = map_features.get_scenario_vector_drivable_areas()
    ped_crossings = map_features.get_scenario_ped_crossings()

    # ids = map_features.get_scenario_lane_segment_ids()
    # max_id = max(ids)
    for seg in vector_lane_segments:
        center = {}
        lane_id = str(seg.id)

        # left_id = seg.left_neighbor_id
        # right_id = seg.right_neighbor_id
        # left_marking = extract_lane_mark(seg.left_lane_marking)
        # right_marking = extract_lane_mark(seg.right_lane_marking)
        # ret[left_id] = left_marking
        # ret[right_id] = right_marking

        # Add right line
        if seg.right_lane_marking is not None:
            right_marking = extract_lane_mark(seg.right_lane_marking)
            right_id = f"{lane_id}_right"
            if right_id not in ret:
                ret[right_id] = right_marking
                # center["right_boundaries"].append(right_id)

        # Add left line
        if seg.left_lane_marking is not None:
            left_marking = extract_lane_mark(seg.left_lane_marking)
            left_id = f"{lane_id}_left"
            if left_id not in ret:
                ret[left_id] = left_marking
                # center["left_boundaries"].append(left_id)

        # Add lane center
        center["speed_limit_mph"] = _HIGHWAY_SPEED_LIMIT_MPH
        center["speed_limit_kmh"] = converter_utils.mph_to_kmh(_HIGHWAY_SPEED_LIMIT_MPH)
        center["type"] = argoverse2_type.get_lane_type(seg.lane_type)
        polyline = map_features.get_lane_segment_centerline(seg.id)
        center["polyline"] = polyline.astype(np.float32)
        center["interpolating"] = True
        center["entry_lanes"] = [str(id) for id in seg.predecessors]
        center["exit_lanes"] = [str(id) for id in seg.successors]
        center["left_boundaries"] = []
        center["right_boundaries"] = []
        center["left_neighbor"] = []
        center["right_neighbor"] = []
        center["width"] = np.zeros([len(polyline), 2], dtype=np.float32)

        ret[lane_id] = center

    polygons = []
    for polygon in vector_drivable_areas:
        # convert to shapely polygon
        points = polygon.area_boundary
        polygons.append(Polygon([(p.x, p.y) for p in points]))

    polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
    boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        # id = f'boundary_{idx}'
        # ret[id] = {SD.TYPE: ScenarioType.BOUNDARY_LINE, SD.POLYLINE: block_points}

        for i in range(0, len(block_points), 20):
            id = f"boundary_{idx}{i}"
            ret[id] = {description.TYPE: scenario_type.BOUNDARY_LINE, description.POLYLINE: block_points[i : i + 20]}

    for cross in ped_crossings:
        bound = {}
        bound["type"] = scenario_type.CROSSWALK
        bound["polygon"] = cross.polygon.astype(np.float32)
        ret[str(cross.id)] = bound

    return ret

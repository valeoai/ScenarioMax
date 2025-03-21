import os

import geopandas as gpd
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import unary_union

from scenariomax.raw_to_unified.converter.nuplan.types import get_line_type, get_traffic_obj_type, set_light_status
from scenariomax.raw_to_unified.converter.nuplan.utils import (
    compute_angular_velocity,
    extract_centerline,
    get_center_vector,
    get_points_from_boundary,
    set_light_position,
)
from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.type import ScenarioType


try:
    import nuplan
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

    NuPlanEgoType = TrackedObjectType.EGO

    NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
except ImportError as e:
    raise RuntimeError("NuPlan package not found. Please install NuPlan to use this module.") from e


EGO = "ego"


def convert_nuplan_scenario(scenario: NuPlanScenario, version):
    """
    Data will be interpolated to 0.1s time interval, while the time interval of original key frames are 0.5s.
    """
    scenario_log_interval = scenario.database_interval
    assert abs(scenario_log_interval - 0.1) < 1e-3, (
        "Log interval should be 0.1 or Interpolating is required! By setting NuPlan subsample ratio can address this"
    )

    _scenario = SD()
    _scenario[SD.ID] = scenario.scenario_name
    _scenario[SD.VERSION] = "nuplan_" + version
    _scenario[SD.LENGTH] = scenario.get_number_of_iterations()
    # metadata
    _scenario[SD.METADATA] = {}
    _scenario[SD.METADATA]["dataset"] = "nuplan"
    _scenario[SD.METADATA]["map"] = scenario.map_api.map_name
    _scenario[SD.METADATA]["map_version"] = scenario.map_version
    _scenario[SD.METADATA]["log_name"] = scenario.log_name
    _scenario[SD.METADATA]["scenario_extraction_info"] = scenario._scenario_extraction_info.__dict__
    _scenario[SD.METADATA]["ego_vehicle_parameters"] = scenario.ego_vehicle_parameters.__dict__
    _scenario[SD.METADATA]["scenario_token"] = scenario.scenario_name
    _scenario[SD.METADATA]["scenario_id"] = scenario.scenario_name
    _scenario[SD.METADATA][SD.ID] = scenario.scenario_name
    _scenario[SD.METADATA]["scenario_type"] = scenario.scenario_type
    _scenario[SD.METADATA]["sample_rate"] = scenario_log_interval
    _scenario[SD.METADATA][SD.TIMESTEP] = np.asarray([i * scenario_log_interval for i in range(_scenario[SD.LENGTH])])
    _scenario[SD.METADATA][SD.SDC_ID] = EGO

    # centered all positions to ego car
    state = scenario.get_ego_state_at_iteration(0)
    scenario_center = [state.waypoint.x, state.waypoint.y]

    _scenario[SD.TRACKS] = extract_traffic(scenario, scenario_center)
    _scenario[SD.DYNAMIC_MAP_STATES] = extract_traffic_light(scenario, scenario_center)
    _scenario[SD.MAP_FEATURES] = extract_map_features(scenario.map_api, scenario_center)

    return _scenario


def extract_traffic_light(scenario, center):
    length = scenario.get_number_of_iterations()

    frames = [
        {str(t.lane_connector_id): t.status for t in scenario.get_traffic_light_status_at_iteration(i)}
        for i in range(length)
    ]
    all_lights = set()
    for frame in frames:
        all_lights.update(frame.keys())

    lights = {
        k: {
            "type": ScenarioType.TRAFFIC_LIGHT,
            "state": {SD.TRAFFIC_LIGHT_STATUS: [ScenarioType.LIGHT_UNKNOWN] * length},
            SD.TRAFFIC_LIGHT_POSITION: None,
            SD.TRAFFIC_LIGHT_LANE: str(k),
            "metadata": {
                "track_length": length,
                "type": None,
                "object_id": str(k),
                "lane_id": str(k),
                "dataset": "nuplan",
            },
        }
        for k in list(all_lights)
    }

    for k, frame in enumerate(frames):
        for lane_id, status in frame.items():
            lane_id = str(lane_id)
            lights[lane_id]["state"][SD.TRAFFIC_LIGHT_STATUS][k] = set_light_status(status)
            if lights[lane_id][SD.TRAFFIC_LIGHT_POSITION] is None:
                assert isinstance(lane_id, str), "Lane ID should be str"
                lights[lane_id][SD.TRAFFIC_LIGHT_POSITION] = set_light_position(scenario, lane_id, center)
                lights[lane_id][SD.METADATA][SD.TYPE] = ScenarioType.TRAFFIC_LIGHT

    return lights


def extract_object_state(obj_state, nuplan_center):
    return {
        "position": get_center_vector([obj_state.center.x, obj_state.center.y], nuplan_center),
        "heading": obj_state.center.heading,
        "velocity": get_center_vector([obj_state.velocity.x, obj_state.velocity.y]),
        "valid": 1,
        "length": obj_state.box.length,
        "width": obj_state.box.width,
        "height": obj_state.box.height,
    }


def extract_ego_vehicle_state(state, nuplan_center):
    return {
        "position": get_center_vector([state.waypoint.x, state.waypoint.y], nuplan_center),
        "heading": state.waypoint.heading,
        "velocity": get_center_vector([state.agent.velocity.x, state.agent.velocity.y]),
        "angular_velocity": state.dynamic_car_state.angular_velocity,
        "valid": 1,
        "length": state.agent.box.length,
        "width": state.agent.box.width,
        "height": state.agent.box.height,
    }


def extract_ego_vehicle_state_trajectory(scenario, nuplan_center):
    data = [
        extract_ego_vehicle_state(scenario.get_ego_state_at_iteration(i), nuplan_center)
        for i in range(scenario.get_number_of_iterations())
    ]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"],
            final_heading=data[i + 1]["heading"],
            dt=scenario.database_interval,
        )
    return data


def extract_traffic(scenario: NuPlanScenario, center):
    episode_len = scenario.get_number_of_iterations()
    detection_ret = []
    all_objs = set()
    all_objs.add(EGO)
    for frame_data in [scenario.get_tracked_objects_at_iteration(i).tracked_objects for i in range(episode_len)]:
        new_frame_data = {}
        for obj in frame_data:
            new_frame_data[obj.track_token] = obj
            all_objs.add(obj.track_token)
        detection_ret.append(new_frame_data)

    tracks = {
        k: {
            "type": ScenarioType.UNSET,
            "state": {
                "position": np.zeros(shape=(episode_len, 3)),
                "heading": np.zeros(shape=(episode_len,)),
                "velocity": np.zeros(shape=(episode_len, 2)),
                "valid": np.zeros(shape=(episode_len,)),
                "length": np.zeros(shape=(episode_len, 1)),
                "width": np.zeros(shape=(episode_len, 1)),
                "height": np.zeros(shape=(episode_len, 1)),
            },
            "metadata": {
                "track_length": episode_len,
                "nuplan_type": None,
                "type": None,
                "object_id": k,
                "nuplan_id": k,
            },
        }
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj_state in frame.items():
            assert isinstance(obj_state, Agent | StaticObject)
            obj_type = get_traffic_obj_type(obj_state.tracked_object_type)
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id][SD.TYPE] = obj_type
            if tracks[nuplan_id][SD.METADATA]["nuplan_type"] is None:
                tracks[nuplan_id][SD.METADATA]["nuplan_type"] = int(obj_state.tracked_object_type)
                tracks[nuplan_id][SD.METADATA]["type"] = obj_type

            state = extract_object_state(obj_state, center)
            tracks[nuplan_id]["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
            tracks[nuplan_id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[nuplan_id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[nuplan_id]["state"]["valid"][frame_idx] = 1
            tracks[nuplan_id]["state"]["length"][frame_idx] = state["length"]
            tracks[nuplan_id]["state"]["width"][frame_idx] = state["width"]
            tracks[nuplan_id]["state"]["height"][frame_idx] = state["height"]

    for track in list(tracks_to_remove):
        tracks.pop(track)

    # ego
    sdc_traj = extract_ego_vehicle_state_trajectory(scenario, center)
    ego_track = tracks[EGO]

    for frame_idx, obj_state in enumerate(sdc_traj):
        obj_type = ScenarioType.VEHICLE
        ego_track[SD.TYPE] = obj_type
        if ego_track[SD.METADATA]["nuplan_type"] is None:
            ego_track[SD.METADATA]["nuplan_type"] = int(NuPlanEgoType)
            ego_track[SD.METADATA]["type"] = obj_type
        state = obj_state
        ego_track["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
        ego_track["state"]["valid"][frame_idx] = 1
        ego_track["state"]["heading"][frame_idx] = state["heading"]
        ego_track["state"]["length"][frame_idx] = state["length"]
        ego_track["state"]["width"][frame_idx] = state["width"]
        ego_track["state"]["height"][frame_idx] = state["height"]

    # get velocity here
    vel = ego_track["state"]["position"][1:] - ego_track["state"]["position"][:-1]
    ego_track["state"]["velocity"][:-1] = vel[..., :2] / 0.1
    ego_track["state"]["velocity"][-1] = ego_track["state"]["velocity"][-2]

    # check
    assert EGO in tracks
    for track_id in tracks:
        assert tracks[track_id][SD.TYPE] != ScenarioType.UNSET

    return tracks


def extract_map_features(map_api, center, radius=300):
    map_features = {}

    layer_names = [
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.LANE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
        # unsupported yet
        # SemanticMapLayer.STOP_SIGN,
        # SemanticMapLayer.DRIVABLE_AREA,
    ]

    center_for_query = Point2D(*center)

    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)

    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]

    block_polygons = []

    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            edges = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )

            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue

                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                    sizes = []

                    for idx, polygon in enumerate(boundary[0]):
                        sizes.append(len(polygon.xy[1]))

                    points = boundary[0][np.argmax(sizes)].xy

                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    points = lane_meta_data.polygon.boundary.xy

                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = get_center_vector(polygon, nuplan_center=[center[0], center[1]])

                # According to the map attributes, lanes are numbered left to right with smaller indices being on the
                # left and larger indices being on the right.
                # @ See NuPlanLane.adjacent_edges()
                map_features[lane_meta_data.id] = {
                    SD.TYPE: ScenarioType.LANE_SURFACE_STREET
                    if layer == SemanticMapLayer.ROADBLOCK
                    else ScenarioType.LANE_SURFACE_UNSTRUCTURE,
                    SD.POLYLINE: extract_centerline(lane_meta_data, center),
                    SD.ENTRY: [edge.id for edge in lane_meta_data.incoming_edges],
                    SD.EXIT: [edge.id for edge in lane_meta_data.outgoing_edges],
                    SD.LEFT_NEIGHBORS: [edge.id for edge in block.interior_edges[:index]]
                    if layer == SemanticMapLayer.ROADBLOCK
                    else [],
                    SD.RIGHT_NEIGHBORS: [edge.id for edge in block.interior_edges[index + 1 :]]
                    if layer == SemanticMapLayer.ROADBLOCK
                    else [],
                    SD.POLYGON: polygon,
                }

                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    continue

                # NOTE: only add lines that are in between lanes
                left = lane_meta_data.left_boundary
                right = lane_meta_data.right_boundary
                adjacent_edges = lane_meta_data.adjacent_edges

                take_points = adjacent_edges[0] and adjacent_edges[1]

                if take_points and left.id not in map_features:
                    line_type = get_line_type(int(boundaries.loc[[str(left.id)]]["boundary_type_fid"].iloc[0]))

                    if line_type != ScenarioType.LINE_UNKNOWN:
                        map_features[left.id] = {
                            SD.TYPE: line_type,
                            SD.POLYLINE: get_points_from_boundary(left, center),
                        }

                if take_points and right.id not in map_features:
                    line_type = get_line_type(int(boundaries.loc[[str(right.id)]]["boundary_type_fid"].iloc[0]))

                    if line_type != ScenarioType.LINE_UNKNOWN:
                        map_features[right.id] = {
                            SD.TYPE: line_type,
                            SD.POLYLINE: get_points_from_boundary(right, center),
                        }

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    map_features = process_walkway(map_features, nearest_vector_map, center)
    map_features = process_crosswalk(map_features, nearest_vector_map, center)
    map_features = process_boundary(map_features, nearest_vector_map, center, block_polygons)

    return map_features


def process_walkway(map_features, nearest_vector_map, center):
    for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []

            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))

            points = boundary[0][np.argmax(sizes)].xy

        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy

        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = get_center_vector(polygon, nuplan_center=[center[0], center[1]])

        map_features[area.id] = {SD.TYPE: ScenarioType.BOUNDARY_SIDEWALK, SD.POLYGON: polygon}

    return map_features


def process_crosswalk(map_features, nearest_vector_map, center):
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []

            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))

            points = boundary[0][np.argmax(sizes)].xy

        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy

        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = get_center_vector(polygon, nuplan_center=[center[0], center[1]])

        map_features[area.id] = {SD.TYPE: ScenarioType.CROSSWALK, SD.POLYGON: polygon}

    return map_features


def process_boundary(map_features, nearest_vector_map, center, block_polygons):
    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)

    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = get_center_vector(block_points, center)[::-1]
        id = f"boundary_{idx}"
        map_features[id] = {SD.TYPE: "ROAD_EDGE_BOUNDARY", SD.POLYLINE: block_points}

    return map_features

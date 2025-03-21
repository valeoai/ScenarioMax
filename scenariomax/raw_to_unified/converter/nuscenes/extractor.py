import copy
import logging

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from scenariomax.logger_utils import get_logger
from scenariomax.raw_to_unified.converter.nuscenes.types import get_scenarionet_type
from scenariomax.raw_to_unified.converter.nuscenes.utils import (
    convert_id_to_int,
    extract_frames_scene_info,
    interpolate,
    interpolate_heading,
    parse_frame,
)
from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.type import ScenarioType


logger = get_logger(__name__)

try:
    import logging

    logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)
    from pyquaternion import Quaternion

    from nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import quaternion_yaw
    from nuscenes.map_expansion.arcline_path_utils import discretize_lane
    from nuscenes.map_expansion.map_api import NuScenesMap
except ImportError as e:
    raise RuntimeError("NuScenes package not found. Please install NuScenes to use this module.") from e


def convert_nuscenes_scenario(
    token,
    version,
    nuscenes: NuScenes,
    map_radius=300,
    prediction=False,
    past=1,
    future=8,
    only_lane=False,
):
    """
    Data will be interpolated to 0.1s time interval, while the time interval of original key frames are 0.5s.
    """
    if prediction:
        past_num = int(past / 0.5)
        future_num = int(future / 0.5)
        nusc = nuscenes
        instance_token, sample_token = token.split("_")
        current_sample = last_sample = next_sample = nusc.get("sample", sample_token)
        past_samples = []
        future_samples = []
        for _ in range(past_num):
            if last_sample["prev"] == "":
                break
            last_sample = nusc.get("sample", last_sample["prev"])
            past_samples.append(parse_frame(last_sample, nusc))

        for _ in range(future_num):
            if next_sample["next"] == "":
                break
            next_sample = nusc.get("sample", next_sample["next"])
            future_samples.append(parse_frame(next_sample, nusc))
        frames = past_samples[::-1] + [parse_frame(current_sample, nusc)] + future_samples
        scene_info = copy.copy(nusc.get("scene", current_sample["scene_token"]))
        scene_info["name"] = scene_info["name"] + "_" + token
        scene_info["prediction"] = True
        frames_scene_info = [frames, scene_info]
    else:
        frames_scene_info = extract_frames_scene_info(token, nuscenes)

    scenario_log_interval = 0.1
    frames, scene_info = frames_scene_info
    result = SD()
    result[SD.ID] = scene_info["name"]
    result[SD.VERSION] = "nuscenes" + version
    result[SD.LENGTH] = (len(frames) - 1) * 5 + 1
    result[SD.METADATA] = {}
    result[SD.METADATA]["dataset"] = "nuscenes"
    result[SD.METADATA]["map"] = nuscenes.get("log", scene_info["log_token"])["location"]
    result[SD.METADATA]["date"] = nuscenes.get("log", scene_info["log_token"])["date_captured"]
    # result[SD.METADATA]["dscenario_token"] = scene_token
    result[SD.METADATA][SD.ID] = scene_info["name"]
    result[SD.METADATA]["scenario_id"] = scene_info["name"]
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = np.arange(0.0, (len(frames) - 1) * 0.5 + 0.1, 0.1)
    # interpolating to 0.1s interval
    result[SD.TRACKS] = get_tracks_from_frames(nuscenes, scene_info, frames, num_to_interpolate=5)
    result[SD.METADATA][SD.SDC_ID] = "ego"

    # No traffic light in nuscenes at this stage
    result[SD.DYNAMIC_MAP_STATES] = {}
    if prediction:
        track_to_predict = result[SD.TRACKS][instance_token]
        result[SD.METADATA]["tracks_to_predict"] = {
            instance_token: {
                "track_index": list(result[SD.TRACKS].keys()).index(instance_token),
                "track_id": instance_token,
                "difficulty": 0,
                "object_type": track_to_predict["type"],
            },
        }

    # map
    map_center = np.array(result[SD.TRACKS]["ego"]["state"]["position"][0])
    result[SD.MAP_FEATURES] = get_map_features(scene_info, nuscenes, map_center, map_radius, only_lane=only_lane)
    del frames_scene_info
    del frames
    del scene_info
    return result


def get_map_features(scene_info, nuscenes: NuScenes, map_center, radius=500, points_distance=1, only_lane=False):
    """
    Extract map features from nuscenes data. The objects in specified region will be returned. Sampling rate determines
    the distance between 2 points when extracting lane center line.
    """
    ret = {}
    map_name = nuscenes.get("log", scene_info["log_token"])["location"]
    map_api = NuScenesMap(dataroot=nuscenes.dataroot, map_name=map_name)

    layer_names = [
        # "line",
        # "polygon",
        # "node",
        "drivable_area",
        "road_segment",
        "road_block",
        "lane",
        "ped_crossing",
        "walkway",
        # 'stop_line',
        # 'carpark_area',
        "lane_connector",
        "road_divider",
        "lane_divider",
        "traffic_light",
    ]
    # road segment includes all roadblocks (a list of lanes in the same direction), intersection and unstructured road

    map_objs = map_api.get_records_in_radius(map_center[0], map_center[1], radius, layer_names)

    if not only_lane:
        # build map boundary
        polygons = []
        for id in map_objs["drivable_area"]:
            seg_info = map_api.get("drivable_area", id)
            assert seg_info["token"] == id
            for polygon_token in seg_info["polygon_tokens"]:
                polygon = map_api.extract_polygon(polygon_token)
                polygons.append(polygon)
        # for id in map_objs["road_segment"]:
        #     seg_info = map_api.get("road_segment", id)
        #     assert seg_info["token"] == id
        #     polygon = map_api.extract_polygon(seg_info["polygon_token"])
        #     polygons.append(polygon)
        # for id in map_objs["road_block"]:
        #     seg_info = map_api.get("road_block", id)
        #     assert seg_info["token"] == id
        #     polygon = map_api.extract_polygon(seg_info["polygon_token"])
        #     polygons.append(polygon)
        polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
        boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])))
            mask = np.linalg.norm(block_points - map_center[:2], axis=-1) < radius

            if np.sum(mask) == 0:
                continue

            id = f"boundary_{idx}"
            ret[id] = {SD.TYPE: ScenarioType.BOUNDARY_LINE, SD.POLYLINE: block_points[mask][::-1]}

        # broken line
        for id in map_objs["lane_divider"]:
            line_info = map_api.get("lane_divider", id)
            assert line_info["token"] == id
            line = map_api.extract_line(line_info["line_token"]).coords.xy
            line = np.asarray([[line[0][i], line[1][i]] for i in range(len(line[0]))])
            mask = np.linalg.norm(line - map_center[:2], axis=-1) < radius
            if np.sum(mask) == 0:
                continue

            ret[convert_id_to_int(id)] = {SD.TYPE: ScenarioType.LINE_BROKEN_SINGLE_WHITE, SD.POLYLINE: line[mask]}

        # solid line
        for id in map_objs["road_divider"]:
            line_info = map_api.get("road_divider", id)
            assert line_info["token"] == id
            line = map_api.extract_line(line_info["line_token"]).coords.xy
            line = np.asarray([[line[0][i], line[1][i]] for i in range(len(line[0]))])
            mask = np.linalg.norm(line - map_center[:2], axis=-1) < radius
            if np.sum(mask) == 0:
                continue

            ret[convert_id_to_int(id)] = {SD.TYPE: ScenarioType.LINE_SOLID_SINGLE_YELLOW, SD.POLYLINE: line[mask]}

        # crosswalk
        for id in map_objs["ped_crossing"]:
            info = map_api.get("ped_crossing", id)
            assert info["token"] == id
            boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
            boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])

            ret[convert_id_to_int(id)] = {SD.TYPE: ScenarioType.CROSSWALK, SD.POLYGON: boundary_polygon}

        # walkway
        for id in map_objs["walkway"]:
            info = map_api.get("walkway", id)
            assert info["token"] == id
            boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
            boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])

            ret[convert_id_to_int(id)] = {SD.TYPE: ScenarioType.BOUNDARY_SIDEWALK, SD.POLYGON: boundary_polygon}

    # normal lane
    for id in map_objs["lane"]:
        lane_info = map_api.get("lane", id)
        assert lane_info["token"] == id
        boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
        mask = np.linalg.norm(boundary_polygon - map_center[:2], axis=-1) < radius
        if np.sum(mask) == 0:
            continue
        # boundary_polygon += [[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))]

        ret[convert_id_to_int(id)] = {
            SD.TYPE: ScenarioType.LANE_SURFACE_STREET,
            SD.POLYLINE: np.asarray(discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            SD.POLYGON: boundary_polygon[mask],
            SD.ENTRY: [convert_id_to_int(_id) for _id in map_api.get_incoming_lane_ids(id)],
            SD.EXIT: [convert_id_to_int(_id) for _id in map_api.get_outgoing_lane_ids(id)],
            SD.LEFT_NEIGHBORS: [],
            SD.RIGHT_NEIGHBORS: [],
        }

    # intersection lane
    for id in map_objs["lane_connector"]:
        lane_info = map_api.get("lane_connector", id)
        assert lane_info["token"] == id
        # boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        # boundary_polygon = [[boundary[0][i], boundary[1][i], 0.1] for i in range(len(boundary[0]))]
        # boundary_polygon += [[boundary[0][i], boundary[1][i], 0.] for i in range(len(boundary[0]))]

        ret[convert_id_to_int(id)] = {
            SD.TYPE: ScenarioType.LANE_SURFACE_UNSTRUCTURE,
            SD.POLYLINE: np.asarray(discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            # SD.POLYGON: boundary_polygon,
            "speed_limit_kmh": 100,
            SD.ENTRY: [convert_id_to_int(_id) for _id in map_api.get_incoming_lane_ids(id)],
            SD.EXIT: [convert_id_to_int(_id) for _id in map_api.get_outgoing_lane_ids(id)],
        }

    # # stop_line
    # for id in map_objs["stop_line"]:
    #     info = map_api.get("stop_line", id)
    #     assert info["token"] == id
    #     boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
    #     boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
    #     ret[id] = {
    #         SD.TYPE: ScenarioType.STOP_LINE,
    #         SD.POLYGON: boundary_polygon ,
    #     }

    #         'stop_line',
    #         'carpark_area',

    return ret


def get_tracks_from_frames(nuscenes: NuScenes, scene_info, frames, num_to_interpolate=5):
    episode_len = len(frames)
    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.keys())
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
            "metadata": {"track_length": episode_len, "type": ScenarioType.UNSET, "object_id": k, "original_id": k},
        }
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx in range(episode_len):
        # Record all agents' states (position, velocity, ...)
        for id, state in frames[frame_idx].items():
            # Fill type
            md_type, meta_type = get_scenarionet_type(state["type"])
            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type
            if md_type is None or md_type == ScenarioType.UNSET:
                tracks_to_remove.add(id)
                continue

            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type

            # Introducing the state item
            tracks[id]["state"]["position"][frame_idx] = state["position"]
            tracks[id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[id]["state"]["velocity"][frame_idx] = tracks[id]["state"]["velocity"][frame_idx]
            tracks[id]["state"]["valid"][frame_idx] = 1

            tracks[id]["state"]["length"][frame_idx] = state["size"][1]
            tracks[id]["state"]["width"][frame_idx] = state["size"][0]
            tracks[id]["state"]["height"][frame_idx] = state["size"][2]

            tracks[id]["metadata"]["original_id"] = id
            tracks[id]["metadata"]["object_id"] = id

    for track in tracks_to_remove:
        track_data = tracks.pop(track)
        obj_type = track_data[SD.METADATA]["type"]
        print(f"\nWARNING: Can not map type: {obj_type} to any Type")

    new_episode_len = (episode_len - 1) * num_to_interpolate + 1

    # interpolate
    interpolate_tracks = {}
    for (
        id,
        track,
    ) in tracks.items():
        interpolate_tracks[id] = copy.deepcopy(track)
        interpolate_tracks[id]["metadata"]["track_length"] = new_episode_len

        # valid first
        new_valid = np.zeros(shape=(new_episode_len,))
        if track["state"]["valid"][0]:
            new_valid[0] = 1
        for k, valid in enumerate(track["state"]["valid"][1:], start=1):
            if valid:
                if abs(new_valid[(k - 1) * num_to_interpolate] - 1) < 1e-2:
                    start_idx = (k - 1) * num_to_interpolate + 1
                else:
                    start_idx = k * num_to_interpolate
                new_valid[start_idx : k * num_to_interpolate + 1] = 1
        interpolate_tracks[id]["state"]["valid"] = new_valid

        # position
        interpolate_tracks[id]["state"]["position"] = interpolate(
            track["state"]["position"],
            track["state"]["valid"],
            new_valid,
        )
        if id == "ego" and not scene_info.get("prediction", False):
            assert "prediction" not in scene_info
            # We can get it from canbus
            try:
                canbus = NuScenesCanBus(dataroot=nuscenes.dataroot)
                imu_pos = np.asarray([state["pos"] for state in canbus.get_messages(scene_info["name"], "pose")[::5]])
                min_len = min(len(imu_pos), new_episode_len)
                interpolate_tracks[id]["state"]["position"][:min_len] = imu_pos[:min_len]
            except Exception as e:
                logger.info("Fail to get canbus data for {} - Error: {}".format(scene_info["name"], e))

        # velocity
        interpolate_tracks[id]["state"]["velocity"] = interpolate(
            track["state"]["velocity"],
            track["state"]["valid"],
            new_valid,
        )
        vel = interpolate_tracks[id]["state"]["position"][1:] - interpolate_tracks[id]["state"]["position"][:-1]
        interpolate_tracks[id]["state"]["velocity"][:-1] = vel[..., :2] / 0.1
        for k, valid in enumerate(new_valid[1:], start=1):
            if valid == 0 or not valid or abs(valid) < 1e-2:
                interpolate_tracks[id]["state"]["velocity"][k] = np.array([0.0, 0.0])
                interpolate_tracks[id]["state"]["velocity"][k - 1] = np.array([0.0, 0.0])
        # speed outlier check
        max_vel = np.max(np.linalg.norm(interpolate_tracks[id]["state"]["velocity"], axis=-1))
        if max_vel > 30:
            print(f"\nWARNING: Too large speed for {id}: {max_vel}")

        # heading
        # then update position
        new_heading = interpolate_heading(track["state"]["heading"], track["state"]["valid"], new_valid)
        interpolate_tracks[id]["state"]["heading"] = new_heading
        if id == "ego" and not scene_info.get("prediction", False):
            assert "prediction" not in scene_info
            # We can get it from canbus
            try:
                canbus = NuScenesCanBus(dataroot=nuscenes.dataroot)
                imu_heading = np.asarray(
                    [
                        quaternion_yaw(Quaternion(state["orientation"]))
                        for state in canbus.get_messages(scene_info["name"], "pose")[::5]
                    ],
                )
                min_len = min(len(imu_heading), new_episode_len)
                interpolate_tracks[id]["state"]["heading"][:min_len] = imu_heading[:min_len]
            except Exception as e:
                logger.info("Fail to get canbus data for {} - Error: {}".format(scene_info["name"], e))

        for k, v in track["state"].items():
            if k in ["valid", "heading", "position", "velocity"]:
                continue
            else:
                interpolate_tracks[id]["state"][k] = interpolate(v, track["state"]["valid"], new_valid)
        # if id == "ego":
        # ego is valid all time, so we can calculate the velocity in this way
    return interpolate_tracks

import logging
import re

import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from scenariomax import logger_utils


logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)


logger = logger_utils.get_logger(__name__)


EGO = "ego"


def convert_id_to_int(id):
    _id = "".join(re.findall(r"\d+", id))[:6]
    if _id[0] == "0":
        _id = _id[1:]

    return _id


def parse_frame(frame, nuscenes: NuScenes):
    ret = {}
    for obj_id in frame["anns"]:
        obj = nuscenes.get("sample_annotation", obj_id)
        # velocity = nuscenes.box_velocity(obj_id)[:2]
        # if np.nan in velocity:
        velocity = np.array([0.0, 0.0])
        ret[obj["instance_token"]] = {
            "position": obj["translation"],
            "obj_id": obj["instance_token"],
            "heading": quaternion_yaw(Quaternion(*obj["rotation"])),
            "rotation": obj["rotation"],
            "velocity": velocity,
            "size": obj["size"],
            "visible": obj["visibility_token"],
            "attribute": [nuscenes.get("attribute", i)["name"] for i in obj["attribute_tokens"]],
            "type": obj["category_name"],
        }
    ego_token = nuscenes.get("sample_data", frame["data"]["LIDAR_TOP"])["ego_pose_token"]
    ego_state = nuscenes.get("ego_pose", ego_token)
    ret[EGO] = {
        "position": ego_state["translation"],
        "obj_id": EGO,
        "heading": quaternion_yaw(Quaternion(*ego_state["rotation"])),
        "rotation": ego_state["rotation"],
        "type": "vehicle.car",
        "velocity": np.array([0.0, 0.0]),
        # size https://en.wikipedia.org/wiki/Renault_Zoe
        "size": [1.73, 4.08, 1.56],
    }
    return ret


def interpolate_heading(heading_data, old_valid, new_valid, num_to_interpolate=5):
    new_heading_theta = np.zeros_like(new_valid)
    for k, valid in enumerate(old_valid[:-1]):
        if abs(valid) > 1e-1 and abs(old_valid[k + 1]) > 1e-1:
            diff = (heading_data[k + 1] - heading_data[k] + np.pi) % (2 * np.pi) - np.pi
            # step = diff
            interpolate_heading = np.linspace(heading_data[k], heading_data[k] + diff, 6)
            new_heading_theta[k * num_to_interpolate : (k + 1) * num_to_interpolate] = interpolate_heading[:-1]
        elif abs(valid) > 1e-1 and abs(old_valid[k + 1]) < 1e-1:
            new_heading_theta[k * num_to_interpolate : (k + 1) * num_to_interpolate] = heading_data[k]
    new_heading_theta[-1] = heading_data[-1]
    return new_heading_theta * new_valid


def _interpolate_one_dim(data, old_valid, new_valid, num_to_interpolate=5):
    new_data = np.zeros_like(new_valid)
    for k, valid in enumerate(old_valid[:-1]):
        if abs(valid) > 1e-1 and abs(old_valid[k + 1]) > 1e-1:
            diff = data[k + 1] - data[k]
            # step = diff
            interpolate_data = np.linspace(data[k], data[k] + diff, num_to_interpolate + 1)
            new_data[k * num_to_interpolate : (k + 1) * num_to_interpolate] = interpolate_data[:-1]
        elif abs(valid) > 1e-1 and abs(old_valid[k + 1]) < 1e-1:
            new_data[k * num_to_interpolate : (k + 1) * num_to_interpolate] = data[k]
    new_data[-1] = data[-1]
    return new_data * new_valid


def interpolate(origin_y, valid, new_valid):
    if len(origin_y.shape) == 1:
        ret = _interpolate_one_dim(origin_y, valid, new_valid)
    elif len(origin_y.shape) == 2:
        ret = []
        for dim in range(origin_y.shape[-1]):
            new_y = _interpolate_one_dim(origin_y[..., dim], valid, new_valid)
            new_y = np.expand_dims(new_y, axis=-1)
            ret.append(new_y)
        ret = np.concatenate(ret, axis=-1)
    else:
        raise ValueError(f"Y has shape {origin_y.shape}, Can not interpolate")
    return ret


def extract_frames_scene_info(scene, nuscenes):
    scene_token = scene["token"]
    scene_info = nuscenes.get("scene", scene_token)
    frames = []
    current_frame = nuscenes.get("sample", scene_info["first_sample_token"])
    while current_frame["token"] != scene_info["last_sample_token"]:
        frames.append(parse_frame(current_frame, nuscenes))
        current_frame = nuscenes.get("sample", current_frame["next"])
    frames.append(parse_frame(current_frame, nuscenes))
    assert current_frame["next"] == ""
    assert len(frames) == scene_info["nbr_samples"], "Number of sample mismatches! "
    return frames, scene_info

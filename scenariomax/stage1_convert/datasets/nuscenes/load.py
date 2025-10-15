from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split


def get_nuscenes_scenarios(dataroot, version, num_workers=2):
    nusc = NuScenes(version=version, dataroot=dataroot)

    def _get_nusc():
        return NuScenes(version=version, dataroot=dataroot)

    return nusc.scene, [nusc for _ in range(num_workers)]


def get_nuscenes_prediction_split(dataroot, version, past, future, num_workers=2):
    def _get_nusc():
        return NuScenes(version="v1.0-mini" if "mini" in version else "v1.0-trainval", dataroot=dataroot)

    nusc = _get_nusc()
    return get_prediction_challenge_split(version, dataroot=dataroot), [nusc for _ in range(num_workers)]

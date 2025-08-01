import os

from scenariomax.tf_utils import create_tf_feature_functions


# Create feature functions using centralized TensorFlow utilities
bytes_feature, float_feature, int64_feature = create_tf_feature_functions()


def list_files(dir):
    r = []
    for root, _, files in os.walk(dir):
        for name in files:
            if name.endswith(".pkl"):
                r.append(os.path.join(root, name))
    return r


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

import os

import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


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

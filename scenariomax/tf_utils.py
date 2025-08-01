"""
Centralized TensorFlow utilities with lazy loading and optimized configuration.
This module provides a single point for TensorFlow imports with proper logging suppression.
"""

import logging
import os
import warnings
from typing import Any


_tensorflow_module = None


def get_tensorflow() -> Any:
    """
    Lazy import TensorFlow with optimized configuration and logging suppression.

    Returns:
        TensorFlow module with optimized settings

    Raises:
        RuntimeError: If TensorFlow is not installed
    """
    global _tensorflow_module

    if _tensorflow_module is not None:
        return _tensorflow_module

    try:
        # Suppress TensorFlow verbose logging before import
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL level only
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations messages
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU by default for data processing

        # Suppress warnings during import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Set logging levels before import
            logging.getLogger("tensorflow").setLevel(logging.FATAL)
            logging.getLogger("absl").setLevel(logging.FATAL)

            import tensorflow as tf

            # Additional suppression after import
            tf.get_logger().setLevel(logging.FATAL)
            tf.autograph.set_verbosity(0)

            # Disable GPU for data processing workloads
            tf.config.set_visible_devices([], "GPU")

            _tensorflow_module = tf
            return tf

    except ImportError as e:
        raise RuntimeError(
            "TensorFlow is required for TFRecord operations. Please install TensorFlow: pip install tensorflow",
        ) from e


def create_tf_feature_functions():
    """
    Create TensorFlow feature utility functions.

    Returns:
        Tuple of (bytes_feature, float_feature, int64_feature) functions
    """
    tf = get_tensorflow()

    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(value):
        """Returns a float_list feature."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def int64_feature(value):
        """Returns an int64_list feature."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    return bytes_feature, float_feature, int64_feature

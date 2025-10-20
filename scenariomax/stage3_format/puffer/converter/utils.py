"""
Utility functions for Puffer format conversion.
"""

import numpy as np

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def convert_numpy_to_json(obj):
    """
    Recursively convert numpy arrays and types to JSON-serializable formats.

    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj

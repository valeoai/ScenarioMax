import json
import pickle

import numpy as np

from scenariomax import logger_utils
from scenariomax.core.types import FORMAT_JSON, FORMAT_PUFFER, FORMAT_TFEXAMPLE


logger = logger_utils.get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy arrays and types efficiently.

    This encoder converts numpy types directly during JSON serialization,
    avoiding the need to create intermediate Python lists, which significantly
    reduces memory usage for large scenarios.
    """

    def default(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_format_function(format: str):
    # Create format function
    if format == FORMAT_TFEXAMPLE:
        from scenariomax.stage3_format.tfexample import convert_to_tfexample

        return convert_to_tfexample.convert
    elif format == FORMAT_JSON:
        from scenariomax.stage3_format.json import convert_to_json

        return convert_to_json.convert
    elif format == FORMAT_PUFFER:
        from scenariomax.stage3_format.puffer import convert_to_puffer

        return convert_to_puffer.convert


def load_pickle(file_path):
    """Load a pickle file from the specified path."""

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path):
    """Save data to a pickle file at the specified path."""

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

import json

import numpy as np

from scenariomax import logger_utils


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

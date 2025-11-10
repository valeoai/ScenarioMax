import json
import os
import pickle
import shutil

import numpy as np

from scenariomax.core.types import FORMAT_GPUDRIVE, FORMAT_PUFFERDRIVE, FORMAT_WAYMAX


def clean_and_create_output_directory(output_path: str) -> None:
    """Remove output directory if it exists, then create fresh.

    Args:
        output_path: Path to output directory

    Raises:
        ValueError: If path is a dangerous system directory
    """
    # Convert to absolute path for safety checks
    abs_path = os.path.abspath(output_path)

    # Dangerous paths that should never be deleted
    dangerous_paths = {
        "/",
        "/usr",
        "/etc",
        "/var",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/boot",
        "/sys",
        "/proc",
        "/dev",
    }

    # Check if path is or starts with a dangerous directory
    if abs_path in dangerous_paths or any(
        abs_path.startswith(d + os.sep)
        for d in ["/usr", "/etc", "/var", "/bin", "/sbin", "/lib", "/boot", "/sys", "/proc", "/dev"]
        if abs_path == d
    ):  # noqa: E501
        raise ValueError(f"Refusing to delete dangerous system directory: {abs_path}")

    # Additional check: path should have at least 2 components beyond root
    path_parts = abs_path.split(os.sep)
    if len([p for p in path_parts if p]) < 2:
        raise ValueError(f"Output path too close to root directory: {abs_path}")

    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)
    os.makedirs(abs_path, exist_ok=True)


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
    if format == FORMAT_WAYMAX:
        from scenariomax.stage3_format.waymax import convert_to_waymax

        return convert_to_waymax.convert
    elif format == FORMAT_GPUDRIVE:
        from scenariomax.stage3_format.gpudrive import convert_to_gpudrive

        return convert_to_gpudrive.convert
    elif format == FORMAT_PUFFERDRIVE:
        from scenariomax.stage3_format.pufferdrive import convert_to_pufferdrive

        return convert_to_pufferdrive.convert


def load_pickle(file_path):
    """Load a pickle file from the specified path.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded data from pickle file

    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If pickle file is corrupted
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to load pickle file {file_path}: {e}")


def save_pickle(data, file_path):
    """Save data to a pickle file at the specified path.

    Uses pickle protocol 4 for compatibility with Python 3.4+

    Args:
        data: Data to save
        file_path: Path to save pickle file

    Raises:
        pickle.PicklingError: If data cannot be pickled
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
    except pickle.PicklingError as e:
        raise pickle.PicklingError(f"Failed to save pickle file {file_path}: {e}")

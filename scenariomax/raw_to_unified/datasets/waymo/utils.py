import numpy as np


SPLIT_KEY = "|"


def convert_values_to_str(data_dict: dict) -> None:
    """Convert all values in the dictionary to strings in-place."""
    for key in data_dict:
        data_dict[key] = str(data_dict[key])

    return data_dict


def compute_polygon(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]

    coord = np.stack((x, y, z), axis=1).astype("float32")

    return coord

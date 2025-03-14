import numpy as np

from scenariomax.unified_to_tfrecord.converter.datatypes import State


def get_object_trajectory(state: State, index: int = 0):
    sdc_past_x = state.past_x[index]
    sdc_past_y = state.past_y[index]
    sdc_past_z = state.past_z[index]

    sdc_current_x = state.current_x[index]
    sdc_current_y = state.current_y[index]
    sdc_current_z = state.current_z[index]

    sdc_future_x = state.future_x[index]
    sdc_future_y = state.future_y[index]
    sdc_future_z = state.future_z[index]

    trajectory = np.vstack(
        [
            np.hstack([sdc_past_x, sdc_current_x, sdc_future_x]),
            np.hstack([sdc_past_y, sdc_current_y, sdc_future_y]),
            np.hstack([sdc_past_z, sdc_current_z, sdc_future_z]),
        ],
    )

    return np.swapaxes(trajectory, 0, 1)


def get_object_heading(state: State, index: int = 0):
    return np.concatenate(
        [
            state.past_bbox_yaw[index],
            state.current_bbox_yaw[None, index],
            state.future_bbox_yaw[index],
        ],
    )

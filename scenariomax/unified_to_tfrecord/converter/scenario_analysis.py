import numpy as np

from scenariomax.unified_to_tfrecord.converter.datatypes import State


def compute_variations(state: State):
    dt = 0.1  # Hard-coded because Valou

    sdc_past_vx = state.past_velocity_x[0]
    sdc_past_vy = state.past_velocity_y[0]

    sdc_current_vx = state.current_velocity_x[0]
    sdc_current_vy = state.current_velocity_y[0]

    sdc_future_vx = state.future_velocity_x[0]
    sdc_future_vy = state.future_velocity_y[0]

    velocity_xy = np.vstack(
        [
            np.hstack([sdc_past_vx, sdc_current_vx, sdc_future_vx]),
            np.hstack([sdc_past_vy, sdc_current_vy, sdc_future_vy]),
        ],
    )

    velocity_xy[np.abs(velocity_xy) < 0.4] = 0
    acceleration_xy = velocity_xy[:, 1:] - velocity_xy[:, :-1]
    acceleration_xy = acceleration_xy / dt
    acceleration_xy_std = np.std(acceleration_xy.flatten())

    sdc_past_yaw = state.past_bbox_yaw[0]
    sdc_current_yaw = state.current_bbox_yaw[0]
    sdc_future_yaw = state.future_bbox_yaw[0]

    yaw = np.abs(np.hstack([sdc_past_yaw, sdc_current_yaw, sdc_future_yaw]))

    velocity_yaw = yaw[1:] - yaw[:-1]
    velocity_yaw = velocity_yaw / dt
    yaw_difficulty = (velocity_yaw**2).sum()

    return acceleration_xy_std, yaw_difficulty


class TrajectoryType:
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array(
        [
            [np.cos(-start_heading), -np.sin(-start_heading)],
            [np.sin(-start_heading), np.cos(-start_heading)],
        ],
    )
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY

    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            return TrajectoryType.STRAIGHT

        return TrajectoryType.STRAIGHT_RIGHT if dy < 0 else TrajectoryType.STRAIGHT_LEFT

    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return (
            TrajectoryType.RIGHT_U_TURN
            if normalized_delta[0] < kMinLongitudinalDisplacementForUTurn
            else TrajectoryType.RIGHT_TURN
        )

    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN

    return TrajectoryType.LEFT_TURN


def classify_scenario(state: State):
    start_point = state.past_x[0][0], state.past_y[0][0]
    end_point = state.future_x[0][-1], state.future_y[0][-1]

    start_velocity = state.past_velocity_x[0][0], state.past_velocity_y[0][0]
    end_velocity = state.future_velocity_x[0][-1], state.future_velocity_y[0][-1]

    start_heading = state.past_bbox_yaw[0][0]
    end_heading = state.future_bbox_yaw[0][-1]

    trajectory_type = classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading)

    return trajectory_type

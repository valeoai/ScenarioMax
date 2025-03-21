import matplotlib.pyplot as plt

from scenariomax.logger_utils import get_logger
from scenariomax.unified_to_tfexample.converter.path import get_scenario_paths
from scenariomax.unified_to_tfexample.converter.roadgraph import get_scenario_map_points
from scenariomax.unified_to_tfexample.converter.scenario_analysis import classify_scenario, compute_variations
from scenariomax.unified_to_tfexample.converter.state import get_state
from scenariomax.unified_to_tfexample.converter.traffic_light import get_traffic_lights_state
from scenariomax.unified_to_tfexample.converter.utils import get_object_trajectory
from scenariomax.unified_to_tfexample.utils import AttrDict, bytes_feature, float_feature, int64_feature


logger = get_logger(__name__)


def build_tfexample(scenario_net_scene, multiagents=False, debug=False):
    """
    Build a TF Example from a scenario.

    Args:
        scenario_net_scene: The scenario to process
        multiagents: Whether to use multi-agent processing
        debug: Whether to print debug information and create plots

    Returns:
        Dictionary containing TF features
    """
    if debug:
        logger.debug(f"Building TF Example for scenario: {getattr(scenario_net_scene, 'id', 'unknown')}")

    scenario = AttrDict(scenario_net_scene)

    state = get_state(scenario, multiagents, debug)

    trajectory_type = classify_scenario(state)
    a_xy, d_yaw = compute_variations(state)

    if debug:
        logger.debug(f"Trajectory type: {trajectory_type}, acceleration: {a_xy:.4f}, yaw: {d_yaw:.4f}")

    tf_state = get_traffic_lights_state(scenario)

    roadgraph_samples, num_roadgraph_points, cropped = get_scenario_map_points(scenario, debug)

    if debug:
        logger.debug(f"Number of roadmap points: {num_roadgraph_points} - CROPPED: {cropped}")

    path_samples = get_scenario_paths(scenario, state, roadgraph_samples, multiagents, debug)

    if debug:
        # Plot the logged trajectory and the kalman estimate
        ax = plt.gca()

        # Plot SDC trajectory
        trajectory = get_object_trajectory(state)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "r",
            label=f"a_xy: {a_xy}, v_yaw: {d_yaw}",
        )

    return {
        # Scenario metadata
        "scenario/id": bytes_feature(scenario.id.encode()),
        # "scenario/trajectory_type": int64_feature(trajectory_type),
        # "scenario/yaw_difficulty": float_feature(d_yaw),
        # "scenario/acceleration_difficulty": float_feature(a_xy),
        # State
        "state/tracks_to_predict": int64_feature(state.tracks_to_predict.flatten()),
        "state/objects_of_interest": int64_feature(state.objects_of_interest.flatten()),
        "state/id": float_feature(state.id.flatten()),
        "state/is_sdc": int64_feature(state.is_sdc.flatten()),
        "state/type": float_feature(state.type.flatten()),
        "state/current/x": float_feature(state.current_x.flatten()),
        "state/current/y": float_feature(state.current_y.flatten()),
        "state/current/z": float_feature(state.current_z.flatten()),
        "state/current/bbox_yaw": float_feature(state.current_bbox_yaw.flatten()),
        "state/current/length": float_feature(state.current_length.flatten()),
        "state/current/width": float_feature(state.current_width.flatten()),
        "state/current/height": float_feature(state.current_height.flatten()),
        "state/current/speed": float_feature(state.current_speed.flatten()),
        "state/current/timestamp_micros": int64_feature(state.current_timestamps.flatten()),
        "state/current/vel_yaw": float_feature(state.current_vel_yaw.flatten()),
        "state/current/velocity_x": float_feature(state.current_velocity_x.flatten()),
        "state/current/velocity_y": float_feature(state.current_velocity_y.flatten()),
        "state/current/valid": int64_feature(state.current_valid.flatten()),
        "state/future/x": float_feature(state.future_x.flatten()),
        "state/future/y": float_feature(state.future_y.flatten()),
        "state/future/z": float_feature(state.future_z.flatten()),
        "state/future/bbox_yaw": float_feature(state.future_bbox_yaw.flatten()),
        "state/future/length": float_feature(state.future_length.flatten()),
        "state/future/width": float_feature(state.future_width.flatten()),
        "state/future/height": float_feature(state.future_height.flatten()),
        "state/future/speed": float_feature(state.future_speed.flatten()),
        "state/future/timestamp_micros": int64_feature(state.future_timestamps.flatten()),
        "state/future/vel_yaw": float_feature(state.future_vel_yaw.flatten()),
        "state/future/velocity_x": float_feature(state.future_velocity_x.flatten()),
        "state/future/velocity_y": float_feature(state.future_velocity_y.flatten()),
        "state/future/valid": int64_feature(state.future_valid.flatten()),
        "state/past/x": float_feature(state.past_x.flatten()),
        "state/past/y": float_feature(state.past_y.flatten()),
        "state/past/z": float_feature(state.past_z.flatten()),
        "state/past/bbox_yaw": float_feature(state.past_bbox_yaw.flatten()),
        "state/past/length": float_feature(state.past_length.flatten()),
        "state/past/width": float_feature(state.past_width.flatten()),
        "state/past/height": float_feature(state.past_height.flatten()),
        "state/past/speed": float_feature(state.past_speed.flatten()),
        "state/past/timestamp_micros": int64_feature(state.past_timestamps.flatten()),
        "state/past/vel_yaw": float_feature(state.past_vel_yaw.flatten()),
        "state/past/velocity_x": float_feature(state.past_velocity_x.flatten()),
        "state/past/velocity_y": float_feature(state.past_velocity_y.flatten()),
        "state/past/valid": int64_feature(state.past_valid.flatten()),
        # Traffic light state
        "traffic_light_state/current/state": int64_feature(tf_state.current_state.flatten()),
        "traffic_light_state/current/x": float_feature(tf_state.current_x.flatten()),
        "traffic_light_state/current/y": float_feature(tf_state.current_y.flatten()),
        "traffic_light_state/current/z": float_feature(tf_state.current_z.flatten()),
        "traffic_light_state/current/id": int64_feature(tf_state.current_id.flatten()),
        "traffic_light_state/current/valid": int64_feature(tf_state.current_valid.flatten()),
        "traffic_light_state/current/timestamp_micros": int64_feature(tf_state.current_timestamps.flatten()),
        "traffic_light_state/future/state": int64_feature(tf_state.future_state.flatten()),
        "traffic_light_state/future/x": float_feature(tf_state.future_x.flatten()),
        "traffic_light_state/future/y": float_feature(tf_state.future_y.flatten()),
        "traffic_light_state/future/z": float_feature(tf_state.future_z.flatten()),
        "traffic_light_state/future/id": int64_feature(tf_state.future_id.flatten()),
        "traffic_light_state/future/valid": int64_feature(tf_state.future_valid.flatten()),
        "traffic_light_state/future/timestamp_micros": int64_feature(tf_state.future_timestamps.flatten()),
        "traffic_light_state/past/state": int64_feature(tf_state.past_state.flatten()),
        "traffic_light_state/past/x": float_feature(tf_state.past_x.flatten()),
        "traffic_light_state/past/y": float_feature(tf_state.past_y.flatten()),
        "traffic_light_state/past/z": float_feature(tf_state.past_z.flatten()),
        "traffic_light_state/past/id": int64_feature(tf_state.past_id.flatten()),
        "traffic_light_state/past/valid": int64_feature(tf_state.past_valid.flatten()),
        "traffic_light_state/past/timestamp_micros": int64_feature(tf_state.past_timestamps.flatten()),
        # Roadgraph samples
        "roadgraph_samples/id": int64_feature(roadgraph_samples.id.flatten()),
        "roadgraph_samples/dir": float_feature(roadgraph_samples.dir.flatten()),
        "roadgraph_samples/valid": int64_feature(roadgraph_samples.valid.flatten()),
        "roadgraph_samples/xyz": float_feature(roadgraph_samples.xyz.flatten()),
        "roadgraph_samples/type": int64_feature(roadgraph_samples.type.flatten()),
        # Path samples
        "path_samples/xyz": float_feature(path_samples.xyz.flatten()),
        "path_samples/valid": int64_feature(path_samples.valid.flatten()),
        "path_samples/id": int64_feature(path_samples.id.flatten()),
        "path_samples/arc_length": float_feature(path_samples.arc_length.flatten()),
        "path_samples/on_route": int64_feature(path_samples.on_route.flatten()),
    }

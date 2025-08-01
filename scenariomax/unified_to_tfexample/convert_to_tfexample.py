from scenariomax import logger_utils
from scenariomax.unified_to_tfexample import utils
from scenariomax.unified_to_tfexample.converter import (
    path,
    roadgraph,
    scenario_analysis,
    state,
    traffic_light,
)
from scenariomax.unified_to_tfexample.converter import utils as converter_utils


logger = logger_utils.get_logger(__name__)


def convert(scenario_net_scene, multiagents=False, debug=False):
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

    scenario = utils.AttrDict(scenario_net_scene)
    roadgraph_samples, num_roadgraph_points, cropped = roadgraph.get_scenario_map_points(scenario, debug)

    state_obj = state.get_state(scenario, multiagents, roadgraph_samples, debug)

    trajectory_type = scenario_analysis.classify_scenario(state_obj)
    a_xy, d_yaw = scenario_analysis.compute_variations(state_obj)

    if debug:
        logger.debug(f"Trajectory type: {trajectory_type}, acceleration: {a_xy:.4f}, yaw: {d_yaw:.4f}")

    tf_state = traffic_light.get_traffic_lights_state(scenario, debug)

    if debug:
        logger.debug(f"Number of roadmap points: {num_roadgraph_points} - CROPPED: {cropped}")

    path_samples = path.get_scenario_paths(scenario, state_obj, roadgraph_samples, multiagents, debug)

    if debug:
        # Plot the logged trajectory and the kalman estimate
        import matplotlib.pyplot as plt

        ax = plt.gca()

        # Plot SDC trajectory
        trajectory = converter_utils.get_object_trajectory(state_obj)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "r",
            label=f"a_xy: {a_xy}, v_yaw: {d_yaw}",
        )

    return {
        # Scenario metadata
        "scenario/id": utils.bytes_feature(scenario.id.encode()),
        # "scenario/trajectory_type": int64_feature(trajectory_type),
        # "scenario/yaw_difficulty": float_feature(d_yaw),
        # "scenario/acceleration_difficulty": float_feature(a_xy),
        # State
        "state/tracks_to_predict": utils.int64_feature(state_obj.tracks_to_predict.flatten()),
        "state/objects_of_interest": utils.int64_feature(state_obj.objects_of_interest.flatten()),
        "state/id": utils.float_feature(state_obj.id.flatten()),
        "state/is_sdc": utils.int64_feature(state_obj.is_sdc.flatten()),
        "state/type": utils.float_feature(state_obj.type.flatten()),
        "state/current/x": utils.float_feature(state_obj.current_x.flatten()),
        "state/current/y": utils.float_feature(state_obj.current_y.flatten()),
        "state/current/z": utils.float_feature(state_obj.current_z.flatten()),
        "state/current/bbox_yaw": utils.float_feature(state_obj.current_bbox_yaw.flatten()),
        "state/current/length": utils.float_feature(state_obj.current_length.flatten()),
        "state/current/width": utils.float_feature(state_obj.current_width.flatten()),
        "state/current/height": utils.float_feature(state_obj.current_height.flatten()),
        "state/current/speed": utils.float_feature(state_obj.current_speed.flatten()),
        "state/current/timestamp_micros": utils.int64_feature(state_obj.current_timestamps.flatten()),
        "state/current/vel_yaw": utils.float_feature(state_obj.current_vel_yaw.flatten()),
        "state/current/velocity_x": utils.float_feature(state_obj.current_velocity_x.flatten()),
        "state/current/velocity_y": utils.float_feature(state_obj.current_velocity_y.flatten()),
        "state/current/valid": utils.int64_feature(state_obj.current_valid.flatten()),
        "state/future/x": utils.float_feature(state_obj.future_x.flatten()),
        "state/future/y": utils.float_feature(state_obj.future_y.flatten()),
        "state/future/z": utils.float_feature(state_obj.future_z.flatten()),
        "state/future/bbox_yaw": utils.float_feature(state_obj.future_bbox_yaw.flatten()),
        "state/future/length": utils.float_feature(state_obj.future_length.flatten()),
        "state/future/width": utils.float_feature(state_obj.future_width.flatten()),
        "state/future/height": utils.float_feature(state_obj.future_height.flatten()),
        "state/future/speed": utils.float_feature(state_obj.future_speed.flatten()),
        "state/future/timestamp_micros": utils.int64_feature(state_obj.future_timestamps.flatten()),
        "state/future/vel_yaw": utils.float_feature(state_obj.future_vel_yaw.flatten()),
        "state/future/velocity_x": utils.float_feature(state_obj.future_velocity_x.flatten()),
        "state/future/velocity_y": utils.float_feature(state_obj.future_velocity_y.flatten()),
        "state/future/valid": utils.int64_feature(state_obj.future_valid.flatten()),
        "state/past/x": utils.float_feature(state_obj.past_x.flatten()),
        "state/past/y": utils.float_feature(state_obj.past_y.flatten()),
        "state/past/z": utils.float_feature(state_obj.past_z.flatten()),
        "state/past/bbox_yaw": utils.float_feature(state_obj.past_bbox_yaw.flatten()),
        "state/past/length": utils.float_feature(state_obj.past_length.flatten()),
        "state/past/width": utils.float_feature(state_obj.past_width.flatten()),
        "state/past/height": utils.float_feature(state_obj.past_height.flatten()),
        "state/past/speed": utils.float_feature(state_obj.past_speed.flatten()),
        "state/past/timestamp_micros": utils.int64_feature(state_obj.past_timestamps.flatten()),
        "state/past/vel_yaw": utils.float_feature(state_obj.past_vel_yaw.flatten()),
        "state/past/velocity_x": utils.float_feature(state_obj.past_velocity_x.flatten()),
        "state/past/velocity_y": utils.float_feature(state_obj.past_velocity_y.flatten()),
        "state/past/valid": utils.int64_feature(state_obj.past_valid.flatten()),
        # Traffic light state
        "traffic_light_state/current/state": utils.int64_feature(tf_state.current_state.flatten()),
        "traffic_light_state/current/x": utils.float_feature(tf_state.current_x.flatten()),
        "traffic_light_state/current/y": utils.float_feature(tf_state.current_y.flatten()),
        "traffic_light_state/current/z": utils.float_feature(tf_state.current_z.flatten()),
        "traffic_light_state/current/id": utils.int64_feature(tf_state.current_id.flatten()),
        "traffic_light_state/current/valid": utils.int64_feature(tf_state.current_valid.flatten()),
        "traffic_light_state/current/timestamp_micros": utils.int64_feature(tf_state.current_timestamps.flatten()),
        "traffic_light_state/future/state": utils.int64_feature(tf_state.future_state.flatten()),
        "traffic_light_state/future/x": utils.float_feature(tf_state.future_x.flatten()),
        "traffic_light_state/future/y": utils.float_feature(tf_state.future_y.flatten()),
        "traffic_light_state/future/z": utils.float_feature(tf_state.future_z.flatten()),
        "traffic_light_state/future/id": utils.int64_feature(tf_state.future_id.flatten()),
        "traffic_light_state/future/valid": utils.int64_feature(tf_state.future_valid.flatten()),
        "traffic_light_state/future/timestamp_micros": utils.int64_feature(tf_state.future_timestamps.flatten()),
        "traffic_light_state/past/state": utils.int64_feature(tf_state.past_state.flatten()),
        "traffic_light_state/past/x": utils.float_feature(tf_state.past_x.flatten()),
        "traffic_light_state/past/y": utils.float_feature(tf_state.past_y.flatten()),
        "traffic_light_state/past/z": utils.float_feature(tf_state.past_z.flatten()),
        "traffic_light_state/past/id": utils.int64_feature(tf_state.past_id.flatten()),
        "traffic_light_state/past/valid": utils.int64_feature(tf_state.past_valid.flatten()),
        "traffic_light_state/past/timestamp_micros": utils.int64_feature(tf_state.past_timestamps.flatten()),
        # Roadgraph samples
        "roadgraph_samples/id": utils.int64_feature(roadgraph_samples.id.flatten()),
        "roadgraph_samples/dir": utils.float_feature(roadgraph_samples.dir.flatten()),
        "roadgraph_samples/speed_limit": utils.float_feature(roadgraph_samples.speed_limit.flatten()),
        "roadgraph_samples/valid": utils.int64_feature(roadgraph_samples.valid.flatten()),
        "roadgraph_samples/xyz": utils.float_feature(roadgraph_samples.xyz.flatten()),
        "roadgraph_samples/type": utils.int64_feature(roadgraph_samples.type.flatten()),
        # Path samples
        "path_samples/xyz": utils.float_feature(path_samples.xyz.flatten()),
        "path_samples/valid": utils.int64_feature(path_samples.valid.flatten()),
        "path_samples/id": utils.int64_feature(path_samples.id.flatten()),
        "path_samples/arc_length": utils.float_feature(path_samples.arc_length.flatten()),
        "path_samples/on_route": utils.int64_feature(path_samples.on_route.flatten()),
    }

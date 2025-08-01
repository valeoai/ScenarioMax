import geopandas as gpd
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import unary_union

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from scenariomax.core import types
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.raw_to_unified.datasets import utils as converter_utils
from scenariomax.raw_to_unified.datasets.nuplan import types as nuplan_types
from scenariomax.raw_to_unified.datasets.nuplan import utils as nuplan_utils


# Constants
EGO_ID = "ego"
NUPLAN_EGO_TYPE = TrackedObjectType.EGO
SAMPLE_RATE = 0.1  # nuPlan default sample rate in seconds
DEFAULT_MAP_EXTRACTION_RADIUS_METERS = 250


def convert_nuplan_scenario(nuplan_scenario: NuPlanScenario, version: str) -> UnifiedScenario:
    """Convert nuPlan scenario to unified format."""
    # Validate sample rate
    scenario_log_interval = nuplan_scenario.database_interval
    if abs(scenario_log_interval - SAMPLE_RATE) > 1e-3:
        raise ValueError(
            f"Log interval should be {SAMPLE_RATE} s or interpolation is required! "
            f"Got {scenario_log_interval} s. Adjust NuPlan subsample ratio to address this.",
        )

    # Extract scenario ID
    scenario_id = nuplan_scenario.scenario_name

    # Create unified scenario
    scenario = UnifiedScenario(scenario_id=scenario_id, dataset_name="nuPlan", dataset_version=version)

    # Get scenario center from ego initial position
    initial_ego_state = nuplan_scenario.get_ego_state_at_iteration(0)
    scenario_center = [initial_ego_state.waypoint.x, initial_ego_state.waypoint.y]

    # Convert data
    dynamic_agents = extract_dynamic_agents(nuplan_scenario, scenario_center)
    static_map_elements = extract_static_map_elements(nuplan_scenario.map_api, scenario_center)
    dynamic_map_elements = extract_dynamic_map_elements(nuplan_scenario, scenario_center)

    # Populate unified scenario
    scenario["dynamic_agents"] = dynamic_agents
    scenario["static_map_elements"] = static_map_elements
    scenario["dynamic_map_elements"] = dynamic_map_elements

    # Add metadata
    scenario["metadata"].update(
        {
            # General metadata
            "scenario_id": scenario_id,
            "source_file": nuplan_scenario.log_name,
            "length": nuplan_scenario.get_number_of_iterations(),
            "ego_id": EGO_ID,
            "timesteps": np.array(
                [i * scenario_log_interval for i in range(nuplan_scenario.get_number_of_iterations())],
                dtype=np.float32,
            ),
            # nuPlan-specific metadata
            "map_name": nuplan_scenario.map_api.map_name,
            "map_version": nuplan_scenario.map_version,
            "scenario_extraction_info": nuplan_scenario._scenario_extraction_info.__dict__,
            "ego_vehicle_parameters": nuplan_scenario.ego_vehicle_parameters.__dict__,
            "scenario_token": nuplan_scenario.scenario_name,
            "sample_rate": scenario_log_interval,
            "initial_lidar_timestamp": nuplan_scenario._initial_lidar_timestamp,
            "scenario_type": nuplan_scenario.scenario_type,
        },
    )

    return scenario


def extract_dynamic_agents(scenario: NuPlanScenario, center: list[float]) -> dict[str, dict]:
    """
    Extract dynamic agent trajectories from nuPlan scenario.

    This function processes all tracked objects (vehicles, pedestrians, cyclists) and the ego vehicle
    to create consistent trajectory data across all simulation timesteps.

    Args:
        scenario: NuPlan scenario containing agent tracking data
        center: Reference point [x, y] for coordinate transformation (typically ego initial position)

    Returns:
        Dictionary mapping agent IDs to their trajectory data, including:
        - type: Agent type (VEHICLE, PEDESTRIAN, etc.)
        - states: Trajectory arrays (position, heading, velocity, dimensions, validity)
        - metadata: Track metadata (length, IDs, etc.)
    """
    episode_length = scenario.get_number_of_iterations()

    # Collect all tracked objects across all simulation frames
    tracked_frames = []
    all_agent_ids = set()
    all_agent_ids.add(EGO_ID)  # Always include ego vehicle

    for frame_idx in range(episode_length):
        tracked_objects = scenario.get_tracked_objects_at_iteration(frame_idx).tracked_objects
        frame_agents = {obj.track_token: obj for obj in tracked_objects}
        tracked_frames.append(frame_agents)
        all_agent_ids.update(frame_agents.keys())

    # Initialize trajectory containers for all detected agents
    agent_trajectories = {
        agent_id: {
            "type": None,
            "states": {
                "position": np.zeros((episode_length, 3), dtype=np.float32),
                "heading": np.zeros((episode_length,), dtype=np.float32),
                "velocity": np.zeros((episode_length, 2), dtype=np.float32),
                "valid": np.zeros((episode_length,), dtype=np.float32),
                "length": np.zeros((episode_length, 1), dtype=np.float32),
                "width": np.zeros((episode_length, 1), dtype=np.float32),
                "height": np.zeros((episode_length, 1), dtype=np.float32),
            },
        }
        for agent_id in all_agent_ids
    }

    # Track agents that should be removed due to invalid or unsupported types
    invalid_agent_ids = set()

    # Process tracked objects frame by frame
    for frame_idx, detection_frame in enumerate(tracked_frames):
        for nuplan_agent_id, agent_state in detection_frame.items():
            # Validate agent state object type
            if not isinstance(agent_state, Agent | StaticObject):
                continue

            # Map nuPlan agent type to unified type system
            unified_agent_type = nuplan_types.get_agent_type(agent_state.tracked_object_type)
            if unified_agent_type is None:
                invalid_agent_ids.add(nuplan_agent_id)
                continue

            # Set agent type and metadata on first valid frame
            if agent_trajectories[nuplan_agent_id]["type"] is None:
                agent_trajectories[nuplan_agent_id]["type"] = unified_agent_type

            # Extract agent state for this frame
            agent_frame_state = _extract_object_state(agent_state, center)

            # Store trajectory data for this frame
            trajectory = agent_trajectories[nuplan_agent_id]["states"]
            trajectory["position"][frame_idx] = [
                agent_frame_state["position"][0],
                agent_frame_state["position"][1],
                0.0,  # Z-coordinate (ground level)
            ]
            trajectory["heading"][frame_idx] = agent_frame_state["heading"]
            trajectory["velocity"][frame_idx] = agent_frame_state["velocity"]
            trajectory["valid"][frame_idx] = 1.0  # Mark frame as valid
            trajectory["length"][frame_idx] = agent_frame_state["length"]
            trajectory["width"][frame_idx] = agent_frame_state["width"]
            trajectory["height"][frame_idx] = agent_frame_state["height"]

    # Remove agents with invalid or unsupported types
    for invalid_id in invalid_agent_ids:
        agent_trajectories.pop(invalid_id, None)

    # Process ego vehicle separately (more detailed state extraction)
    ego_trajectory = _extract_ego_vehicle_state_trajectory(scenario, center)
    ego_track = agent_trajectories[EGO_ID]
    ego_track["type"] = types.VEHICLE

    # Fill ego trajectory data frame by frame
    for frame_idx, ego_frame_state in enumerate(ego_trajectory):
        ego_states = ego_track["states"]
        ego_states["position"][frame_idx] = [ego_frame_state["position"][0], ego_frame_state["position"][1], 0.0]
        ego_states["valid"][frame_idx] = 1.0
        ego_states["heading"][frame_idx] = ego_frame_state["heading"]
        ego_states["length"][frame_idx] = ego_frame_state["length"]
        ego_states["width"][frame_idx] = ego_frame_state["width"]
        ego_states["height"][frame_idx] = ego_frame_state["height"]

    # Compute ego velocity from position differences (numerical differentiation)
    ego_positions = ego_track["states"]["position"]
    position_diffs = ego_positions[1:] - ego_positions[:-1]  # Frame-to-frame differences
    ego_track["states"]["velocity"][:-1] = position_diffs[..., :2] / SAMPLE_RATE  # Convert to velocity
    ego_track["states"]["velocity"][-1] = ego_track["states"]["velocity"][-2]  # Copy last velocity

    return agent_trajectories


def extract_dynamic_map_elements(nuplan_scenario: NuPlanScenario, center: list[float]) -> dict[str, dict]:
    """Extract dynamic map element data from nuPlan scenario."""
    dynamic_map_elements = {}
    episode_len = nuplan_scenario.get_number_of_iterations()

    # Collect all traffic light states across frames
    traffic_light_frames = []
    all_lane_connectors = set()

    for i in range(episode_len):
        traffic_lights = nuplan_scenario.get_traffic_light_status_at_iteration(i)
        frame_data = {str(tl.lane_connector_id): tl.status for tl in traffic_lights}
        traffic_light_frames.append(frame_data)
        all_lane_connectors.update(frame_data.keys())

    # Initialize traffic light states
    for lane_id in all_lane_connectors:
        # Get traffic light position
        position = nuplan_utils.set_light_position(nuplan_scenario, lane_id, center)

        dynamic_map_elements[lane_id] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([position[0], position[1], 0.0], dtype=np.float32),
            "states": [types.TRAFFIC_LIGHT_UNKNOWN] * episode_len,
            "lane": lane_id,
        }

    # Fill in traffic light states for each frame
    for frame_idx, frame_data in enumerate(traffic_light_frames):
        for lane_id, status in frame_data.items():
            if lane_id in dynamic_map_elements:
                unified_status = nuplan_types.get_traffic_light_state(status)
                dynamic_map_elements[lane_id]["states"][frame_idx] = unified_status

    return dynamic_map_elements


def extract_static_map_elements(
    map_api: NuPlanMap,
    center: list[float],
    radius: int = DEFAULT_MAP_EXTRACTION_RADIUS_METERS,
) -> dict[str, dict]:
    """Extract static map elements from nuPlan map."""
    static_map_elements = {}

    # Define layers to extract
    layer_names = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
    ]

    map_objects = map_api.get_proximal_map_objects(Point2D(*center), radius, layer_names)

    # Get boundary information for road lines
    try:
        boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)
    except Exception:
        boundaries = None

    # Filter stop lines (exclude turn stops)
    if SemanticMapLayer.STOP_LINE in map_objects:
        stop_lines = map_objects[SemanticMapLayer.STOP_LINE]
        map_objects[SemanticMapLayer.STOP_LINE] = [
            stop_line for stop_line in stop_lines if stop_line.stop_line_type != StopLineType.TURN_STOP
        ]

    # Process roadblocks and connectors (lanes)
    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        if layer not in map_objects:
            continue

        for block in map_objects[layer]:
            # Sort edges for roadblocks, keep original order for connectors
            edges = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )

            for index, lane_data in enumerate(edges):
                if not hasattr(lane_data, "baseline_path"):
                    continue

                # Extract lane polygon
                # Handle different geometry types that may result from complex lane shapes
                if isinstance(lane_data.polygon.boundary, MultiLineString):
                    # MultiLineString: Lane boundary consists of multiple disconnected line segments
                    # Use GeoPandas to decompose into individual LineString components
                    boundary_series = gpd.GeoSeries(lane_data.polygon.boundary).explode(index_parts=True)
                    # Find the longest segment (main boundary) by counting coordinate points
                    sizes = [len(polygon.xy[1]) for polygon in boundary_series[0]]
                    # Extract coordinates from the longest boundary segment
                    points = boundary_series[0][np.argmax(sizes)].xy
                elif isinstance(lane_data.polygon.boundary, LineString):
                    # Simple case: single continuous boundary line
                    points = lane_data.polygon.boundary.xy
                else:
                    # Skip unsupported geometry types (e.g., Point, empty geometries)
                    continue

                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = nuplan_utils.get_center_vector(polygon, center)

                # Extract lane centerline
                lane_polyline = nuplan_utils.extract_centerline(lane_data, center)

                # Get speed limit (convert from m/s to mph and km/h)
                speed_limit_mps = lane_data.speed_limit_mps
                speed_limit_mph = speed_limit_mps * 2.23694 if speed_limit_mps else None
                speed_limit_kmh = converter_utils.mph_to_kmh(speed_limit_mph) if speed_limit_mph else None

                # Create lane element
                static_map_elements[lane_data.id] = {
                    "type": types.LANE_SURFACE_STREET,
                    "polyline": lane_polyline,
                    "speed_limit_mph": speed_limit_mph,
                    "speed_limit_kmh": speed_limit_kmh,
                    "entry_lanes": [str(edge.id) for edge in lane_data.incoming_edges],
                    "exit_lanes": [str(edge.id) for edge in lane_data.outgoing_edges],
                    "left_neighbor": (
                        [str(edge.id) for edge in block.interior_edges[:index]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    "right_neighbor": (
                        [str(edge.id) for edge in block.interior_edges[index + 1 :]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    "polygon": polygon,
                }

                # Process lane boundaries (only for roadblocks)
                if layer == SemanticMapLayer.ROADBLOCK and boundaries is not None:
                    left_boundary = lane_data.left_boundary
                    right_boundary = lane_data.right_boundary
                    adjacent_edges = lane_data.adjacent_edges

                    # Only process boundaries between adjacent lanes
                    if adjacent_edges[0] and adjacent_edges[1]:
                        # Process left boundary
                        if left_boundary.id not in static_map_elements:
                            try:
                                boundary_type_id = int(
                                    boundaries.loc[[str(left_boundary.id)]]["boundary_type_fid"].iloc[0],
                                )
                                left_line_type = nuplan_types.get_road_line_type(boundary_type_id)
                            except Exception:
                                left_line_type = types.ROAD_LINE_UNKNOWN

                            if left_line_type != types.ROAD_LINE_UNKNOWN:
                                static_map_elements[left_boundary.id] = {
                                    "type": left_line_type,
                                    "polyline": nuplan_utils.get_points_from_boundary(left_boundary, center),
                                }

                        # Process right boundary
                        if right_boundary.id not in static_map_elements:
                            try:
                                boundary_type_id = int(
                                    boundaries.loc[[str(right_boundary.id)]]["boundary_type_fid"].iloc[0],
                                )
                                right_line_type = nuplan_types.get_road_line_type(boundary_type_id)
                            except Exception:
                                right_line_type = types.ROAD_LINE_UNKNOWN

                            if right_line_type != types.ROAD_LINE_UNKNOWN:
                                static_map_elements[right_boundary.id] = {
                                    "type": right_line_type,
                                    "polyline": nuplan_utils.get_points_from_boundary(right_boundary, center),
                                }
            # Store block polygon for boundary processings
            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    # Process crosswalks
    if SemanticMapLayer.CROSSWALK in map_objects:
        for crosswalk in map_objects[SemanticMapLayer.CROSSWALK]:
            # Extract crosswalk boundary geometry using same approach as lanes
            # Crosswalks can have complex shapes requiring careful geometry handling
            if isinstance(crosswalk.polygon.exterior, MultiLineString):
                # Handle fragmented crosswalk boundaries (multiple disconnected segments)
                boundary_series = gpd.GeoSeries(crosswalk.polygon.exterior).explode(index_parts=True)
                # Select the longest boundary segment as the primary crosswalk outline
                sizes = [len(polygon.xy[1]) for polygon in boundary_series[0]]
                points = boundary_series[0][np.argmax(sizes)].xy
            elif isinstance(crosswalk.polygon.exterior, LineString):
                # Standard case: continuous crosswalk boundary
                points = crosswalk.polygon.exterior.xy
            else:
                # Skip malformed or unsupported crosswalk geometries
                continue

            polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
            polygon = nuplan_utils.get_center_vector(polygon, center)

            static_map_elements[crosswalk.id] = {
                "type": types.CROSSWALK,
                "polygon": polygon,
            }

    # Process road boundaries
    # Collect all block and intersection polygons
    intersection_polygons = []
    if SemanticMapLayer.INTERSECTION in map_objects:
        intersection_polygons.extend(
            [intersection.polygon for intersection in map_objects[SemanticMapLayer.INTERSECTION]],
        )

    # Create unified boundary from all polygons
    # This section generates road edge boundaries by combining all road surfaces
    all_polygons = intersection_polygons + block_polygons
    if all_polygons:
        # Use GeoPandas unary_union to merge overlapping/adjacent polygons into unified shapes
        # This eliminates duplicate boundaries between adjacent road blocks
        boundaries = gpd.GeoSeries(unary_union(all_polygons)).boundary.explode(index_parts=True)

        # Extract individual boundary segments from the unified road surface
        for idx, boundary in enumerate(boundaries[0]):
            # Convert Shapely boundary coordinates to numpy array format
            boundary_points = np.array(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])))
            # Transform coordinates relative to scenario center and reverse point order
            # (reversal ensures consistent boundary direction)
            boundary_points = nuplan_utils.get_center_vector(boundary_points, center)[::-1]
            boundary_id = f"boundary_{idx}"

            static_map_elements[boundary_id] = {
                "type": types.ROAD_EDGE_BOUNDARY,
                "polyline": boundary_points,
            }

    return static_map_elements


def _extract_object_state(obj_state, nuplan_center):
    return {
        "position": nuplan_utils.get_center_vector([obj_state.center.x, obj_state.center.y], nuplan_center),
        "heading": obj_state.center.heading,
        "velocity": nuplan_utils.get_center_vector([obj_state.velocity.x, obj_state.velocity.y]),
        "valid": 1,
        "length": obj_state.box.length,
        "width": obj_state.box.width,
        "height": obj_state.box.height,
    }


def _extract_ego_vehicle_state(state, nuplan_center):
    return {
        "position": nuplan_utils.get_center_vector([state.waypoint.x, state.waypoint.y], nuplan_center),
        "heading": state.waypoint.heading,
        "velocity": nuplan_utils.get_center_vector([state.agent.velocity.x, state.agent.velocity.y]),
        "angular_velocity": state.dynamic_car_state.angular_velocity,
        "valid": 1,
        "length": state.agent.box.length,
        "width": state.agent.box.width,
        "height": state.agent.box.height,
    }


def _extract_ego_vehicle_state_trajectory(scenario, nuplan_center):
    data = [
        _extract_ego_vehicle_state(scenario.get_ego_state_at_iteration(i), nuplan_center)
        for i in range(scenario.get_number_of_iterations())
    ]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = nuplan_utils.compute_angular_velocity(
            initial_heading=data[i]["heading"],
            final_heading=data[i + 1]["heading"],
            dt=scenario.database_interval,
        )
    return data

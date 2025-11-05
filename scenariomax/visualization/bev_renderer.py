"""
Bird's Eye View (BEV) renderer for unified scenarios using matplotlib.

This module provides functions to visualize unified scenario data in BEV mode,
rendering agents, map elements, and trajectories.
"""

import os
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.core import types, utils


logger = logger_utils.get_logger(__name__)

# Visualization constants
# Optimized for ~30-frame trajectory history (3s @ 10fps) with multiple agents
MAX_DYNAMIC_ELEMENTS_TO_KEEP = 100

# Color scheme
COLORS = {
    "ego": "#FF0000",  # Red
    "vehicle": "#1F77B4",  # Blue
    "pedestrian": "#2CA02C",  # Green
    "cyclist": "#FF7F0E",  # Orange
    "road_line": "#808080",  # Grey
    "road_edge": "#000000",  # Black
    "lane": "#D3D3D3",  # Light gray
    "crosswalk": "#FFD700",  # Gold
    "speed_bump": "#FF69B4",  # Pink
    "stop_sign": "#FF0000",  # Red
    "traffic_light": "#00FF00",  # Green (default)
}

AGENT_TYPE_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}


def render_scenario_bev(
    scenario: dict[str, Any],
    output_path: str,
    show_trajectory: bool = True,
    figsize: tuple = (24, 24),
    dpi: int = 300,
    scatter_map: bool = False,
    follow_ego: bool = False,
    field_radius: float | None = None,
) -> None:
    """
    Render a unified scenario in Bird's Eye View (BEV) and save as PNG.

    Args:
        scenario: Unified scenario dict
        output_path: Output PNG file path
        show_trajectory: Show trajectory (default: True)
        figsize: Figure size in inches (default: (24, 24))
        dpi: Image resolution (default: 300)
        scatter_map: Render road map as scattered points instead of lines (default: False)
        follow_ego: Center view on ego vehicle (default: False)
        field_radius: Viewport radius in meters around center (default: None = auto-scale)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    # ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    metadata = scenario.get("metadata", {})
    scenario_id = metadata.get("scenario_id", "unknown")
    dataset_name = metadata.get("dataset_name", "unknown")

    ax.set_title(
        f"BEV Visualization - {dataset_name}\nScenario: {scenario_id}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Render static map elements (roads, lanes, crosswalks)
    _render_static_map(ax, scenario, scatter_map=scatter_map)

    # 2. Render traffic lights
    _render_traffic_lights(ax, scenario, timestep=0)

    # 3. Render dynamic agents (vehicles, pedestrians, cyclists)
    _render_dynamic_agents(ax, scenario, 0, show_trajectory)

    # 4. Add legend
    _add_legend(ax, scenario)

    # 5. Set view bounds (zoom or auto-scale)
    zoom_center = None
    if follow_ego:
        ego_pos = _get_ego_position(scenario, timestep=0)
        if ego_pos is not None:
            zoom_center = ego_pos
            if field_radius is None:
                field_radius = 50.0  # Default 50m radius

    if zoom_center is not None and field_radius is not None:
        x_c, y_c = zoom_center
        ax.set_xlim(x_c - field_radius, x_c + field_radius)
        ax.set_ylim(y_c - field_radius, y_c + field_radius)
    else:
        ax.autoscale()
        ax.margins(0.1)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Saved BEV visualization to {output_path}")


def _render_static_map(ax: plt.Axes, scenario: dict[str, Any], scatter_map: bool = False) -> None:
    """Render static map elements (lanes, road edges, crosswalks).

    Args:
        ax: Matplotlib axes
        scenario: Unified scenario dict
        scatter_map: If True, render road map elements as scattered points instead of lines
    """
    static_map = scenario.get("static_map_elements", {})

    for _, element in static_map.items():
        element_type = element.get("type", 0)
        if types.is_road_map_element(element_type):
            polyline = np.array(element.get("polyline", []))
            x, y = polyline[:, 0], polyline[:, 1]
        elif element_type in [types.CROSSWALK, types.SPEED_BUMP, types.DRIVEWAY]:
            polygon = np.array(element.get("polygon", []))
            x, y = polygon[:, 0], polygon[:, 1]
            # Close the polygon if not already closed
            if not np.array_equal(polygon[0], polygon[-1]):
                x = np.append(x, x[0])
                y = np.append(y, y[0])
        elif element_type == types.STOP_SIGN:
            position = np.array(element.get("position", []))
            x, y = position[0], position[1]
        else:
            logger.warning(f"Unknown static map element type: {element_type}")

        # Map element types to colors and styles
        if types.is_lane(element_type):
            if scatter_map:
                ax.scatter(x, y, color=COLORS["lane"], s=2, alpha=0.7, zorder=1)
            else:
                ax.plot(x, y, color=COLORS["lane"], linewidth=0.8, alpha=0.7, linestyle="-")
        elif types.is_road_line(element_type):
            if scatter_map:
                ax.scatter(x, y, color=COLORS["road_line"], s=2, alpha=0.7, zorder=1)
            else:
                ax.plot(x, y, color=COLORS["road_line"], linewidth=1.2, alpha=0.7, linestyle="--")
        elif types.is_road_edge(element_type):
            if scatter_map:
                ax.scatter(x, y, color=COLORS["road_edge"], s=3, alpha=1.0, zorder=1)
            else:
                ax.plot(x, y, color=COLORS["road_edge"], linewidth=1.5, alpha=1.0)
        elif element_type == types.CROSSWALK:
            if scatter_map:
                ax.scatter(x, y, color=COLORS["crosswalk"], s=4, alpha=0.6, zorder=1)
            else:
                ax.plot(x, y, color=COLORS["crosswalk"], linewidth=2.0, alpha=0.6)
        elif element_type == types.SPEED_BUMP:
            if scatter_map:
                ax.scatter(x, y, color=COLORS["speed_bump"], s=4, alpha=0.7, zorder=1)
            else:
                ax.plot(x, y, color=COLORS["speed_bump"], linewidth=2.5, alpha=0.7)
        elif element_type == types.STOP_SIGN:
            ax.scatter(x, y, color=COLORS["stop_sign"], s=20, marker="s", label="Stop Sign", zorder=10)
        elif element_type == types.DRIVEWAY:
            if scatter_map:
                ax.scatter(x, y, color="#8B4513", s=2, alpha=0.7, zorder=1)
            else:
                ax.plot(x, y, color="#8B4513", linewidth=1.5, alpha=0.7, linestyle="--")  # Brown dashed line
        else:
            logger.warning(f"Unknown static map element type: {element_type}")


def _render_traffic_lights(ax: plt.Axes, scenario: dict[str, Any], timestep: int) -> None:
    """Render traffic lights with their states.

    Args:
        ax: Matplotlib axes
        scenario: Unified scenario dict
        timestep: Current timestep for traffic light state
    """
    dynamic_map = scenario.get("dynamic_map_elements", {})

    TRAFFIC_LIGHT_COLORS = {
        types.TRAFFIC_LIGHT_UNKNOWN: "#808080",
        types.TRAFFIC_LIGHT_ARROW_RED: "#FF0000",
        types.TRAFFIC_LIGHT_ARROW_YELLOW: "#FFFF00",
        types.TRAFFIC_LIGHT_ARROW_GREEN: "#00FF00",
        types.TRAFFIC_LIGHT_RED: "#FF0000",
        types.TRAFFIC_LIGHT_YELLOW: "#FFFF00",
        types.TRAFFIC_LIGHT_GREEN: "#00FF00",
        types.TRAFFIC_LIGHT_FLASHING_RED: "#FF6600",
        types.TRAFFIC_LIGHT_FLASHING_YELLOW: "#FFFF00",
    }

    for _, element in dynamic_map.items():
        element_type = element.get("type", 0)

        # Only render traffic lights (type 1)
        if element_type != types.TRAFFIC_LIGHT:
            continue

        position = element.get("position", [])
        if len(position) < 2:
            continue

        x, y = position[0], position[1]

        # Get traffic light state at current timestep
        traffic_light_states = element.get("states", {})

        # Bounds check for timestep
        state = traffic_light_states[timestep] if timestep < len(traffic_light_states) else types.TRAFFIC_LIGHT_UNKNOWN

        # Get color based on state
        color = TRAFFIC_LIGHT_COLORS.get(state, "#808080")

        # Draw traffic light as a circle with border
        ax.add_patch(
            plt.Circle(
                (x, y),
                radius=0.6,
                alpha=0.9,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=15,
            ),
        )


def _render_dynamic_agents(
    ax: plt.Axes,
    scenario: dict[str, Any],
    timestep: int,
    show_trajectory: bool,
) -> None:
    """Render dynamic agents (vehicles, pedestrians, cyclists)."""
    dynamic_agents = scenario.get("dynamic_agents", {})
    ego_id = scenario.get("metadata", {}).get("ego_id")

    for agent_id, agent in dynamic_agents.items():
        agent_type = agent.get("type", 1)
        states = agent.get("states", {})

        # Get states for current timestep
        positions = states.get("position", [])
        headings = states.get("heading", [])
        valids = states.get("valid", [])
        lengths = states.get("length", [4.5])
        widths = states.get("width", [2.0])

        if timestep >= len(positions) or not valids[timestep]:
            continue

        is_ego = agent_id == ego_id
        color = COLORS["ego"] if is_ego else COLORS.get(AGENT_TYPE_NAMES.get(agent_type, "vehicle"), "#000000")

        # Current position and heading
        x, y = positions[timestep][:2]
        heading = headings[timestep]
        length = lengths[timestep]
        width = widths[timestep]

        if agent_type == types.VEHICLE or agent_type == types.CYCLIST or agent_type == "BIKE" or agent_type == "TRUCK":
            _draw_oriented_box(ax, x, y, heading, length, width, color, is_ego)
        elif agent_type == types.PEDESTRIAN:  # Pedestrian (circle)
            circle = plt.Circle((x, y), radius=0.5, color=color, alpha=0.8, zorder=10)
            ax.add_patch(circle)
        else:
            logger.warning(f"Unknown agent type: {agent_type} - position at ({x}, {y})")

        # Draw trajectory
        if show_trajectory:
            future_positions = positions[timestep + 1 : min(len(positions), timestep + 31)]
            future_valids = valids[timestep + 1 : min(len(valids), timestep + 31)]
            _draw_trajectory(ax, future_positions, future_valids, color, alpha=0.4)


def _draw_oriented_box(
    ax: plt.Axes,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: str,
    is_ego: bool = False,
) -> None:
    """Draw an oriented bounding box for a vehicle/cyclist."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # Box corners in local frame
    corners = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
    )

    # Rotate and translate
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners_world = corners @ rotation.T + np.array([x, y])

    # Draw box
    linewidth = 2.5 if is_ego else 1.5
    alpha = 0.9 if is_ego else 0.7
    rect = mpatches.Polygon(
        corners_world,
        closed=True,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=10,
    )
    ax.add_patch(rect)

    # Draw heading arrow
    arrow_length = length * 0.2
    dx = arrow_length * cos_h
    dy = arrow_length * sin_h
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=width * 0.5,
        head_length=length * 0.2,
        fc="white",
        ec="white",
        alpha=0.9,
        linewidth=1.5,
        zorder=11,
    )


def _draw_trajectory(
    ax: plt.Axes,
    positions: list,
    valids: list,
    color: str,
    alpha: float = 0.5,
    linestyle: str = "-",
) -> None:
    """Draw a trajectory line."""
    if len(positions) < 2:
        return

    # Filter valid positions
    valid_positions = [pos for pos, valid in zip(positions, valids) if valid]

    if len(valid_positions) < 2:
        return

    positions_array = np.array(valid_positions)
    ax.plot(
        positions_array[:, 0],
        positions_array[:, 1],
        color=color,
        alpha=alpha,
        linewidth=1.5,
        linestyle=linestyle,
        zorder=5,
    )


def _add_legend(ax: plt.Axes, scenario: dict[str, Any]) -> None:
    """Add legend to the plot."""
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["ego"], edgecolor=COLORS["ego"], label="Ego Vehicle"),
        mpatches.Patch(facecolor=COLORS["vehicle"], edgecolor=COLORS["vehicle"], label="Vehicle"),
        mpatches.Patch(facecolor=COLORS["pedestrian"], edgecolor=COLORS["pedestrian"], label="Pedestrian"),
        mpatches.Patch(facecolor=COLORS["cyclist"], edgecolor=COLORS["cyclist"], label="Cyclist"),
        mpatches.Patch(facecolor=COLORS["road_line"], edgecolor=COLORS["road_line"], label="Road Line"),
        mpatches.Patch(facecolor=COLORS["crosswalk"], edgecolor=COLORS["crosswalk"], label="Crosswalk"),
    ]

    # Add traffic light legend if present
    dynamic_map = scenario.get("dynamic_map_elements", {})
    has_traffic_lights = any(elem.get("type") == types.TRAFFIC_LIGHT for elem in dynamic_map.values())
    if has_traffic_lights:
        legend_elements.extend(
            [
                mpatches.Patch(facecolor="#FF0000", edgecolor="black", label="Traffic Light (Red)"),
                mpatches.Patch(facecolor="#FFFF00", edgecolor="black", label="Traffic Light (Yellow)"),
                mpatches.Patch(facecolor="#00FF00", edgecolor="black", label="Traffic Light (Green)"),
            ],
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)


def render_scenario_video(
    scenario: dict[str, Any],
    output_path: str,
    show_trajectory: bool = True,
    figsize: tuple = (24, 24),
    dpi: int = 150,
    fps: int = 10,
    scatter_map: bool = False,
    follow_ego: bool = False,
    field_radius: float | None = None,
) -> None:
    """
    Render a unified scenario as an animated video showing all timesteps.

    Args:
        scenario: Unified scenario dict
        output_path: Output MP4 file path
        show_trajectory: Show log trajectory (default: True)
        figsize: Figure size in inches (default: (24, 24))
        dpi: Image resolution (default: 150, lower for faster rendering)
        fps: Frames per second (default: 10)
        scatter_map: Render road map as scattered points instead of lines (default: False)
        follow_ego: Center view on ego vehicle at each timestep (default: False)
        field_radius: Viewport radius in meters around center (default: None = auto-scale or 50m if follow_ego)
    """
    import matplotlib.animation as animation

    metadata = scenario.get("metadata", {})
    scenario_id = metadata.get("scenario_id", "unknown")
    dataset_name = metadata.get("dataset_name", "unknown")
    scenario_length = metadata.get("length", 0)

    if scenario_length == 0:
        logger.warning(f"Scenario {scenario_id} has no timesteps, skipping video")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    # Render static map once (doesn't change)
    _render_static_map(ax, scenario, scatter_map=scatter_map)

    # Calculate bounds for consistent view (only if not following ego)
    if not follow_ego:
        bounds = _calculate_scenario_bounds(scenario)
        if bounds:
            x_min, x_max, y_min, y_max = bounds
            margin = max(x_max - x_min, y_max - y_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

    def update_frame(timestep):
        """Update function for animation."""
        # Clear dynamic elements (keep static map)
        for artist in ax.patches[:]:
            artist.remove()
        # Keep static map lines, remove only recent dynamic elements
        for line in ax.lines[len(ax.lines) - MAX_DYNAMIC_ELEMENTS_TO_KEEP :]:
            if line.get_zorder() >= 5:  # Remove only dynamic elements (trajectories, etc.)
                line.remove()

        # Update title
        ax.set_title(
            f"BEV Visualization - {dataset_name}\nScenario: {scenario_id} | Timestep: {timestep}/{scenario_length - 1}",
            fontsize=14,
            fontweight="bold",
        )

        # Render traffic lights at this timestep
        _render_traffic_lights(ax, scenario, timestep)

        # Render dynamic agents at this timestep
        _render_dynamic_agents(ax, scenario, timestep, show_trajectory)

        # Update view bounds if following ego
        if follow_ego:
            ego_pos = _get_ego_position(scenario, timestep)
            if ego_pos is not None:
                current_radius = field_radius if field_radius is not None else 50.0
                x_c, y_c = ego_pos
                ax.set_xlim(x_c - current_radius, x_c + current_radius)
                ax.set_ylim(y_c - current_radius, y_c + current_radius)

        return ax.patches + ax.lines

    # Create animation
    logger.info(f"Creating video for scenario {scenario_id} ({scenario_length} frames at {fps} fps)")
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=scenario_length,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    ax.autoscale()
    ax.margins(0.1)

    plt.tight_layout()

    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec="libx264")
    anim.save(output_path, writer=writer)
    plt.close(fig)

    logger.debug(f"Saved BEV video to {output_path}")


def _calculate_scenario_bounds(scenario: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Calculate bounding box for the entire scenario."""
    x_coords = []
    y_coords = []

    # Get bounds from static map
    static_map = scenario.get("static_map_elements", {})
    for element in static_map.values():
        element_type = element.get("type", 0)
        if types.is_road_map_element(element_type):
            polyline = element.get("polyline", [])
            if len(polyline) > 0:
                polyline = np.array(polyline)
                x_coords.extend(polyline[:, 0])
                y_coords.extend(polyline[:, 1])

    # Get bounds from dynamic agents
    dynamic_agents = scenario.get("dynamic_agents", {})
    for agent in dynamic_agents.values():
        states = agent.get("states", {})
        positions = states.get("position", [])
        valids = states.get("valid", [])
        for pos, valid in zip(positions, valids):
            if valid:
                x_coords.append(pos[0])
                y_coords.append(pos[1])

    if not x_coords or not y_coords:
        return None

    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def _get_ego_position(scenario: dict[str, Any], timestep: int) -> tuple | None:
    """
    Get ego vehicle position at a specific timestep.

    Args:
        scenario: UnifiedScenario dict
        timestep: Timestep index

    Returns:
        (x, y) position tuple, or None if ego not found
    """
    metadata = scenario.get("metadata", {})
    ego_id = metadata.get("ego_id")

    if ego_id is None:
        return None

    dynamic_agents = scenario.get("dynamic_agents", {})

    if ego_id in dynamic_agents:
        agent = dynamic_agents[ego_id]
        states = agent.get("states", {})
        positions = np.array(states.get("position", []))
        valid = np.array(states.get("valid", []))

        if timestep < len(positions) and valid[timestep]:
            return (positions[timestep][0], positions[timestep][1])

    return None


def visualize_scenarios(
    input_path: str,
    output_path: str,
    max_scenarios: int | None = None,
    show_trajectory: bool = True,
    output_format: str = "png",
    fps: int = 10,
    scatter_map: bool = False,
    follow_ego: bool = False,
    field_radius: float | None = None,
) -> dict[str, Any]:
    """
    Visualize multiple scenarios from a directory of pickle files.

    Args:
        input_path: Directory containing unified pickle files
        output_path: Output directory for PNG/MP4 files
        max_scenarios: Maximum number of scenarios to process (default: None = all)
        show_trajectory: Show log trajectory (default: True)
        show_future: Show trajectory future (default: True)
        output_format: Output format - "png" or "video" (default: "png")
        fps: Frames per second for video output (default: 10)
        scatter_map: Render road map as scattered points instead of lines (default: False)
        follow_ego: Center view on ego vehicle (default: False)
        field_radius: Viewport radius in meters around center (default: None = auto-scale or 50m if follow_ego)

    Returns:
        Dict with visualization statistics
    """
    import pickle

    logger.info(f"🎨 Visualizing scenarios from {input_path}")
    logger.info(f"   • Output: {output_path}")
    logger.info(f"   • Format: {output_format}")
    logger.info(f"   • Map rendering: {'scatter points' if scatter_map else 'lines'}")

    # Find all pickle files
    pickle_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    if max_scenarios:
        pickle_files = pickle_files[:max_scenarios]

    logger.info(f"Found {len(pickle_files)} scenarios to visualize")

    # Create output directory
    utils.clean_and_create_output_directory(output_path)

    for pickle_file in tqdm(pickle_files, desc="Visualizing"):
        # Load scenario
        with open(pickle_file, "rb") as f:
            scenario = pickle.load(f)

        # Generate output filename
        scenario_id = scenario.get("metadata", {}).get("scenario_id", os.path.basename(pickle_file))
        # Clean scenario_id for filename
        scenario_id = scenario_id.replace("/", "_").replace("\\", "_")

        if output_format == "png":
            # Use first timestep (timestep 0)
            output_file = os.path.join(output_path, f"{scenario_id}.png")
            # Render PNG
            render_scenario_bev(
                scenario=scenario,
                output_path=output_file,
                show_trajectory=show_trajectory,
                scatter_map=scatter_map,
                field_radius=field_radius,
            )
        elif output_format == "video":
            output_file = os.path.join(output_path, f"{scenario_id}.mp4")
            # Render video
            render_scenario_video(
                scenario=scenario,
                output_path=output_file,
                show_trajectory=show_trajectory,
                fps=fps,
                scatter_map=scatter_map,
                follow_ego=follow_ego,
                field_radius=field_radius,
            )
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    logger.info("✅ Visualization complete")

    return {}

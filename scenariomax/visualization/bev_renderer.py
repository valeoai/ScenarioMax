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
from matplotlib.collections import LineCollection
from tqdm import tqdm

from scenariomax import logger_utils
from scenariomax.core import types


logger = logger_utils.get_logger(__name__)

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
    timestep: int = 10,
    show_history: bool = True,
    show_future: bool = True,
    figsize: tuple = (24, 24),
    dpi: int = 300,
) -> None:
    """
    Render a unified scenario in Bird's Eye View (BEV) and save as PNG.

    Args:
        scenario: Unified scenario dict
        output_path: Output PNG file path
        timestep: Timestep to visualize (default: 10, middle of scenario)
        show_history: Show trajectory history (default: True)
        show_future: Show trajectory future (default: True)
        figsize: Figure size in inches (default: (24, 24))
        dpi: Image resolution (default: 300)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    # ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    metadata = scenario.get("metadata", {})
    scenario_id = metadata.get("scenario_id", "unknown")
    dataset_name = metadata.get("dataset_name", "unknown")
    total_timesteps = len(metadata.get("timesteps", 0))

    # Clamp timestep to valid range
    timestep = max(0, min(timestep, total_timesteps - 1))

    ax.set_title(
        f"BEV Visualization - {dataset_name}\nScenario: {scenario_id} | Timestep: {timestep}/{total_timesteps - 1}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Render static map elements (roads, lanes, crosswalks)
    _render_static_map(ax, scenario)

    # 2. Render dynamic agents (vehicles, pedestrians, cyclists)
    _render_dynamic_agents(ax, scenario, timestep, show_history, show_future)

    # 3. Add legend
    _add_legend(ax, scenario)

    # 4. Auto-scale view to fit all elements
    ax.autoscale()
    ax.margins(0.1)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Saved BEV visualization to {output_path}")


def _render_static_map(ax: plt.Axes, scenario: dict[str, Any]) -> None:
    """Render static map elements (lanes, road edges, crosswalks)."""
    static_map = scenario.get("static_map_elements", {})

    for _, element in static_map.items():
        element_type = element.get("type", 0)
        if types.is_road_map_element(element_type):
            polyline = np.array(element.get("polyline", []))
            x, y = polyline[:, 0], polyline[:, 1]
        elif element_type in [6, 9]:  # Crosswalk or Speed bump
            polygon = element.get("polygon", [])
            x, y = zip(*polygon)
            x = np.array(x + (x[0],))
            y = np.array(y + (y[0],))
        elif element_type == 8:  # Stop sign
            position = element.get("position", [])
            x, y = np.array([position[0]]), np.array([position[1]])

        if len(x) == 0:
            continue

        # Map element types to colors and styles
        if types.is_lane(element_type):
            ax.plot(x, y, color=COLORS["lane"], linewidth=0.8, alpha=0.7, linestyle="-")
        elif types.is_road_line(element_type):
            ax.plot(x, y, color=COLORS["road_line"], linewidth=1.2, alpha=0.7, linestyle="--")
        elif types.is_road_edge(element_type):
            ax.plot(x, y, color=COLORS["road_edge"], linewidth=1.5, alpha=1.0)
        elif element_type == types.CROSSWALK:
            ax.plot(x, y, color=COLORS["crosswalk"], linewidth=2.0, alpha=0.6)
        elif element_type == types.SPEED_BUMP:
            ax.plot(x, y, color=COLORS["speed_bump"], linewidth=2.5, alpha=0.7)
        elif element_type == types.STOP_SIGN:
            ax.scatter(x, y, color=COLORS["stop_sign"], s=100, marker="s", label="Stop Sign", zorder=10)
        elif element_type == types.DRIVEWAY:
            ax.plot(x, y, color="#8B4513", linewidth=1.5, alpha=0.7, linestyle="--")  # Brown dashed line
        else:
            raise ValueError(f"Unknown static map element type: {element_type}")


def _render_dynamic_agents(
    ax: plt.Axes,
    scenario: dict[str, Any],
    timestep: int,
    show_history: bool,
    show_future: bool,
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

        if timestep >= len(positions) or not valids[timestep]:
            continue

        is_ego = agent_id == ego_id
        color = COLORS["ego"] if is_ego else COLORS.get(AGENT_TYPE_NAMES.get(agent_type, "vehicle"), "#000000")

        # Current position and heading
        x, y = positions[timestep][:2]
        heading = headings[timestep]

        # Draw agent as oriented rectangle
        if agent_type == 2:  # Pedestrian (circle)
            circle = plt.Circle((x, y), radius=0.5, color=color, alpha=0.8, zorder=10)
            ax.add_patch(circle)
        else:  # Vehicle/Cyclist (oriented rectangle)
            length = 4.5 if agent_type == 1 else 2.0
            width = 2.0 if agent_type == 1 else 0.8
            _draw_oriented_box(ax, x, y, heading, length, width, color, is_ego)

        # Draw trajectory history
        if show_history and timestep > 0:
            hist_positions = positions[max(0, timestep - 10) : timestep]
            hist_valids = valids[max(0, timestep - 10) : timestep]
            _draw_trajectory(ax, hist_positions, hist_valids, color, alpha=0.4)

        # Draw trajectory future
        if show_future and timestep < len(positions) - 1:
            future_positions = positions[timestep + 1 : min(len(positions), timestep + 31)]
            future_valids = valids[timestep + 1 : min(len(valids), timestep + 31)]
            _draw_trajectory(ax, future_positions, future_valids, color, alpha=0.3, linestyle="--")


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
        corners_world, closed=True, edgecolor=color, facecolor=color, alpha=alpha, linewidth=linewidth, zorder=10
    )
    ax.add_patch(rect)

    # Draw heading arrow
    arrow_length = length * 0.6
    dx = arrow_length * cos_h
    dy = arrow_length * sin_h
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=width * 0.6,
        head_length=length * 0.3,
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

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)


def visualize_scenarios(
    input_path: str,
    output_path: str,
    timestep: int = 10,
    max_scenarios: int | None = None,
    show_history: bool = True,
    show_future: bool = True,
    num_workers: int = 1,
) -> dict[str, Any]:
    """
    Visualize multiple scenarios from a directory of pickle files.

    Args:
        input_path: Directory containing unified pickle files
        output_path: Output directory for PNG files
        timestep: Timestep to visualize (default: 10)
        max_scenarios: Maximum number of scenarios to process (default: None = all)
        show_history: Show trajectory history (default: True)
        show_future: Show trajectory future (default: True)
        num_workers: Number of parallel workers (currently unused, sequential processing)

    Returns:
        Dict with visualization statistics
    """
    import pickle

    logger.info(f"🎨 Visualizing scenarios from {input_path}")
    logger.info(f"   • Output: {output_path}")
    logger.info(f"   • Timestep: {timestep}")
    logger.info(f"   • History: {show_history}, Future: {show_future}")

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
    os.makedirs(output_path, exist_ok=True)

    # Process scenarios
    success_count = 0
    error_count = 0

    for pickle_file in tqdm(pickle_files, desc="Visualizing"):
        # Load scenario
        with open(pickle_file, "rb") as f:
            scenario = pickle.load(f)

        # Generate output filename
        scenario_id = scenario.get("metadata", {}).get("scenario_id", os.path.basename(pickle_file))
        # Clean scenario_id for filename
        scenario_id = scenario_id.replace("/", "_").replace("\\", "_")
        output_file = os.path.join(output_path, f"{scenario_id}_t{timestep}.png")

        # Render
        render_scenario_bev(
            scenario=scenario,
            output_path=output_file,
            timestep=timestep,
            show_history=show_history,
            show_future=show_future,
        )

        success_count += 1

    stats = {
        "total_scenarios": len(pickle_files),
        "success": success_count,
        "errors": error_count,
    }

    logger.info("✅ Visualization complete")
    logger.info(f"   • Success: {success_count}")
    logger.info(f"   • Errors: {error_count}")

    return stats

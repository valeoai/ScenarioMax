"""
Polyline interpolation processor for Stage 2.

This processor interpolates polylines in static_map_elements to ensure
no segment exceeds a specified maximum length.
"""

from typing import Any

from scenariomax import logger_utils
from scenariomax.stage2_process.polyline_interpolation.utils import (
    distance_based_interpolate,
    validate_polyline,
)


logger = logger_utils.get_logger(__name__)


def interpolate_polylines(
    unified_scenario: dict[str, Any],
    max_segment_length: float = 2.0,
    **kwargs,
) -> dict[str, Any]:
    """
    Interpolate polylines in static_map_elements to ensure dense representation.

    This processor adds intermediate points to polylines where segments exceed
    the maximum allowed length. Original vertices are preserved.

    Args:
        unified_scenario: UnifiedScenario dict with static_map_elements
        max_segment_length: Maximum allowed distance between consecutive points (meters)
        **kwargs: Additional arguments (ignored)

    Returns:
        Modified scenario with interpolated polylines
    """
    scenario_id = unified_scenario.get("id", "<unknown>")

    # Validate input
    if max_segment_length <= 0:
        logger.warning(
            f"Scenario {scenario_id}: max_segment_length must be positive, got {max_segment_length}. Skipping interpolation.",  # noqa: E501
        )
        return unified_scenario

    static_map_elements = unified_scenario.get("static_map_elements", {})

    if not static_map_elements:
        logger.debug(f"Scenario {scenario_id}: no static_map_elements to interpolate")
        return unified_scenario

    # Track statistics
    total_elements_processed = 0
    total_elements_with_polylines = 0
    total_points_added = 0
    elements_skipped = 0

    # Process each map element
    for element_id, element in static_map_elements.items():
        # Skip elements without polylines
        if "polyline" not in element:
            continue

        total_elements_with_polylines += 1

        polyline = element["polyline"]

        # Skip short polylines (< 2 points)
        if len(polyline) < 2:
            logger.debug(
                f"Scenario {scenario_id}, element {element_id}: polyline too short ({len(polyline)} points), skipping",
            )
            elements_skipped += 1
            continue

        # Validate original polyline
        if not validate_polyline(polyline, element_id=str(element_id)):
            logger.warning(f"Scenario {scenario_id}, element {element_id}: invalid polyline, skipping")
            elements_skipped += 1
            continue

        original_point_count = len(polyline)

        # Interpolate polyline
        try:
            interpolated_polyline = distance_based_interpolate(polyline, max_segment_length)
        except Exception as e:
            logger.error(f"Scenario {scenario_id}, element {element_id}: interpolation failed: {e}. Skipping element.")
            elements_skipped += 1
            continue

        # Validate interpolated polyline
        if not validate_polyline(interpolated_polyline, element_id=str(element_id)):
            logger.warning(
                f"Scenario {scenario_id}, element {element_id}: interpolated polyline is invalid, keeping original",
            )
            elements_skipped += 1
            continue

        # Update element with interpolated polyline
        element["polyline"] = interpolated_polyline
        points_added = len(interpolated_polyline) - original_point_count
        total_points_added += points_added
        total_elements_processed += 1

        if points_added > 0:
            logger.debug(
                f"Scenario {scenario_id}, element {element_id}: "
                f"interpolated {original_point_count} → {len(interpolated_polyline)} points "
                f"({points_added} added)",
            )

    return unified_scenario

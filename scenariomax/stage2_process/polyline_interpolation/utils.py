"""Utility functions for polyline interpolation."""

import numpy as np

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def distance_based_interpolate(polyline: np.ndarray, max_segment_length: float) -> np.ndarray:
    """
    Interpolate polyline points to ensure no segment exceeds max_segment_length.

    Args:
        polyline: Input polyline as (N, 3) array of [x, y, z] coordinates
        max_segment_length: Maximum allowed distance between consecutive points (meters)

    Returns:
        Interpolated polyline as (M, 3) array where M >= N
    """
    if len(polyline) < 2:
        return polyline

    interpolated_points = [polyline[0]]

    for i in range(len(polyline) - 1):
        current_point = polyline[i]
        next_point = polyline[i + 1]

        # Calculate distance between current and next point
        segment_vector = next_point - current_point
        segment_length = np.linalg.norm(segment_vector)

        # If segment is longer than max_segment_length, subdivide it
        if segment_length > max_segment_length:
            # Calculate number of subdivisions needed
            num_subdivisions = int(np.ceil(segment_length / max_segment_length))

            # Add intermediate points (exclude start, include end on last iteration)
            for j in range(1, num_subdivisions):
                t = j / num_subdivisions
                intermediate_point = current_point + t * segment_vector
                interpolated_points.append(intermediate_point)

        # Add the next point (always include original vertices)
        interpolated_points.append(next_point)

    return np.array(interpolated_points, dtype=polyline.dtype)


def validate_polyline(polyline: np.ndarray, element_id: str = "") -> bool:
    """
    Validate that a polyline contains valid numeric values.

    Args:
        polyline: Polyline array to validate
        element_id: Element identifier for logging

    Returns:
        True if valid, False if contains NaN/inf values
    """
    if not isinstance(polyline, np.ndarray):
        logger.warning(f"Element {element_id}: polyline is not ndarray (got {type(polyline).__name__})")
        return False

    if polyline.ndim != 2 or polyline.shape[1] != 3:
        logger.warning(f"Element {element_id}: polyline has invalid shape {polyline.shape}, expected (N, 3)")
        return False

    if np.any(np.isnan(polyline)):
        logger.warning(f"Element {element_id}: polyline contains NaN values")
        return False

    if np.any(np.isinf(polyline)):
        logger.warning(f"Element {element_id}: polyline contains infinite values")
        return False

    return True

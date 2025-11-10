"""
Stage 2 Processing Module - Simple processor registry.

Available processors:
- 'validation': Validate scenario structure and physics
- 'traffic_lights': Add/interpolate traffic light states
- 'polyline_interpolation': Interpolate polylines to ensure dense representation
- 'overpass_filtering': Detect and filter scenarios with overpasses
"""

from collections.abc import Callable
from functools import partial

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def _get_processors(
    processor_names: list[str],
    configs: dict[str, dict] | None = None,
) -> list[Callable]:
    """
    Get processor functions by name.

    Args:
        processor_names: List of processor names (e.g., ['validation', 'traffic_lights'])
        configs: Optional dict of processor-specific configurations

    Returns:
        List of processor functions that can be called with a scenario

    Example:
        processors = get_processors(['validation', 'traffic_lights'])
        for processor_fn in processors:
            scenario = processor_fn(scenario)
    """
    if configs is None:
        configs = {}

    processors = []

    for name in processor_names:
        if name == "validation":
            from scenariomax.stage2_process.validation.processor import validate_scenario

            # Get validation config and bind it using partial
            config = configs.get("validation", {})
            processor_fn = partial(validate_scenario, **config) if config else validate_scenario
            processors.append(processor_fn)

        elif name == "polyline_interpolation":
            from scenariomax.stage2_process.polyline_interpolation.processor import interpolate_polylines

            # Get config and bind it using partial
            config = configs.get("polyline_interpolation", {})
            processor_fn = partial(interpolate_polylines, **config) if config else interpolate_polylines
            processors.append(processor_fn)

        elif name == "overpass_filtering":
            from scenariomax.stage2_process.overpass_filtering.processor import detect_overpass_in_scenario

            # Get config and bind it using partial
            config = configs.get("overpass_filtering", {})
            processor_fn = partial(detect_overpass_in_scenario, **config) if config else detect_overpass_in_scenario
            processors.append(processor_fn)

        elif name == "traffic_lights":
            from scenariomax.stage2_process.traffic_lights.processor import add_traffic_lights_to_scenario

            # Get config and bind it using partial
            config = configs.get("traffic_lights", {})
            processor_fn = (
                partial(add_traffic_lights_to_scenario, **config) if config else add_traffic_lights_to_scenario
            )
            processors.append(processor_fn)
        else:
            raise ValueError(
                f"Unknown processor: {name}. Available: validation, traffic_lights, polyline_interpolation, overpass_filtering",  # noqa: E501
            )

    logger.debug(f"Loaded {len(processors)} processors: {processor_names}")
    return processors


def apply_processors(scenario, processor_names: list[str], configs: dict[str, dict] | None = None):
    """
    Apply a sequence of processors to a scenario.

    Args:
        scenario: The scenario object to process
        processor_names: List of processor names to apply
        configs: Optional dict of processor-specific configurations

    Returns:
        Processed scenario after applying all processors
    """
    processors = _get_processors(processor_names, configs)

    for processor_fn in processors:
        scenario = processor_fn(scenario)

    return scenario


# Export main function
__all__ = ["apply_processors"]

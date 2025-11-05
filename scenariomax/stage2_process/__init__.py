"""
Stage 2 Processing Module - Simple processor registry.

Available processors:
- 'validation': Validate scenario structure and physics
- 'traffic_lights': Add/interpolate traffic light states
- 'polyline_interpolation': Interpolate polylines to ensure dense representation

Usage:
    from scenariomax.stage2_process import get_processors

    # Get processors as callable functions
    processors = get_processors(['validation', 'traffic_lights'])
    for processor_fn in processors:
        scenario = processor_fn(scenario)
"""

from collections.abc import Callable

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)


def get_processors(
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

            # Get validation config
            validation_config = configs.get("validation", {})

            # Create processor function with config
            def validation_processor(scenario):
                return validate_scenario(scenario, **validation_config)

            processors.append(validation_processor)
        elif name == "polyline_interpolation":
            from scenariomax.stage2_process.polyline_interpolation.processor import interpolate_polylines

            # Get polyline_interpolation config
            pi_config = configs.get("polyline_interpolation", {})

            # Create processor function
            def polyline_processor(scenario):
                return interpolate_polylines(scenario, **pi_config)

            processors.append(polyline_processor)
        elif name == "traffic_lights":
            from scenariomax.stage2_process.traffic_lights.processor import add_traffic_lights_to_scenario

            # Get traffic_lights config (currently no options)
            tl_config = configs.get("traffic_lights", {})

            # Create processor function
            def traffic_lights_processor(scenario):
                return add_traffic_lights_to_scenario(scenario, **tl_config)

            processors.append(traffic_lights_processor)
        else:
            raise ValueError(
                f"Unknown processor: {name}. Available: validation, traffic_lights, polyline_interpolation"
            )

    logger.debug(f"Loaded {len(processors)} processors: {processor_names}")
    return processors


# Export main function
__all__ = ["get_processors"]

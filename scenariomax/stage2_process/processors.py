"""
Unified processor registry for Stage 2 processing.

This module provides a clean API for selecting and applying processors to unified scenarios.

Available processors:
- 'validation': Validate scenario structure and physics
- 'traffic_lights': Add/interpolate traffic light states

Usage:
    from scenariomax.stage2_process import get_processor

    # Get single processor
    validator = get_processor('validation', strict=True)
    scenario = validator(scenario)

    # Get multiple processors
    processors = get_processors(['validation', 'traffic_lights'])
    for processor in processors:
        scenario = processor(scenario)
"""

from collections.abc import Callable
from typing import Any

from scenariomax import logger_utils


logger = logger_utils.get_logger(__name__)

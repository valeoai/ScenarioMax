"""
Validation module for UnifiedScenario data structures.

This module provides comprehensive validation capabilities:
- Soft validation: Structural checks (keys, types, shapes)
- Strict validation: Physics-based checks (trajectory coherence, map topology)
- Processor wrapper: Integration with Stage 2 pipeline

Usage:
    # Functional API (direct validation)
    from scenariomax.stage2_process.validation import soft_validate, strict_validate
    is_valid, errors, warnings = soft_validate(scenario)
    is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

    # Processor API (pipeline integration)
    from scenariomax.stage2_process.validation import validate_scenario
    scenario = validate_scenario(scenario, strict=True)
"""

from scenariomax.stage2_process.validation.processor import validate_scenario
from scenariomax.stage2_process.validation.validate import (
    ValidationError,
    soft_validate,
    strict_validate,
)


__all__ = [
    # Core validation functions
    "soft_validate",
    "strict_validate",
    "ValidationError",
    # Processor wrapper
    "validate_scenario",
]

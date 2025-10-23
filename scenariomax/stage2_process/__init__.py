"""
Enhancement module for ScenarioMax pipeline.

This module provides functionality to enhance unified scenario data before
conversion to target formats (TFExample/GPUDrive).

Available processors:
- enhance_scenarios: No-op processor (reserved for future enhancements)
- validate_scenario: Validate scenario structure and physics (read-only)

Available validation functions (from validation module):
- soft_validate: Structural validation (keys, types, shapes)
- strict_validate: Physics-based validation (trajectory coherence, map topology)

Note: Traffic light processing and trajectory enhancements are planned for future releases.
"""

from scenariomax.stage2_process.core import enhance_scenarios
from scenariomax.stage2_process.validation import (
    ValidationError,
    soft_validate,
    strict_validate,
    validate_scenario,
)


__all__ = [
    "enhance_scenarios",
    "validate_scenario",
    "soft_validate",
    "strict_validate",
    "ValidationError",
]

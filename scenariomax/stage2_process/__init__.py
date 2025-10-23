"""
Enhancement module for ScenarioMax pipeline.

This module provides functionality to enhance unified scenario data before
conversion to target formats (TFExample/GPUDrive).

Available processors:
- enhance_scenarios: No-op processor (reserved for future enhancements)
- validate_scenario: Validate scenario structure and physics (read-only)

Note: Traffic light processing and trajectory enhancements are planned for future releases.
"""

from scenariomax.stage2_process.core import enhance_scenarios
from scenariomax.stage2_process.validate import validate_scenario


__all__ = [
    "enhance_scenarios",
    "validate_scenario",
]

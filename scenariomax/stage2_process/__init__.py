"""
Enhancement module for ScenarioMax pipeline.

This module provides functionality to enhance unified scenario data before
conversion to target formats (TFExample/GPUDrive).

Available processors:
- enhance_scenarios: Add traffic light data (legacy, uses traffic_lights module)
- validate_scenario: Validate scenario structure and physics (read-only)
- interpolate_missing_states: Fill gaps in agent trajectories
- smooth_trajectories: Smooth noisy trajectories
- fill_trajectory_gaps: Fill small gaps (convenience wrapper)
"""

from scenariomax.stage2_process.core import enhance_scenarios
from scenariomax.stage2_process.validate import validate_scenario


__all__ = [
    "enhance_scenarios",
    "validate_scenario",
]

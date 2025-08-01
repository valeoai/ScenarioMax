"""
Enhancement module for ScenarioMax pipeline.

This module provides functionality to enhance unified scenario data before
conversion to target formats (TFExample/GPUDrive).
"""

from scenariomax.enhancement.core import enhance_scenarios


__all__ = ["enhance_scenarios"]

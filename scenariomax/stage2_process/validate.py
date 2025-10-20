"""
Validation processor for UnifiedScenario.

This module provides validation functionality for Stage 2 processing.
Validation is a read-only operation - scenarios are returned unchanged.
"""

from typing import Any

from scenariomax import logger_utils
from scenariomax.core.unified_scenario import UnifiedScenario


logger = logger_utils.get_logger(__name__)


def validate_scenario(
    unified_scenario: dict[str, Any],
    strict: bool = False,
    fail_on_error: bool = False,
    validation_level: int = 2,
    speed_limit_tolerance: float = 0.12,
    position_jump_threshold: float = 50.0,
    velocity_tolerance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
) -> dict[str, Any]:
    """
    Validate UnifiedScenario structure and data.

    This is a read-only processor - the scenario is returned unchanged.
    Validation errors and warnings are logged.

    Args:
        unified_scenario: The scenario to validate
        strict: If True, perform strict validation (physics checks)
                If False, perform soft validation (structure checks only)
        fail_on_error: If True, raise exception when validation fails
        validation_level: Strictness level for strict validation (1-4)
        speed_limit_tolerance: Relative tolerance for speed limit checks
        position_jump_threshold: Max position jump between timesteps (meters)
        velocity_tolerance: Tolerance for velocity-position consistency (m/s)
        heading_tolerance_deg: Tolerance for heading-velocity alignment (degrees)

    Returns:
        The original scenario unchanged

    Raises:
        ValueError: If fail_on_error=True and validation fails

    Examples:
        # Soft validation
        scenario = validate_scenario(scenario, strict=False)

        # Strict validation with custom thresholds
        scenario = validate_scenario(
            scenario,
            strict=True,
            validation_level=3,
            position_jump_threshold=30.0
        )

        # Fail on errors
        scenario = validate_scenario(scenario, fail_on_error=True)
    """
    # Convert to UnifiedScenario if needed
    if not isinstance(unified_scenario, UnifiedScenario):
        scenario = UnifiedScenario.from_dict(unified_scenario)
    else:
        scenario = unified_scenario

    scenario_id = scenario.get("id", "unknown")

    # Perform validation
    if strict:
        is_valid, errors, warnings = scenario.strict_validate(
            validation_level=validation_level,
            speed_limit_tolerance=speed_limit_tolerance,
            position_jump_threshold=position_jump_threshold,
            velocity_tolerance=velocity_tolerance,
            heading_tolerance_deg=heading_tolerance_deg,
        )
        validation_type = "strict"
    else:
        is_valid, errors, warnings = scenario.soft_validate(strict_keys=False)
        validation_type = "soft"

    # Log results
    if is_valid:
        logger.debug(f"Scenario {scenario_id}: {validation_type} validation passed")
    else:
        logger.warning(f"Scenario {scenario_id}: {validation_type} validation FAILED with {len(errors)} error(s)")

        # Log errors
        for i, error in enumerate(errors):
            if i < 10:  # Limit to first 10 errors
                logger.warning(f"  Error: {error}")
            elif i == 10:
                logger.warning(f"  ... and {len(errors) - 10} more errors")
                break

    # Log warnings
    if warnings:
        logger.debug(f"Scenario {scenario_id}: {len(warnings)} warning(s)")
        for i, warning in enumerate(warnings):
            if i < 5:  # Limit to first 5 warnings
                logger.debug(f"  Warning: {warning}")
            elif i == 5:
                logger.debug(f"  ... and {len(warnings) - 5} more warnings")
                break

    # Fail if requested
    if fail_on_error and not is_valid:
        raise ValueError(
            f"Scenario {scenario_id} validation failed with {len(errors)} error(s). "
            f"First error: {errors[0] if errors else 'unknown'}",
        )

    # Return original scenario unchanged
    return unified_scenario

"""
Standalone validation functions for UnifiedScenario data structures.

This module provides two levels of functional validation:
1. Soft validation: Structural checks (keys, types, shapes)
2. Strict validation: Physics-based checks (trajectory coherence, map topology)

These are pure functions that take scenario dictionaries and return validation results.
"""

from typing import Any

import numpy as np

from scenariomax.core import types


class ValidationError(Exception):
    """Raised when validation fails."""


# ═══════════════════════════════════════════════════════════════════════════
# SOFT VALIDATION - Structural checks
# ═══════════════════════════════════════════════════════════════════════════

# Expected keys
_REQUIRED_TOP_LEVEL_KEYS = {"id", "dynamic_agents", "static_map_elements", "dynamic_map_elements", "metadata"}
_REQUIRED_METADATA_KEYS = {"dataset_name", "dataset_version", "length", "timesteps", "ego_id"}
_OPTIONAL_METADATA_KEYS = {
    "scenario_id",
    "source_file",
    "current_frame_index",
    "sdc_track_index",
    "objects_of_interest",
    "tracks_to_predict",
}
_REQUIRED_DYNAMIC_AGENT_KEYS = {"type", "states"}
_DYNAMIC_AGENT_STATE_KEYS = {"position", "heading", "velocity", "length", "width", "height", "valid"}
_REQUIRED_LANE_KEYS = {"type", "polyline"}
_OPTIONAL_LANE_KEYS = {
    "speed_limit_mph",
    "speed_limit_kmh",
    "entry_lanes",
    "exit_lanes",
    "left_boundaries",
    "right_boundaries",
    "left_neighbor",
    "right_neighbor",
}
_REQUIRED_MAP_ELEMENT_KEYS = {"type"}
_REQUIRED_DYNAMIC_MAP_ELEMENT_KEYS = {"type", "position", "states"}
_OPTIONAL_DYNAMIC_MAP_ELEMENT_KEYS = {"lane"}


def soft_validate(scenario: dict, strict_keys: bool = False) -> tuple[bool, list[str], list[str]]:
    """
    Perform soft validation on a UnifiedScenario.

    Checks:
    - Key existence and spelling
    - Data types
    - Array shapes (dimensions, not exact lengths)
    - Type values are valid according to types.py

    Args:
        scenario: The scenario dict to validate
        strict_keys: If True, fail on unexpected keys. If False, only warn.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    try:
        _validate_top_level(scenario, errors, warnings, strict_keys)
        _validate_metadata(scenario.get("metadata", {}), errors, warnings, strict_keys)
        _validate_dynamic_agents(scenario.get("dynamic_agents", {}), errors, warnings, strict_keys)
        _validate_static_map_elements(scenario.get("static_map_elements", {}), errors, warnings, strict_keys)
        _validate_dynamic_map_elements(scenario.get("dynamic_map_elements", {}), errors, warnings, strict_keys)
    except ValidationError as e:
        errors.append(str(e))

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def _validate_top_level(scenario: dict, errors: list, warnings: list, strict_keys: bool) -> None:
    """Validate top-level keys."""
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in scenario:
            errors.append(f"Missing required top-level key: '{key}'")

    unexpected_keys = set(scenario.keys()) - _REQUIRED_TOP_LEVEL_KEYS
    if unexpected_keys:
        msg = f"Unexpected top-level keys: {unexpected_keys}"
        (errors if strict_keys else warnings).append(msg)

    if "id" in scenario and not isinstance(scenario["id"], str):
        errors.append(f"'id' must be a string, got {type(scenario['id']).__name__}")


def _validate_metadata(metadata: dict, errors: list, warnings: list, strict_keys: bool) -> None:
    """Validate metadata section."""
    for key in _REQUIRED_METADATA_KEYS:
        if key not in metadata:
            errors.append(f"Missing required metadata key: '{key}'")

    all_valid_keys = _REQUIRED_METADATA_KEYS | _OPTIONAL_METADATA_KEYS
    unexpected_keys = set(metadata.keys()) - all_valid_keys
    if unexpected_keys:
        msg = f"Unexpected metadata keys: {unexpected_keys}"
        (errors if strict_keys else warnings).append(msg)

    if "dataset_name" in metadata and not isinstance(metadata["dataset_name"], str):
        errors.append(f"metadata['dataset_name'] must be a string, got {type(metadata['dataset_name']).__name__}")

    if "dataset_version" in metadata and not isinstance(metadata["dataset_version"], str):
        errors.append(f"metadata['dataset_version'] must be a string, got {type(metadata['dataset_version']).__name__}")

    if "length" in metadata and not isinstance(metadata["length"], int):
        errors.append(f"metadata['length'] must be an int, got {type(metadata['length']).__name__}")

    if "ego_id" in metadata and not isinstance(metadata["ego_id"], int):
        errors.append(f"metadata['ego_id'] must be an int, got {type(metadata['ego_id']).__name__}")

    if "timesteps" in metadata and not isinstance(metadata["timesteps"], (list, np.ndarray)):
        errors.append(f"metadata['timesteps'] must be a list or ndarray, got {type(metadata['timesteps']).__name__}")


def _validate_dynamic_agents(dynamic_agents: dict, errors: list, warnings: list, strict_keys: bool) -> None:
    """Validate dynamic_agents section."""
    if not isinstance(dynamic_agents, dict):
        errors.append(f"'dynamic_agents' must be a dict, got {type(dynamic_agents).__name__}")
        return

    for agent_id, agent in dynamic_agents.items():
        if not isinstance(agent_id, int):
            errors.append(f"Agent ID must be an int, got {type(agent_id).__name__}")

        if not isinstance(agent, dict):
            errors.append(f"Agent '{agent_id}' must be a dict, got {type(agent).__name__}")
            continue

        for key in _REQUIRED_DYNAMIC_AGENT_KEYS:
            if key not in agent:
                errors.append(f"Agent '{agent_id}' missing required key: '{key}'")

        if "type" in agent:
            agent_type = agent["type"]
            if not isinstance(agent_type, str):
                errors.append(f"Agent '{agent_id}' type must be a string, got {type(agent_type).__name__}")
            elif not types.is_participant(agent_type):
                errors.append(
                    f"Agent '{agent_id}' has invalid type '{agent_type}'. Must be one of: {types.PARTICIPANT_TYPES}",
                )

        if "states" in agent:
            states = agent["states"]
            if not isinstance(states, dict):
                errors.append(f"Agent '{agent_id}' states must be a dict, got {type(states).__name__}")
                continue

            unexpected_keys = set(states.keys()) - _DYNAMIC_AGENT_STATE_KEYS
            if unexpected_keys:
                msg = f"Agent '{agent_id}' has unexpected state keys: {unexpected_keys}"
                (errors if strict_keys else warnings).append(msg)

            _validate_state_array(agent_id, states, "position", 2, 3, None, errors)
            _validate_state_array(agent_id, states, "heading", 1, None, None, errors)
            _validate_state_array(agent_id, states, "velocity", 2, 2, None, errors)
            _validate_state_array(agent_id, states, "length", 1, None, None, errors)
            _validate_state_array(agent_id, states, "width", 1, None, None, errors)
            _validate_state_array(agent_id, states, "height", 1, None, None, errors)
            _validate_state_array(agent_id, states, "valid", 1, None, bool, errors)


def _validate_state_array(
    agent_id: str,
    states: dict,
    key: str,
    expected_ndim: int,
    expected_last_dim: int | None,
    expected_dtype: type | None,
    errors: list,
) -> None:
    """Validate a state array's shape and dtype."""
    if key not in states:
        return

    arr = states[key]
    if not isinstance(arr, np.ndarray):
        errors.append(f"Agent '{agent_id}' state '{key}' must be an ndarray, got {type(arr).__name__}")
        return

    if arr.ndim != expected_ndim:
        errors.append(f"Agent '{agent_id}' state '{key}' must be {expected_ndim}D, got shape {arr.shape}")

    if expected_last_dim is not None and arr.ndim == expected_ndim and arr.shape[-1] != expected_last_dim:
        errors.append(
            f"Agent '{agent_id}' state '{key}' last dimension must be {expected_last_dim}, got shape {arr.shape}",
        )

    if expected_dtype is not None and expected_dtype is bool and arr.dtype != bool:
        errors.append(f"Agent '{agent_id}' state '{key}' must be boolean, got dtype {arr.dtype}")


def _validate_static_map_elements(static_map_elements: dict, errors: list, warnings: list, strict_keys: bool) -> None:
    """Validate static_map_elements section."""
    if not isinstance(static_map_elements, dict):
        errors.append(f"'static_map_elements' must be a dict, got {type(static_map_elements).__name__}")
        return

    for element_id, element in static_map_elements.items():
        if not isinstance(element_id, int):
            errors.append(f"Map element ID must be an int, got {type(element_id).__name__}")

        if not isinstance(element, dict):
            errors.append(f"Map element '{element_id}' must be a dict, got {type(element).__name__}")
            continue

        for key in _REQUIRED_MAP_ELEMENT_KEYS:
            if key not in element:
                errors.append(f"Map element '{element_id}' missing required key: '{key}'")

        if "type" not in element:
            errors.append(f"Map element '{element_id}' missing required key: 'type'")
            continue

        element_type = element["type"]

        # Check that element has either polyline or polygon (general check)
        if "polyline" not in element and "polygon" not in element and "position" not in element:
            errors.append(f"Map element '{element_id}' must have either 'polyline' or 'polygon' or 'position'")

        # Check geometry based on type
        if types.is_road_map_element(element_type):
            if "polyline" not in element:
                errors.append(f"Map element '{element_id}' of type '{element_type}' must have 'polyline'")
        elif types.is_marking_element(element_type):
            if "polygon" not in element:
                errors.append(f"Map element '{element_id}' of type '{element_type}' must have 'polygon'")
        elif types.is_traffic_object(element_type):
            if "position" not in element:
                errors.append(f"Map element '{element_id}' of type '{element_type}' must have 'position'")
        else:
            errors.append(
                f"Map element '{element_id}' has invalid type '{element_type}'. "
                f"Must be a valid lane, road line, road edge, map feature, or traffic object type.",
            )

        if not isinstance(element_type, str):
            errors.append(f"Map element '{element_id}' type must be a string, got {type(element_type).__name__}")

        # Lane-specific validation
        if types.is_lane(element_type):
            all_valid_keys = _REQUIRED_LANE_KEYS | _OPTIONAL_LANE_KEYS
            unexpected_keys = set(element.keys()) - all_valid_keys
            if unexpected_keys:
                msg = f"Lane '{element_id}' has unexpected keys: {unexpected_keys}"
                (errors if strict_keys else warnings).append(msg)

            if "speed_limit_mph" in element and not isinstance(element["speed_limit_mph"], (int, float)):
                errors.append(
                    f"Lane '{element_id}' speed_limit_mph must be numeric, got {type(element['speed_limit_mph']).__name__}",  # noqa: E501
                )
            if "speed_limit_kmh" in element and not isinstance(element["speed_limit_kmh"], (int, float)):
                errors.append(
                    f"Lane '{element_id}' speed_limit_kmh must be numeric, got {type(element['speed_limit_kmh']).__name__}",  # noqa: E501
                )

            for list_key in [
                "entry_lanes",
                "exit_lanes",
                "left_boundaries",
                "right_boundaries",
                "left_neighbor",
                "right_neighbor",
            ]:
                if list_key in element and not isinstance(element[list_key], list):
                    errors.append(
                        f"Lane '{element_id}' {list_key} must be a list, got {type(element[list_key]).__name__}",
                    )

        # Validate polyline
        if "polyline" in element:
            _validate_geometry_array(element_id, element["polyline"], "polyline", errors)

        # Validate polygon
        if "polygon" in element:
            _validate_geometry_array(element_id, element["polygon"], "polygon", errors)


def _validate_geometry_array(element_id: str, geometry: Any, geom_type: str, errors: list) -> None:
    """Validate polyline or polygon array."""
    if not isinstance(geometry, np.ndarray):
        errors.append(f"Map element '{element_id}' {geom_type} must be an ndarray, got {type(geometry).__name__}")
    elif geometry.ndim != 2:
        errors.append(f"Map element '{element_id}' {geom_type} must be 2D (N, 3), got shape {geometry.shape}")
    elif geometry.shape[1] != 3:
        errors.append(f"Map element '{element_id}' {geom_type} last dimension must be 3, got shape {geometry.shape}")


def _validate_dynamic_map_elements(dynamic_map_elements: dict, errors: list, warnings: list, strict_keys: bool) -> None:
    """Validate dynamic_map_elements section (traffic lights)."""
    if not isinstance(dynamic_map_elements, dict):
        errors.append(f"'dynamic_map_elements' must be a dict, got {type(dynamic_map_elements).__name__}")
        return

    for element_id, element in dynamic_map_elements.items():
        if not isinstance(element_id, int):
            errors.append(f"Dynamic map element ID must be an int, got {type(element_id).__name__}")

        if not isinstance(element, dict):
            errors.append(f"Dynamic map element '{element_id}' must be a dict, got {type(element).__name__}")
            continue

        for key in _REQUIRED_DYNAMIC_MAP_ELEMENT_KEYS:
            if key not in element:
                errors.append(f"Dynamic map element '{element_id}' missing required key: '{key}'")

        all_valid_keys = _REQUIRED_DYNAMIC_MAP_ELEMENT_KEYS | _OPTIONAL_DYNAMIC_MAP_ELEMENT_KEYS
        unexpected_keys = set(element.keys()) - all_valid_keys
        if unexpected_keys:
            msg = f"Dynamic map element '{element_id}' has unexpected keys: {unexpected_keys}"
            (errors if strict_keys else warnings).append(msg)

        if "type" in element:
            element_type = element["type"]
            if not isinstance(element_type, str):
                errors.append(
                    f"Dynamic map element '{element_id}' type must be a string, got {type(element_type).__name__}",
                )
            elif element_type != types.TRAFFIC_LIGHT:
                errors.append(
                    f"Dynamic map element '{element_id}' has invalid type '{element_type}'. "
                    f"Must be '{types.TRAFFIC_LIGHT}'",
                )

        if "position" in element:
            position = element["position"]
            if not isinstance(position, np.ndarray):
                errors.append(
                    f"Dynamic map element '{element_id}' position must be an ndarray, got {type(position).__name__}",
                )
            elif position.ndim != 1:
                errors.append(
                    f"Dynamic map element '{element_id}' position must be 1D (3,), got shape {position.shape}",
                )
            elif position.shape[0] != 3:
                errors.append(
                    f"Dynamic map element '{element_id}' position must have 3 elements (x, y, z), got shape {position.shape}",  # noqa: E501
                )

        if "states" in element:
            states = element["states"]
            if not isinstance(states, list):
                errors.append(f"Dynamic map element '{element_id}' states must be a list, got {type(states).__name__}")
            else:
                for i, state in enumerate(states):
                    if not isinstance(state, str):
                        errors.append(
                            f"Dynamic map element '{element_id}' state at index {i} must be a string, got {type(state).__name__}",  # noqa: E501
                        )
                    elif not types.is_traffic_light_state(state):
                        errors.append(
                            f"Dynamic map element '{element_id}' state at index {i} has invalid value '{state}'. "
                            f"Must be one of: {types.TRAFFIC_LIGHT_STATES}",
                        )

        if "lane" in element and not isinstance(element["lane"], int):
            errors.append(
                f"Dynamic map element '{element_id}' lane must be an int, got {type(element['lane']).__name__}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# STRICT VALIDATION - Physics-based checks
# ═══════════════════════════════════════════════════════════════════════════


def strict_validate(
    scenario: dict,
    validation_level: int = 2,
    speed_limit_tolerance: float = 0.12,
    position_jump_threshold: float = 50.0,
    velocity_tolerance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
) -> tuple[bool, list[str], list[str]]:
    """
    Perform strict validation on a UnifiedScenario.

    Checks:
    - Map geometry and topology (lane connectivity, boundaries, neighbors)
    - Agent trajectory physics (velocity-position consistency, heading alignment)
    - Physical constraints (dimensions, speed limits, acceleration)
    - Traffic light placement and state transitions
    - Cross-element relationships (ego agent, objects of interest)

    Args:
        scenario: The scenario dict to validate
        validation_level: Strictness level (1=basic, 2=standard, 3=strict, 4=pedantic)
        speed_limit_tolerance: Relative tolerance for speed limit conversion check
        position_jump_threshold: Maximum allowed position jump between timesteps (meters)
        velocity_tolerance: Tolerance for velocity-position consistency (m/s)
        heading_tolerance_deg: Tolerance for heading-velocity alignment (degrees)

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    try:
        _validate_strict_map(
            scenario.get("static_map_elements", {}),
            validation_level,
            speed_limit_tolerance,
            errors,
            warnings,
        )
        _validate_strict_agents(
            scenario.get("dynamic_agents", {}),
            scenario.get("metadata", {}),
            validation_level,
            position_jump_threshold,
            velocity_tolerance,
            heading_tolerance_deg,
            errors,
            warnings,
        )
        _validate_strict_traffic_lights(
            scenario.get("dynamic_map_elements", {}),
            scenario.get("static_map_elements", {}),
            scenario.get("metadata", {}),
            validation_level,
            errors,
            warnings,
        )
        _validate_scenario_coherence(scenario, validation_level, errors, warnings)
    except ValidationError as e:
        errors.append(str(e))

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def _validate_strict_map(
    static_map_elements: dict,
    validation_level: int,
    speed_limit_tolerance: float,
    errors: list,
    warnings: list,
) -> None:
    """Validate static map elements with strict checks."""
    for element_id, element in static_map_elements.items():
        element_type = element.get("type", "")

        if "polyline" in element:
            _validate_polyline_quality(element_id, element["polyline"], validation_level, errors, warnings)

        if types.is_lane(element_type):
            _validate_strict_lane(
                element_id,
                element,
                static_map_elements,
                validation_level,
                speed_limit_tolerance,
                errors,
                warnings,
            )


def _validate_polyline_quality(
    element_id: str,
    polyline: np.ndarray,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate polyline quality."""
    if len(polyline) < 2:
        warnings.append(f"Map element '{element_id}' has polyline with < 2 points")
        return

    for i in range(len(polyline) - 1):
        dist = np.linalg.norm(polyline[i + 1] - polyline[i])
        if dist < 1e-6:
            errors.append(f"Map element '{element_id}' has duplicate consecutive points at index {i}")

    total_length = sum(np.linalg.norm(polyline[i + 1] - polyline[i]) for i in range(len(polyline) - 1))
    if total_length < 0.5:
        warnings.append(f"Map element '{element_id}' has very short polyline (length={total_length:.2f}m)")

    if validation_level >= 3:
        for i in range(1, len(polyline) - 1):
            v1 = polyline[i] - polyline[i - 1]
            v2 = polyline[i + 1] - polyline[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                dot_product = np.clip(np.dot(v1[:2], v2[:2]), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_product))

                if angle_deg > 170:
                    warnings.append(
                        f"Map element '{element_id}' has sharp corner at point {i} (angle={angle_deg:.1f}°)",
                    )


def _validate_strict_lane(
    lane_id: str,
    lane: dict,
    all_elements: dict,
    validation_level: int,
    speed_limit_tolerance: float,
    errors: list,
    warnings: list,
) -> None:
    """Validate lane with strict checks."""
    # Speed limit validation
    if "speed_limit_mph" in lane and "speed_limit_kmh" in lane:
        mph = lane["speed_limit_mph"]
        kmh = lane["speed_limit_kmh"]

        if mph < -1 or kmh < -1:
            errors.append(f"Lane '{lane_id}' has non-positive speed limit")

        expected_kmh = mph * 1.60934
        relative_error = abs(kmh - expected_kmh) / expected_kmh if expected_kmh > 0 else 0

        if relative_error > speed_limit_tolerance:
            errors.append(
                f"Lane '{lane_id}' speed limit conversion inconsistent: "
                f"{mph} mph should be {expected_kmh:.1f} km/h, got {kmh} km/h",
            )

        if not (5 <= mph <= 130):
            warnings.append(f"Lane '{lane_id}' has unusual speed limit: {mph} mph (expected 5-130 mph)")

    _validate_lane_connectivity(lane_id, lane, all_elements, validation_level, errors, warnings)
    _validate_lane_boundaries(lane_id, lane, all_elements, errors)
    _validate_lane_neighbors(lane_id, lane, all_elements, validation_level, errors, warnings)


def _validate_lane_connectivity(
    lane_id: str,
    lane: dict,
    all_elements: dict,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate lane connectivity."""
    all_lane_ids = {eid for eid, elem in all_elements.items() if types.is_lane(elem.get("type", ""))}

    for entry_id in lane.get("entry_lanes", []):
        if str(entry_id) not in all_lane_ids:
            errors.append(f"Lane '{lane_id}' references non-existent entry lane '{entry_id}'")

    for exit_id in lane.get("exit_lanes", []):
        if str(exit_id) not in all_lane_ids:
            errors.append(f"Lane '{lane_id}' references non-existent exit lane '{exit_id}'")

    if validation_level >= 3:
        for exit_id in lane.get("exit_lanes", []):
            exit_id_str = str(exit_id)
            if exit_id_str in all_elements:
                exit_lane = all_elements[exit_id_str]
                exit_entries = [str(e) for e in exit_lane.get("entry_lanes", [])]
                if lane_id not in exit_entries:
                    warnings.append(
                        f"Lane '{lane_id}' exits to '{exit_id_str}', but '{exit_id_str}' doesn't list '{lane_id}' as entry",  # noqa: E501
                    )


def _validate_lane_boundaries(lane_id: str, lane: dict, all_elements: dict, errors: list) -> None:
    """Validate lane boundaries."""
    for boundary_id in lane.get("left_boundaries", []):
        if str(boundary_id) not in all_elements:
            errors.append(f"Lane '{lane_id}' references non-existent left boundary '{boundary_id}'")

    for boundary_id in lane.get("right_boundaries", []):
        if str(boundary_id) not in all_elements:
            errors.append(f"Lane '{lane_id}' references non-existent right boundary '{boundary_id}'")


def _validate_lane_neighbors(
    lane_id: str,
    lane: dict,
    all_elements: dict,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate lane neighbors."""
    all_lane_ids = {eid for eid, elem in all_elements.items() if types.is_lane(elem.get("type", ""))}

    for neighbor_id in lane.get("left_neighbor", []):
        if str(neighbor_id) not in all_lane_ids:
            errors.append(f"Lane '{lane_id}' references non-existent left neighbor '{neighbor_id}'")

    for neighbor_id in lane.get("right_neighbor", []):
        if str(neighbor_id) not in all_lane_ids:
            errors.append(f"Lane '{lane_id}' references non-existent right neighbor '{neighbor_id}'")

    if validation_level >= 3:
        for right_neighbor_id in lane.get("right_neighbor", []):
            right_neighbor_id_str = str(right_neighbor_id)
            if right_neighbor_id_str in all_elements:
                right_neighbor = all_elements[right_neighbor_id_str]
                left_neighbors = [str(n) for n in right_neighbor.get("left_neighbor", [])]
                if lane_id not in left_neighbors:
                    warnings.append(
                        f"Lane '{lane_id}' has right neighbor '{right_neighbor_id_str}', "
                        f"but '{right_neighbor_id_str}' doesn't list '{lane_id}' as left neighbor",
                    )


def _validate_strict_agents(
    dynamic_agents: dict,
    metadata: dict,
    validation_level: int,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
    errors: list,
    warnings: list,
) -> None:
    """Validate dynamic agents with strict checks."""
    length = metadata.get("length", 0)
    timesteps = metadata.get("timesteps", [])
    dt = _compute_timestep(timesteps)

    for agent_id, agent in dynamic_agents.items():
        agent_type = agent.get("type", "")
        states = agent.get("states", {})

        _validate_trajectory_coherence(
            agent_id,
            agent_type,
            states,
            dt,
            length,
            position_jump_threshold,
            velocity_tolerance,
            heading_tolerance_deg,
            errors,
            warnings,
        )
        _validate_physical_constraints(agent_id, agent_type, states, dt, validation_level, errors, warnings)
        _validate_temporal_consistency(agent_id, states, length, errors, warnings)


def _compute_timestep(timesteps: Any) -> float:
    """Compute average timestep duration."""
    if not isinstance(timesteps, np.ndarray) or len(timesteps) < 2:
        return 0.1
    dts = np.diff(timesteps)
    return float(np.mean(dts)) if len(dts) > 0 else 0.1


def _validate_trajectory_coherence(
    agent_id: str,
    agent_type: str,
    states: dict,
    dt: float,
    expected_length: int,
    position_jump_threshold: float,
    velocity_tolerance: float,
    heading_tolerance_deg: float,
    errors: list,
    warnings: list,
) -> None:
    """Validate trajectory coherence."""
    if "position" not in states or "valid" not in states:
        return

    position = states["position"]
    valid = states["valid"]
    velocity = states.get("velocity")
    heading = states.get("heading")

    # Check for teleportation
    for i in range(len(position) - 1):
        if valid[i] and valid[i + 1]:
            dist = np.linalg.norm(position[i + 1, :2] - position[i, :2])
            if dist > position_jump_threshold and dist > 1.0:
                errors.append(
                    f"Agent '{agent_id}' teleports at timestep {i}: "
                    f"moved {dist:.2f}m in {dt:.3f}s (max allowed: {position_jump_threshold:.2f}m)",
                )

    # Velocity-position consistency
    if velocity is not None and dt > 0:
        for i in range(len(position) - 1):
            if valid[i] and valid[i + 1] and i < len(velocity):
                computed_velocity = (position[i + 1, :2] - position[i, :2]) / dt
                stated_velocity = velocity[i, :2]
                velocity_error = np.linalg.norm(computed_velocity - stated_velocity)

                if velocity_error > velocity_tolerance:
                    warnings.append(
                        f"Agent '{agent_id}' at timestep {i}: velocity-position mismatch "
                        f"(error={velocity_error:.2f} m/s, tolerance={velocity_tolerance} m/s)",
                    )

    # Heading-velocity consistency
    if heading is not None and velocity is not None:
        for i in range(len(heading)):
            if valid[i] and i < len(velocity):
                speed = np.linalg.norm(velocity[i, :2])
                if speed > 0.5:
                    velocity_angle = np.arctan2(velocity[i, 1], velocity[i, 0])
                    heading_angle = heading[i]
                    angle_diff = np.abs(velocity_angle - heading_angle)
                    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                    angle_diff_deg = np.degrees(angle_diff)

                    if angle_diff_deg > heading_tolerance_deg:
                        warnings.append(
                            f"Agent '{agent_id}' at timestep {i}: heading-velocity misalignment "
                            f"(diff={angle_diff_deg:.1f}°, tolerance={heading_tolerance_deg}°)",
                        )


def _validate_physical_constraints(
    agent_id: str,
    agent_type: str,
    states: dict,
    dt: float,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate physical constraints."""
    # Dimension validation
    for dim_key, (min_val, max_val) in [("length", (0.1, 30.0)), ("width", (0.1, 10.0)), ("height", (0.1, 10.0))]:
        if dim_key in states:
            dim_values = states[dim_key]

            if np.any(dim_values < 0):
                errors.append(f"Agent '{agent_id}' has non-positive {dim_key}")

            if len(dim_values) > 1 and np.std(dim_values) > 0.01:
                warnings.append(f"Agent '{agent_id}' has varying {dim_key} over time (should be constant)")

            mean_dim = np.mean(dim_values)
            if not (min_val <= mean_dim <= max_val):
                warnings.append(
                    f"Agent '{agent_id}' has unusual {dim_key}={mean_dim:.2f}m (expected {min_val}-{max_val}m)",
                )

    # Type-specific dimensions
    if "length" in states and "width" in states and "height" in states:
        length = np.mean(states["length"])
        width = np.mean(states["width"])
        height = np.mean(states["height"])

        if agent_type == types.VEHICLE:
            if not (2.0 <= length <= 20.0):
                warnings.append(f"Vehicle '{agent_id}' has unusual length={length:.2f}m (expected 2-20m)")
            if not (1.5 <= width <= 3.0):
                warnings.append(f"Vehicle '{agent_id}' has unusual width={width:.2f}m (expected 1.5-3m)")
            if not (1.0 <= height <= 4.0):
                warnings.append(f"Vehicle '{agent_id}' has unusual height={height:.2f}m (expected 1-4m)")
        elif agent_type == types.PEDESTRIAN:
            if not (0.3 <= length <= 0.6):
                warnings.append(f"Pedestrian '{agent_id}' has unusual length={length:.2f}m (expected 0.3-0.6m)")
            if not (0.3 <= width <= 0.6):
                warnings.append(f"Pedestrian '{agent_id}' has unusual width={width:.2f}m (expected 0.3-0.6m)")
            if not (1.4 <= height <= 2.0):
                warnings.append(f"Pedestrian '{agent_id}' has unusual height={height:.2f}m (expected 1.4-2m)")
        elif agent_type == types.CYCLIST:
            if not (1.5 <= length <= 2.0):
                warnings.append(f"Cyclist '{agent_id}' has unusual length={length:.2f}m (expected 1.5-2m)")
            if not (0.5 <= width <= 0.8):
                warnings.append(f"Cyclist '{agent_id}' has unusual width={width:.2f}m (expected 0.5-0.8m)")

    # Velocity limits
    if "velocity" in states and "valid" in states:
        velocity = states["velocity"]
        valid = states["valid"]

        for i in range(len(velocity)):
            if valid[i]:
                speed = np.linalg.norm(velocity[i, :2])
                max_speed = {
                    types.VEHICLE: 60.0,
                    types.PEDESTRIAN: 5.0,
                    types.CYCLIST: 15.0,
                    types.OTHER: 60.0,
                }.get(agent_type, 60.0)

                if speed > max_speed:
                    warnings.append(
                        f"Agent '{agent_id}' ({agent_type}) at timestep {i}: "
                        f"speed={speed:.2f} m/s exceeds limit {max_speed} m/s",
                    )

    # Acceleration limits
    if validation_level >= 2 and "velocity" in states and "valid" in states and dt > 0:
        velocity = states["velocity"]
        valid = states["valid"]

        for i in range(len(velocity) - 1):
            if valid[i] and valid[i + 1]:
                dv = velocity[i + 1, :2] - velocity[i, :2]
                acceleration = np.linalg.norm(dv) / dt

                max_accel = {types.PEDESTRIAN: 3.0, types.CYCLIST: 5.0}.get(agent_type, 10.0)

                if acceleration > max_accel:
                    warnings.append(
                        f"Agent '{agent_id}' at timestep {i}: "
                        f"acceleration={acceleration:.2f} m/s² exceeds limit {max_accel} m/s²",
                    )


def _validate_temporal_consistency(
    agent_id: str,
    states: dict,
    expected_length: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate temporal consistency."""
    state_lengths = {key: len(value) for key, value in states.items() if isinstance(value, np.ndarray)}

    if len(state_lengths) > 0:
        lengths = set(state_lengths.values())
        if len(lengths) > 1:
            errors.append(f"Agent '{agent_id}' has inconsistent state array lengths: {state_lengths}")

        for key, length in state_lengths.items():
            if length != expected_length:
                errors.append(f"Agent '{agent_id}' state '{key}' has length {length}, expected {expected_length}")

    if "valid" in states and "position" in states:
        valid = states["valid"]
        position = states["position"]

        for i in range(1, len(valid)):
            if not valid[i - 1] and valid[i]:
                if np.any(np.isnan(position[i])):
                    warnings.append(f"Agent '{agent_id}' appears at timestep {i} with NaN position")
                elif np.linalg.norm(position[i, :2]) < 1e-3:
                    warnings.append(f"Agent '{agent_id}' appears at timestep {i} at origin (suspicious)")


def _validate_strict_traffic_lights(
    dynamic_map_elements: dict,
    static_map_elements: dict,
    metadata: dict,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate traffic lights with strict checks."""
    length = metadata.get("length", 0)
    all_lane_ids = {eid for eid, elem in static_map_elements.items() if types.is_lane(elem.get("type", ""))}

    for element_id, element in dynamic_map_elements.items():
        if element.get("type") == types.TRAFFIC_LIGHT:
            _validate_traffic_light(
                element_id,
                element,
                all_lane_ids,
                static_map_elements,
                length,
                validation_level,
                errors,
                warnings,
            )


def _validate_traffic_light(
    tl_id: str,
    traffic_light: dict,
    all_lane_ids: set,
    static_map_elements: dict,
    expected_length: int,
    validation_level: int,
    errors: list,
    warnings: list,
) -> None:
    """Validate traffic light."""
    if "lane" in traffic_light:
        lane_id = str(traffic_light["lane"])
        if lane_id not in all_lane_ids:
            errors.append(f"Traffic light '{tl_id}' references non-existent lane '{lane_id}'")
        elif "position" in traffic_light and "polyline" in static_map_elements[lane_id]:
            tl_position = traffic_light["position"][:2]
            lane_polyline = static_map_elements[lane_id]["polyline"][:, :2]
            distances = np.linalg.norm(lane_polyline - tl_position, axis=1)
            min_distance = np.min(distances)

            if min_distance > 50.0:
                warnings.append(
                    f"Traffic light '{tl_id}' is {min_distance:.1f}m away from controlled lane '{lane_id}' (expected < 50m)",  # noqa: E501
                )

    if "position" in traffic_light:
        z_height = traffic_light["position"][2]
        if not (0.0 <= z_height <= 10.0):
            warnings.append(f"Traffic light '{tl_id}' has unusual height z={z_height:.2f}m (expected 0-10m)")

    if "states" in traffic_light:
        states = traffic_light["states"]

        if len(states) != expected_length:
            errors.append(f"Traffic light '{tl_id}' has {len(states)} states, expected {expected_length}")

        if validation_level >= 2 and len(states) > 1:
            for i in range(len(states) - 1):
                if states[i] == types.TRAFFIC_LIGHT_GREEN and states[i + 1] == types.TRAFFIC_LIGHT_RED:
                    warnings.append(
                        f"Traffic light '{tl_id}' at timestep {i}: invalid transition GREEN -> RED (should go through YELLOW)",  # noqa: E501
                    )

            state_changes = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
            if state_changes > len(states) * 0.5:
                warnings.append(
                    f"Traffic light '{tl_id}' has rapid flickering: {state_changes} state changes in {len(states)} timesteps",  # noqa: E501
                )


def _validate_scenario_coherence(scenario: dict, validation_level: int, errors: list, warnings: list) -> None:
    """Validate scenario coherence."""
    metadata = scenario.get("metadata", {})
    dynamic_agents = scenario.get("dynamic_agents", {})

    ego_id = metadata.get("ego_id", "")
    if ego_id:
        if ego_id not in dynamic_agents:
            errors.append(f"Ego agent '{ego_id}' not found in dynamic_agents")
        else:
            ego_agent = dynamic_agents[ego_id]
            required_states = ["position", "heading", "velocity", "valid"]
            missing_states = [s for s in required_states if s not in ego_agent.get("states", {})]
            if missing_states:
                warnings.append(f"Ego agent '{ego_id}' missing critical states: {missing_states}")

    if "objects_of_interest" in metadata:
        for obj_id in metadata["objects_of_interest"]:
            if str(obj_id) not in dynamic_agents:
                errors.append(f"Object of interest '{obj_id}' not found in dynamic_agents")

    if "tracks_to_predict" in metadata:
        for track_id in metadata["tracks_to_predict"]:
            if str(track_id) not in dynamic_agents:
                errors.append(f"Track to predict '{track_id}' not found in dynamic_agents")

    if "timesteps" in metadata:
        timesteps = metadata["timesteps"]
        if isinstance(timesteps, np.ndarray) and len(timesteps) > 1:
            if not np.all(np.diff(timesteps) > 0):
                errors.append("Timestamps are not monotonically increasing")
            if len(np.unique(timesteps)) < len(timesteps):
                errors.append("Duplicate timestamps found")

    if validation_level >= 3:
        num_agents = len(dynamic_agents)
        if num_agents > 500:
            warnings.append(f"Scenario has {num_agents} agents (unusually high, expected 5-50 for typical scenarios)")
        elif num_agents == 0:
            warnings.append("Scenario has no dynamic agents")

        num_map_elements = len(scenario.get("static_map_elements", {}))
        if num_map_elements > 1000:
            warnings.append(f"Scenario has {num_map_elements} map elements (unusually high, expected 10-500)")

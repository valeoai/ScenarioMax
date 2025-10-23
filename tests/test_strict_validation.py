"""
Tests for strict validation functionality.
"""

import numpy as np
import pytest

from scenariomax.core import types
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.stage2_process.validation import strict_validate


class TestStrictValidator:
    """Test suite for StrictValidator class."""

    def create_basic_scenario(self) -> UnifiedScenario:
        """Create a basic valid scenario for testing."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Add basic metadata
        scenario["metadata"]["length"] = 10
        scenario["metadata"]["timesteps"] = np.linspace(0, 0.9, 10)
        scenario["metadata"]["ego_id"] = "ego"

        return scenario

    # ═══════════════════════════════════════════════════════════════════════════
    # Map Validation Tests
    # ═══════════════════════════════════════════════════════════════════════════

    def test_valid_lane(self):
        """Test that a valid lane passes validation."""
        scenario = self.create_basic_scenario()

        # Add ego agent
        num_steps = 10
        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": np.zeros((num_steps, 2)),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            "speed_limit_mph": 65.0,
            "speed_limit_kmh": 104.6,  # 65 * 1.60934
            "entry_lanes": [],
            "exit_lanes": [],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert is_valid or len(errors) == 0  # May have warnings but no errors

    def test_duplicate_polyline_points(self):
        """Test that duplicate consecutive points are caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),  # Duplicate at index 0,1
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("duplicate consecutive points" in err.lower() for err in errors)

    def test_speed_limit_conversion_mismatch(self):
        """Test that speed limit conversion inconsistency is caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "speed_limit_mph": 65.0,
            "speed_limit_kmh": 200.0,  # Wrong conversion
        }

        is_valid, errors, warnings = strict_validate(scenario, speed_limit_tolerance=0.1)

        assert not is_valid
        assert any("speed limit conversion" in err.lower() for err in errors)

    def test_nonexistent_entry_lane(self):
        """Test that reference to non-existent entry lane is caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "entry_lanes": ["nonexistent_lane"],
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("non-existent entry lane" in err.lower() for err in errors)

    def test_nonexistent_boundary(self):
        """Test that reference to non-existent boundary is caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "left_boundaries": ["nonexistent_boundary"],
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("non-existent left boundary" in err.lower() for err in errors)

    def test_nonexistent_neighbor(self):
        """Test that reference to non-existent neighbor is caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "right_neighbor": ["nonexistent_neighbor"],
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("non-existent right neighbor" in err.lower() for err in errors)

    # ═══════════════════════════════════════════════════════════════════════════
    # Agent Validation Tests
    # ═══════════════════════════════════════════════════════════════════════════

    def test_valid_vehicle_trajectory(self):
        """Test that a valid vehicle trajectory passes validation."""
        scenario = self.create_basic_scenario()

        # Create realistic vehicle trajectory
        num_steps = 10
        dt = 0.1
        positions = np.zeros((num_steps, 3))
        velocities = np.zeros((num_steps, 2))
        headings = np.zeros(num_steps)

        # Moving forward at constant 10 m/s
        for i in range(num_steps):
            positions[i] = [i * 10.0 * dt, 0.0, 0.0]
            velocities[i] = [10.0, 0.0]
            headings[i] = 0.0

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": positions,
                "heading": headings,
                "velocity": velocities,
                "length": np.full(num_steps, 4.5),
                "width": np.full(num_steps, 2.0),
                "height": np.full(num_steps, 1.5),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        # Should pass or only have warnings
        if not is_valid:
            print("Errors:", errors)
        assert is_valid or len(errors) == 0

    def test_teleportation_detection(self):
        """Test that teleportation (large position jumps) is detected."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        positions = np.zeros((num_steps, 3))
        positions[0] = [0.0, 0.0, 0.0]
        positions[1] = [100.0, 0.0, 0.0]  # Teleport 100m

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": positions,
                "heading": np.zeros(num_steps),
                "velocity": np.zeros((num_steps, 2)),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, position_jump_threshold=50.0)

        assert not is_valid
        assert any("teleport" in err.lower() for err in errors)

    def test_velocity_position_mismatch(self):
        """Test that velocity-position inconsistency is caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        positions = np.zeros((num_steps, 3))
        velocities = np.zeros((num_steps, 2))

        # Position says moving, velocity says stationary
        for i in range(num_steps):
            positions[i] = [i * 1.0, 0.0, 0.0]  # Moving 1 m/step
        velocities[:] = [0.0, 0.0]  # But velocity is zero

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": positions,
                "heading": np.zeros(num_steps),
                "velocity": velocities,
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, velocity_tolerance=0.5, validation_level=2)

        # Should have warnings about velocity-position mismatch
        assert len(warnings) > 0
        assert any("velocity-position mismatch" in warn.lower() for warn in warnings)

    def test_heading_velocity_misalignment(self):
        """Test that heading-velocity misalignment is caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        positions = np.zeros((num_steps, 3))
        velocities = np.zeros((num_steps, 2))
        headings = np.zeros(num_steps)

        # Moving forward but heading is 90 degrees off
        for i in range(num_steps):
            positions[i] = [i * 1.0, 0.0, 0.0]
            velocities[i] = [10.0, 0.0]  # Moving in +x direction
            headings[i] = np.pi / 2  # Heading in +y direction

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": positions,
                "heading": headings,
                "velocity": velocities,
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, heading_tolerance_deg=30.0, validation_level=2)

        # Should have warnings about heading-velocity misalignment
        assert len(warnings) > 0
        assert any("heading-velocity misalignment" in warn.lower() for warn in warnings)

    def test_excessive_speed(self):
        """Test that excessive speed is caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        velocities = np.zeros((num_steps, 2))
        velocities[:] = [100.0, 0.0]  # 100 m/s = 360 km/h (too fast for vehicle)

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": velocities,
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        # Should have warnings about excessive speed
        assert len(warnings) > 0
        assert any("exceeds limit" in warn.lower() for warn in warnings)

    def test_excessive_acceleration(self):
        """Test that excessive acceleration is caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        velocities = np.zeros((num_steps, 2))
        velocities[0] = [0.0, 0.0]
        velocities[1] = [20.0, 0.0]  # Instant 20 m/s change in 0.1s = 200 m/s²

        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": velocities,
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        # Should have warnings about excessive acceleration
        assert len(warnings) > 0
        assert any("acceleration" in warn.lower() and "exceeds" in warn.lower() for warn in warnings)

    def test_unrealistic_vehicle_dimensions(self):
        """Test that unrealistic vehicle dimensions are caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": np.zeros((num_steps, 2)),
                "length": np.full(num_steps, 50.0),  # 50m vehicle (too long)
                "width": np.full(num_steps, 2.0),
                "height": np.full(num_steps, 1.5),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        # Should have warnings about unusual dimensions
        assert len(warnings) > 0
        assert any("unusual length" in warn.lower() for warn in warnings)

    def test_state_array_length_mismatch(self):
        """Test that state array length mismatch is caught."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(5),  # Wrong length
                "velocity": np.zeros((num_steps, 2)),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("inconsistent state array lengths" in err.lower() for err in errors)

    # ═══════════════════════════════════════════════════════════════════════════
    # Traffic Light Validation Tests
    # ═══════════════════════════════════════════════════════════════════════════

    def test_valid_traffic_light(self):
        """Test that a valid traffic light passes validation."""
        scenario = self.create_basic_scenario()

        # Add ego agent
        num_steps = 10
        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": np.zeros((num_steps, 2)),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        # Add lane
        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_SURFACE_STREET,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        }

        # Add traffic light near the lane
        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": [types.TRAFFIC_LIGHT_RED] * 10,
            "lane": "lane_1",
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert is_valid or len(errors) == 0

    def test_traffic_light_nonexistent_lane(self):
        """Test that traffic light referencing non-existent lane is caught."""
        scenario = self.create_basic_scenario()

        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": [types.TRAFFIC_LIGHT_RED] * 10,
            "lane": "nonexistent_lane",
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("non-existent lane" in err.lower() for err in errors)

    def test_traffic_light_invalid_transition(self):
        """Test that invalid traffic light state transitions are caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_SURFACE_STREET,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        }

        # Invalid transition: GREEN -> RED (should go through YELLOW)
        states = [types.TRAFFIC_LIGHT_GREEN] * 5 + [types.TRAFFIC_LIGHT_RED] * 5

        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": states,
            "lane": "lane_1",
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        # Should have warnings about invalid transition
        assert len(warnings) > 0
        assert any("invalid transition" in warn.lower() for warn in warnings)

    def test_traffic_light_state_length_mismatch(self):
        """Test that traffic light state length mismatch is caught."""
        scenario = self.create_basic_scenario()

        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_SURFACE_STREET,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        }

        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": [types.TRAFFIC_LIGHT_RED] * 5,  # Wrong length (expected 10)
            "lane": "lane_1",
        }

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("states" in err and "expected 10" in err for err in errors)

    # ═══════════════════════════════════════════════════════════════════════════
    # Cross-Validation Tests
    # ═══════════════════════════════════════════════════════════════════════════

    def test_nonexistent_ego_agent(self):
        """Test that non-existent ego agent is caught."""
        scenario = self.create_basic_scenario()
        scenario["metadata"]["ego_id"] = "nonexistent_ego"

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("ego agent" in err.lower() and "not found" in err.lower() for err in errors)

    def test_nonexistent_object_of_interest(self):
        """Test that non-existent object of interest is caught."""
        scenario = self.create_basic_scenario()
        scenario["metadata"]["objects_of_interest"] = ["nonexistent_obj"]

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("object of interest" in err.lower() and "not found" in err.lower() for err in errors)

    def test_non_monotonic_timestamps(self):
        """Test that non-monotonic timestamps are caught."""
        scenario = self.create_basic_scenario()
        scenario["metadata"]["timesteps"] = np.array([0.0, 0.1, 0.05, 0.3])  # Not monotonic

        is_valid, errors, warnings = strict_validate(scenario)

        assert not is_valid
        assert any("not monotonically increasing" in err.lower() for err in errors)

    def test_validation_levels(self):
        """Test that different validation levels work."""
        scenario = self.create_basic_scenario()

        # Level 1 should be less strict
        is_valid_1, errors_1, warnings_1 = strict_validate(scenario, validation_level=1)

        # Level 3 should be more strict
        is_valid_3, errors_3, warnings_3 = strict_validate(scenario, validation_level=3)

        # Both should run without crashing
        assert isinstance(is_valid_1, bool)
        assert isinstance(is_valid_3, bool)

    def test_functional_validation(self):
        """Test that functional strict_validate() works correctly."""
        scenario = self.create_basic_scenario()

        num_steps = 10
        scenario["dynamic_agents"]["ego"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.zeros((num_steps, 3)),
                "heading": np.zeros(num_steps),
                "velocity": np.zeros((num_steps, 2)),
                "valid": np.ones(num_steps, dtype=bool),
            },
        }

        is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

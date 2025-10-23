"""
Tests for soft validation functionality.
"""

import numpy as np
import pytest

from scenariomax.core import types
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.stage2_process.validation import soft_validate


class TestSoftValidator:
    """Test suite for SoftValidator class."""

    def test_valid_scenario(self):
        """Test that a valid scenario passes validation."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Add dynamic agent
        scenario["dynamic_agents"]["agent_1"] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
                "heading": np.array([0.0, 0.1]),
                "velocity": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "length": np.array([4.5, 4.5]),
                "width": np.array([2.0, 2.0]),
                "height": np.array([1.5, 1.5]),
                "valid": np.array([True, True]),
            },
        }

        # Add static map element
        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "speed_limit_mph": 65.0,
            "speed_limit_kmh": 105.0,
            "entry_lanes": [],
            "exit_lanes": [],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        # Add dynamic map element (traffic light)
        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": [types.TRAFFIC_LIGHT_RED, types.TRAFFIC_LIGHT_GREEN],
            "lane": 1,
        }

        # Update metadata
        scenario["metadata"]["length"] = 2
        scenario["metadata"]["timesteps"] = np.array([0.0, 0.1])

        is_valid, errors, warnings = soft_validate(scenario)

        assert is_valid, f"Expected valid scenario, got errors: {errors}"
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_missing_top_level_key(self):
        """Test that missing top-level keys are caught."""
        scenario = {
            "id": "test_001",
            "dynamic_agents": {},
            # Missing: static_map_elements, dynamic_map_elements, metadata
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("static_map_elements" in err for err in errors)
        assert any("dynamic_map_elements" in err for err in errors)
        assert any("metadata" in err for err in errors)

    def test_invalid_agent_type(self):
        """Test that invalid agent types are caught."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["dynamic_agents"]["agent_1"] = {
            "type": "INVALID_TYPE",  # Invalid type
            "states": {},
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("invalid type" in err.lower() for err in errors)

    def test_invalid_array_shape(self):
        """Test that incorrect array shapes are caught."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["dynamic_agents"]["agent_1"] = {
            "type": types.VEHICLE,
            "states": {
                # Position should be (N, 3), but we give (N, 2)
                "position": np.array([[0.0, 0.0], [1.0, 1.0]]),
            },
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("position" in err and "last dimension" in err for err in errors)

    def test_invalid_traffic_light_state(self):
        """Test that invalid traffic light states are caught."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["dynamic_map_elements"]["tl_1"] = {
            "type": types.TRAFFIC_LIGHT,
            "position": np.array([5.0, 0.0, 3.0]),
            "states": ["INVALID_STATE", types.TRAFFIC_LIGHT_GREEN],  # Invalid state
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("invalid value" in err.lower() for err in errors)

    def test_wrong_datatype(self):
        """Test that wrong datatypes are caught."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["metadata"]["length"] = "not_an_int"  # Should be int

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("length" in err and "must be an int" in err for err in errors)

    def test_missing_metadata_keys(self):
        """Test that missing metadata keys are caught."""
        scenario = {
            "id": "test_001",
            "dynamic_agents": {},
            "static_map_elements": {},
            "dynamic_map_elements": {},
            "metadata": {
                "dataset_name": "test",
                # Missing: dataset_version, length, timesteps, ego_id
            },
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("dataset_version" in err for err in errors)
        assert any("length" in err for err in errors)
        assert any("timesteps" in err for err in errors)
        assert any("ego_id" in err for err in errors)

    def test_strict_keys_mode(self):
        """Test that strict_keys mode catches unexpected keys."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["unexpected_key"] = "some_value"

        # Non-strict mode: should warn
        is_valid, errors, warnings = soft_validate(scenario, strict_keys=False)
        assert is_valid  # Still valid, just warnings
        assert len(warnings) > 0
        assert any("unexpected" in warn.lower() for warn in warnings)

        # Strict mode: should fail
        is_valid, errors, warnings = soft_validate(scenario, strict_keys=True)
        assert not is_valid
        assert any("unexpected" in err.lower() for err in errors)

    def test_functional_validation(self):
        """Test that functional soft_validate() works correctly."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Valid scenario
        scenario["metadata"]["length"] = 0
        scenario["metadata"]["timesteps"] = np.array([])

        is_valid, errors, warnings = soft_validate(scenario, strict_keys=False)

        assert is_valid
        assert len(errors) == 0

    def test_map_element_type_validation(self):
        """Test that map element types are validated correctly."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Invalid map element type
        scenario["static_map_elements"]["elem_1"] = {
            "type": "INVALID_MAP_TYPE",
            "polyline": np.array([[0.0, 0.0, 0.0]]),
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("invalid type" in err.lower() for err in errors)

    def test_lane_specific_fields(self):
        """Test that lane-specific fields are validated."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Lane with invalid speed_limit_mph type
        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "speed_limit_mph": "not_a_number",  # Should be numeric
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("speed_limit_mph" in err and "numeric" in err for err in errors)

    def test_polyline_shape_validation(self):
        """Test that polyline shapes are validated."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Polyline with wrong shape (should be N x 3)
        scenario["static_map_elements"]["lane_1"] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0.0, 0.0]]),  # Missing z coordinate
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("polyline" in err and "last dimension" in err for err in errors)

    def test_valid_mask_dtype(self):
        """Test that valid mask must be boolean."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        scenario["dynamic_agents"]["agent_1"] = {
            "type": types.VEHICLE,
            "states": {
                "valid": np.array([1, 0], dtype=int),  # Should be bool
            },
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("valid" in err and "boolean" in err for err in errors)

    def test_map_element_with_polygon(self):
        """Test that map elements can have polygon instead of polyline."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Map element with polygon (e.g., crosswalk)
        scenario["static_map_elements"]["crosswalk_1"] = {
            "type": types.CROSSWALK,
            "polygon": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert is_valid, f"Expected valid scenario with polygon, got errors: {errors}"

    def test_map_element_missing_polyline_and_polygon(self):
        """Test that map elements must have either polyline or polygon."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Map element without polyline or polygon
        scenario["static_map_elements"]["elem_1"] = {
            "type": types.CROSSWALK,
            # Missing both polyline and polygon
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("must have either 'polyline' or 'polygon'" in err for err in errors)

    def test_polygon_shape_validation(self):
        """Test that polygon shapes are validated."""
        scenario = UnifiedScenario(scenario_id="test_001", dataset_name="test_dataset", dataset_version="v1.0")

        # Polygon with wrong shape (should be N x 3)
        scenario["static_map_elements"]["crosswalk_1"] = {
            "type": types.CROSSWALK,
            "polygon": np.array([[0.0, 0.0]]),  # Missing z coordinate
        }

        is_valid, errors, warnings = soft_validate(scenario)

        assert not is_valid
        assert any("polygon" in err and "last dimension" in err for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

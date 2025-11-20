"""
Unit tests for ScenarioMax processors.

Tests individual processor functions and configurations without full pipeline integration.
"""

import numpy as np
import pytest

from scenariomax.core import types
from scenariomax.core.unified_scenario import UnifiedScenario
from scenariomax.stage2_process.polyline_interpolation import processor as polyline_processor
from scenariomax.stage2_process.traffic_lights import processor as traffic_light_processor


class TestPolylineInterpolation:
    """Test polyline_interpolation processor."""

    def test_interpolate_polyline_basic(self, sample_unified_scenario):
        """Test basic polyline interpolation."""
        # Create lane with sparse polyline (large segments)
        scenario = sample_unified_scenario
        scenario["static_map_elements"][100]["polyline"] = np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
            dtype=np.float32,
        )

        # Apply interpolation with max_segment_length=1.0
        config = {"max_segment_length": 1.0}
        result = polyline_processor.process_polyline_interpolation(scenario, config)

        # Check that polyline has more points
        new_polyline = result["static_map_elements"][100]["polyline"]
        assert len(new_polyline) > 3, "Polyline should be interpolated with more points"

        # Check that all segments are <= max_segment_length
        for i in range(len(new_polyline) - 1):
            segment_length = np.linalg.norm(new_polyline[i + 1, :2] - new_polyline[i, :2])
            assert segment_length <= 1.1, f"Segment {i} too long: {segment_length}"

    def test_interpolate_polyline_already_dense(self, sample_unified_scenario):
        """Test interpolation on already dense polyline (should not change much)."""
        scenario = sample_unified_scenario
        # Create already dense polyline
        dense_polyline = np.array([[i, 0, 0] for i in range(20)], dtype=np.float32)
        scenario["static_map_elements"][100]["polyline"] = dense_polyline

        config = {"max_segment_length": 1.0}
        result = polyline_processor.process_polyline_interpolation(scenario, config)

        new_polyline = result["static_map_elements"][100]["polyline"]
        # Should have similar number of points (maybe a few more due to rounding)
        assert abs(len(new_polyline) - len(dense_polyline)) <= 2

    def test_interpolate_polyline_config_override(self, sample_unified_scenario):
        """Test interpolation with different config values."""
        scenario = sample_unified_scenario
        scenario["static_map_elements"][100]["polyline"] = np.array(
            [[0, 0, 0], [10, 0, 0]],
            dtype=np.float32,
        )

        # Test with larger max_segment_length (fewer points)
        config_large = {"max_segment_length": 5.0}
        result_large = polyline_processor.process_polyline_interpolation(scenario, config_large)
        polyline_large = result_large["static_map_elements"][100]["polyline"]

        # Test with smaller max_segment_length (more points)
        config_small = {"max_segment_length": 1.0}
        result_small = polyline_processor.process_polyline_interpolation(scenario, config_small)
        polyline_small = result_small["static_map_elements"][100]["polyline"]

        assert len(polyline_small) > len(polyline_large), "Smaller segment length should create more points"

    def test_interpolate_multiple_lanes(self, sample_unified_scenario):
        """Test interpolation with multiple lanes."""
        scenario = sample_unified_scenario

        # Add multiple lanes with different polylines
        scenario["static_map_elements"][101] = {
            "type": types.LANE_FREEWAY,
            "polyline": np.array([[0, 5, 0], [15, 5, 0]], dtype=np.float32),
            "speed_limit_mph": 65.0,
            "speed_limit_kmh": 104.6,
            "entry_lanes": [],
            "exit_lanes": [],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        scenario["static_map_elements"][102] = {
            "type": types.LANE_BIKE_LANE,
            "polyline": np.array([[0, -5, 0], [20, -5, 0]], dtype=np.float32),
            "speed_limit_mph": 15.0,
            "speed_limit_kmh": 24.1,
            "entry_lanes": [],
            "exit_lanes": [],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        config = {"max_segment_length": 2.0}
        result = polyline_processor.process_polyline_interpolation(scenario, config)

        # All lanes should be interpolated
        for lane_id in [100, 101, 102]:
            assert lane_id in result["static_map_elements"]
            polyline = result["static_map_elements"][lane_id]["polyline"]
            assert len(polyline) >= 2

    def test_interpolate_preserves_non_polyline_elements(self, sample_unified_scenario):
        """Test that interpolation preserves stop signs, crosswalks, etc."""
        scenario = sample_unified_scenario

        # Add stop sign
        scenario["static_map_elements"][999] = {
            "type": types.STOP_SIGN,
            "lanes": [100],
            "position": np.array([5.0, 0.0, 0.0], dtype=np.float32),
        }

        config = {"max_segment_length": 1.0}
        result = polyline_processor.process_polyline_interpolation(scenario, config)

        # Stop sign should be unchanged
        assert 999 in result["static_map_elements"]
        assert result["static_map_elements"][999]["type"] == types.STOP_SIGN
        assert np.array_equal(
            result["static_map_elements"][999]["position"],
            scenario["static_map_elements"][999]["position"],
        )


class TestTrafficLightsProcessor:
    """Test traffic_lights processor."""

    def test_generate_traffic_lights_from_lanes(self, sample_unified_scenario):
        """Test generating traffic lights from lane topology."""
        scenario = sample_unified_scenario

        # Remove existing traffic lights
        scenario["dynamic_map_elements"] = {}

        # Add lanes with connectivity
        scenario["static_map_elements"][100] = {
            "type": types.LANE_SURFACE_STREET,
            "polyline": np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            "speed_limit_mph": 35.0,
            "speed_limit_kmh": 56.3,
            "entry_lanes": [],
            "exit_lanes": [101],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        scenario["static_map_elements"][101] = {
            "type": types.LANE_SURFACE_STREET,
            "polyline": np.array([[10, 0, 0], [20, 0, 0]], dtype=np.float32),
            "speed_limit_mph": 35.0,
            "speed_limit_kmh": 56.3,
            "entry_lanes": [100],
            "exit_lanes": [],
            "left_boundaries": [],
            "right_boundaries": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        config = None
        result = traffic_light_processor.process_traffic_lights(scenario, config)

        # Should have generated some traffic lights
        assert len(result["dynamic_map_elements"]) > 0, "Should generate traffic lights from topology"

    def test_merge_traffic_lights_preserves_existing(self, sample_unified_scenario):
        """Test that processor merges rather than replaces existing traffic lights."""
        scenario = sample_unified_scenario

        # Keep existing traffic light
        existing_tl_id = 200
        existing_tl = scenario["dynamic_map_elements"][existing_tl_id].copy()

        config = None
        result = traffic_light_processor.process_traffic_lights(scenario, config)

        # Existing traffic light should still be present
        assert existing_tl_id in result["dynamic_map_elements"]
        assert result["dynamic_map_elements"][existing_tl_id]["type"] == existing_tl["type"]

    def test_traffic_lights_validation(self, sample_unified_scenario):
        """Test that generated traffic lights have valid structure."""
        scenario = sample_unified_scenario

        config = None
        result = traffic_light_processor.process_traffic_lights(scenario, config)

        # Validate all traffic lights
        for tl_id, tl_data in result["dynamic_map_elements"].items():
            assert "type" in tl_data
            assert "position" in tl_data
            assert "states" in tl_data
            assert "controlled_lane" in tl_data

            # States should match scenario length
            assert len(tl_data["states"]) == scenario["metadata"]["scenario_length"]

            # All states should be valid traffic light states
            for state in tl_data["states"]:
                assert types.is_traffic_light_state(state), f"Invalid state: {state}"

    def test_traffic_lights_with_no_lanes(self):
        """Test traffic lights processor with scenario having no lanes."""
        scenario = UnifiedScenario(scenario_id="test_no_lanes", dataset_name="test")
        scenario["metadata"].update({
            "scenario_length": 10,
            "sdc_index": 0,
            "timesteps": np.arange(10, dtype=np.float32),
        })

        # Add only agents, no map elements
        scenario["dynamic_agents"][1] = {
            "type": types.VEHICLE,
            "states": {
                "position": np.random.rand(10, 3).astype(np.float32),
                "heading": np.random.rand(10).astype(np.float32),
                "velocity": np.random.rand(10, 2).astype(np.float32),
                "length": np.full(10, 4.5, dtype=np.float32),
                "width": np.full(10, 2.0, dtype=np.float32),
                "height": np.full(10, 1.5, dtype=np.float32),
                "valid": np.ones(10, dtype=bool),
            },
        }

        config = None
        result = traffic_light_processor.process_traffic_lights(scenario, config)

        # Should not crash, may or may not generate traffic lights
        assert "dynamic_map_elements" in result


class TestProcessorComposition:
    """Test combining multiple processors."""

    def test_processor_chain_execution(self, sample_unified_scenario):
        """Test that processors can be chained together."""
        scenario = sample_unified_scenario

        # Apply traffic_lights first
        config_tl = None
        scenario = traffic_light_processor.process_traffic_lights(scenario, config_tl)

        # Then apply polyline_interpolation
        config_poly = {"max_segment_length": 1.0}
        scenario = polyline_processor.process_polyline_interpolation(scenario, config_poly)

        # Both processors should have been applied
        # Check polyline is interpolated
        polyline = scenario["static_map_elements"][100]["polyline"]
        assert len(polyline) > 3

        # Check traffic lights exist
        assert len(scenario["dynamic_map_elements"]) > 0

    def test_processor_order_independence(self, sample_unified_scenario):
        """Test that processor order doesn't break anything."""
        scenario1 = sample_unified_scenario

        # Order 1: interpolation -> traffic_lights
        s1 = polyline_processor.process_polyline_interpolation(scenario1, {"max_segment_length": 1.0})
        s1 = traffic_light_processor.process_traffic_lights(s1, None)

        scenario2 = sample_unified_scenario

        # Order 2: traffic_lights -> interpolation
        s2 = traffic_light_processor.process_traffic_lights(scenario2, None)
        s2 = polyline_processor.process_polyline_interpolation(s2, {"max_segment_length": 1.0})

        # Both should complete without errors
        assert "dynamic_map_elements" in s1
        assert "dynamic_map_elements" in s2
        assert "static_map_elements" in s1
        assert "static_map_elements" in s2

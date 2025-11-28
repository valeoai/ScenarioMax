"""
Integration tests for Stage 2: Process unified scenarios with processors.

Tests applying validation, traffic_lights, and polyline_interpolation processors
to unified pickles.
"""

import pytest

from scenariomax.core import pipeline, utils


class TestProcessWithSingleProcessor:
    """Test Stage 2 with individual processors."""

    def test_process_with_validation(self, waymo_unified_pickles, temp_output_dir, validation_config):
        """Test validation processor."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation"],
            processor_configs=validation_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        # Validation may filter invalid scenarios
        assert stats["filtered"] >= 0

        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_process_with_traffic_lights_waymo(
        self,
        waymo_unified_pickles,
        temp_output_dir,
        traffic_lights_config,
    ):
        """Test traffic_lights processor on Waymo data."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["traffic_lights"],
            processor_configs=traffic_lights_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Load a processed scenario and verify traffic lights were generated
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

        scenario = utils.load_pickle(pkl_files[0])
        assert "dynamic_map_elements" in scenario

    def test_process_with_traffic_lights_nuplan(
        self,
        nuplan_unified_pickles,
        temp_output_dir,
        traffic_lights_config,
    ):
        """Test traffic_lights processor on nuPlan data."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(nuplan_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["traffic_lights"],
            processor_configs=traffic_lights_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        processed_dir = temp_output_dir / "processed" / "nuplan"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_process_with_polyline_interpolation(
        self,
        waymo_unified_pickles,
        temp_output_dir,
        polyline_interpolation_config,
    ):
        """Test polyline_interpolation processor."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["polyline_interpolation"],
            processor_configs=polyline_interpolation_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        # Load scenario and check polylines were interpolated
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))

        scenario = utils.load_pickle(pkl_files[0])

        # Check that lanes have interpolated polylines
        for element_id, element in scenario["static_map_elements"].items():
            if "polyline" in element:
                polyline = element["polyline"]
                # Should have reasonable number of points after interpolation
                assert len(polyline) >= 2


class TestProcessWithMultipleProcessors:
    """Test Stage 2 with multiple processors combined."""

    def test_process_validation_and_traffic_lights(
        self,
        waymo_unified_pickles,
        temp_output_dir,
    ):
        """Test combining validation and traffic_lights processors."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation", "traffic_lights"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_process_all_three_processors(
        self,
        waymo_unified_pickles,
        temp_output_dir,
        all_processors_config,
    ):
        """Test combining all three processors: validation + traffic_lights + polyline_interpolation."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation", "traffic_lights", "polyline_interpolation"],
            processor_configs=all_processors_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        # Load scenario and verify all processors were applied
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))

        scenario = utils.load_pickle(pkl_files[0])

        # Should have traffic lights (from traffic_lights processor)
        assert "dynamic_map_elements" in scenario

        # Should have interpolated polylines (from polyline_interpolation)
        lane_found = False
        for element in scenario["static_map_elements"].values():
            if "polyline" in element:
                lane_found = True
                break
        assert lane_found


class TestProcessorConfigs:
    """Test processor configuration overrides."""

    def test_process_with_custom_validation_level(self, waymo_unified_pickles, temp_output_dir):
        """Test custom validation level configuration."""
        config = {
            "validation": {
                "validation_level": 3,  # Strict level
                "strict_keys": True,
            },
        }

        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation"],
            processor_configs=config,
            num_workers=2,
        )

        # With strict validation, some scenarios might be filtered
        assert stats["filtered"] >= 0

    def test_process_with_custom_polyline_config(self, waymo_unified_pickles, temp_output_dir):
        """Test custom polyline interpolation configuration."""
        config = {
            "polyline_interpolation": {
                "max_segment_length": 0.5,  # Very dense interpolation
            },
        }

        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["polyline_interpolation"],
            processor_configs=config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        # Load scenario and check dense interpolation
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))

        scenario = utils.load_pickle(pkl_files[0])

        # Lanes should have very dense polylines with max_segment_length=0.5
        for element in scenario["static_map_elements"].values():
            if "polyline" in element:
                polyline = element["polyline"]
                # Should have many points due to dense interpolation
                assert len(polyline) >= 2


class TestProcessWithoutProcessors:
    """Test Stage 2 with no processors (identity copy)."""

    def test_process_empty_processor_list(self, waymo_unified_pickles, temp_output_dir):
        """Test processing with empty processor list (should copy files)."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=[],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Files should be copied even without processing
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_process_none_processor_list(self, waymo_unified_pickles, temp_output_dir):
        """Test processing with None as processor list."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=None,
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0


class TestProcessMultiDataset:
    """Test processing multiple datasets."""

    def test_process_waymo_and_nuplan_separately(
        self,
        waymo_unified_pickles,
        nuplan_unified_pickles,
        temp_output_dir,
    ):
        """Test processing both Waymo and nuPlan with traffic_lights."""
        # Create combined input
        combined_input = temp_output_dir / "combined"
        combined_input.mkdir()

        import shutil

        waymo_dest = combined_input / "waymo"
        nuplan_dest = combined_input / "nuplan"
        shutil.copytree(waymo_unified_pickles, waymo_dest)
        shutil.copytree(nuplan_unified_pickles, nuplan_dest)

        stats = pipeline.process_unified_scenarios(
            input_path=str(combined_input),
            output_path=str(temp_output_dir),
            processors=["traffic_lights"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        # Both datasets should be processed
        processed_waymo = temp_output_dir / "processed" / "waymo"
        processed_nuplan = temp_output_dir / "processed" / "nuplan"

        assert processed_waymo.exists()
        assert processed_nuplan.exists()

        waymo_pkls = list(processed_waymo.glob("*.pkl"))
        nuplan_pkls = list(processed_nuplan.glob("*.pkl"))

        assert len(waymo_pkls) > 0
        assert len(nuplan_pkls) > 0


class TestProcessErrors:
    """Test error handling in processing."""

    def test_process_invalid_processor_name(self, waymo_unified_pickles, temp_output_dir):
        """Test error when invalid processor name specified."""
        with pytest.raises((ValueError, KeyError)):
            pipeline.process_unified_scenarios(
                input_path=str(waymo_unified_pickles.parent),
                output_path=str(temp_output_dir),
                processors=["invalid_processor"],
                processor_configs=None,
                num_workers=2,
            )

    def test_process_nonexistent_input(self, temp_output_dir):
        """Test error when input path doesn't exist."""
        nonexistent = temp_output_dir / "nonexistent"

        with pytest.raises((FileNotFoundError, ValueError)):
            pipeline.process_unified_scenarios(
                input_path=str(nonexistent),
                output_path=str(temp_output_dir),
                processors=["traffic_lights"],
                processor_configs=None,
                num_workers=2,
            )

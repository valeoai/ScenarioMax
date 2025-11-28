"""
Integration tests for Stage 1: Convert raw datasets to unified format.

Tests conversion from Waymo TFRecords and nuPlan DBs to unified pickle format.
"""

import pytest

from scenariomax.core import pipeline, utils


class TestConvertWaymo:
    """Test converting Waymo dataset to unified format."""

    def test_convert_waymo_single_file(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test converting a single Waymo TFRecord to unified format."""
        # Limit to 1 file
        config = sample_waymo_config.copy()
        config["file_limit"] = 1

        datasets = {"waymo": config}

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=1,
            batch_size=10,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Check unified pickles created
        unified_dir = temp_output_dir / "unified" / "waymo"
        assert unified_dir.exists()

        pkl_files = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_convert_waymo_multiple_files(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test converting multiple Waymo TFRecords."""
        datasets = {"waymo": sample_waymo_config}

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        unified_dir = temp_output_dir / "unified" / "waymo"
        pkl_files = list(unified_dir.glob("*.pkl"))

        # Should have multiple pickles from multiple TFRecords
        assert len(pkl_files) > 1

    def test_convert_waymo_metadata_preserved(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test that Waymo-specific metadata is preserved in unified format."""
        datasets = {"waymo": sample_waymo_config}

        pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        unified_dir = temp_output_dir / "unified" / "waymo"
        pkl_files = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

        # Load first pickle and check metadata
        scenario = utils.load_pickle(pkl_files[0])

        assert "metadata" in scenario
        assert scenario["metadata"]["dataset_name"] == "waymo"
        assert "scenario_length" in scenario["metadata"]
        assert "sdc_index" in scenario["metadata"]
        assert "timesteps" in scenario["metadata"]

        # Waymo-specific fields
        assert "objects_of_interest" in scenario["metadata"]
        assert "tracks_to_predict" in scenario["metadata"]

    def test_convert_waymo_with_num_workers(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test Waymo conversion with different worker counts."""
        for num_workers in [1, 2, 4]:
            output_dir = temp_output_dir / f"workers_{num_workers}"
            output_dir.mkdir()

            datasets = {"waymo": sample_waymo_config}

            stats = pipeline.convert_raw_to_unified(
                datasets=datasets,
                output_path=str(output_dir),
                num_workers=num_workers,
                batch_size=10,
            )

            assert stats["total_scenarios"] > 0, f"Failed with {num_workers} workers"


class TestConvertNuPlan:
    """Test converting nuPlan dataset to unified format."""

    def test_convert_nuplan_single_file(
        self,
        nuplan_test_data,
        setup_nuplan_env,
        temp_output_dir,
        sample_nuplan_config,
    ):
        """Test converting a single nuPlan DB to unified format."""
        config = sample_nuplan_config.copy()
        config["file_limit"] = 1

        datasets = {"nuplan": config}

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=1,
            batch_size=10,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        unified_dir = temp_output_dir / "unified" / "nuplan"
        assert unified_dir.exists()

        pkl_files = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_convert_nuplan_multiple_files(
        self,
        nuplan_test_data,
        setup_nuplan_env,
        temp_output_dir,
        sample_nuplan_config,
    ):
        """Test converting multiple nuPlan DBs."""
        datasets = {"nuplan": sample_nuplan_config}

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        assert stats["total_scenarios"] > 0

        unified_dir = temp_output_dir / "unified" / "nuplan"
        pkl_files = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files) > 1

    def test_convert_nuplan_metadata_preserved(
        self,
        nuplan_test_data,
        setup_nuplan_env,
        temp_output_dir,
        sample_nuplan_config,
    ):
        """Test that nuPlan metadata is preserved."""
        datasets = {"nuplan": sample_nuplan_config}

        pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        unified_dir = temp_output_dir / "unified" / "nuplan"
        pkl_files = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

        scenario = utils.load_pickle(pkl_files[0])

        assert scenario["metadata"]["dataset_name"] == "nuplan"
        assert "scenario_length" in scenario["metadata"]
        assert "timesteps" in scenario["metadata"]


class TestConvertMultiDataset:
    """Test converting multiple datasets simultaneously."""

    def test_convert_waymo_and_nuplan(
        self,
        waymo_test_data,
        nuplan_test_data,
        setup_nuplan_env,
        temp_output_dir,
        sample_waymo_config,
        sample_nuplan_config,
    ):
        """Test converting both Waymo and nuPlan in single operation."""
        datasets = {
            "waymo": sample_waymo_config,
            "nuplan": sample_nuplan_config,
        }

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        assert stats["total_scenarios"] > 0

        # Both datasets should have unified directories
        waymo_dir = temp_output_dir / "unified" / "waymo"
        nuplan_dir = temp_output_dir / "unified" / "nuplan"

        assert waymo_dir.exists()
        assert nuplan_dir.exists()

        waymo_pkls = list(waymo_dir.glob("*.pkl"))
        nuplan_pkls = list(nuplan_dir.glob("*.pkl"))

        assert len(waymo_pkls) > 0
        assert len(nuplan_pkls) > 0

    def test_convert_multi_dataset_statistics(
        self,
        waymo_test_data,
        nuplan_test_data,
        setup_nuplan_env,
        temp_output_dir,
        sample_waymo_config,
        sample_nuplan_config,
    ):
        """Test that statistics are correctly aggregated for multiple datasets."""
        datasets = {
            "waymo": sample_waymo_config,
            "nuplan": sample_nuplan_config,
        }

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        # Stats should include scenarios from both datasets
        waymo_dir = temp_output_dir / "unified" / "waymo"
        nuplan_dir = temp_output_dir / "unified" / "nuplan"

        waymo_count = len(list(waymo_dir.glob("*.pkl")))
        nuplan_count = len(list(nuplan_dir.glob("*.pkl")))

        # Total should be sum of both
        assert stats["total_scenarios"] >= waymo_count + nuplan_count


class TestConversionValidation:
    """Test conversion with validation enabled."""

    def test_convert_with_validation(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test conversion with soft validation enabled."""
        datasets = {"waymo": sample_waymo_config}

        stats = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        # All Waymo test scenarios should pass validation
        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

    def test_convert_scenario_structure(self, waymo_test_data, temp_output_dir, sample_waymo_config):
        """Test that converted scenarios have correct structure."""
        datasets = {"waymo": sample_waymo_config}

        pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(temp_output_dir),
            num_workers=2,
            batch_size=10,
        )

        unified_dir = temp_output_dir / "unified" / "waymo"
        pkl_files = list(unified_dir.glob("*.pkl"))

        scenario = utils.load_pickle(pkl_files[0])

        # Check required top-level keys
        assert "id" in scenario
        assert "dynamic_agents" in scenario
        assert "static_map_elements" in scenario
        assert "dynamic_map_elements" in scenario
        assert "metadata" in scenario

        # Check metadata
        assert "dataset_name" in scenario["metadata"]
        assert "scenario_length" in scenario["metadata"]
        assert "timesteps" in scenario["metadata"]


class TestConversionErrors:
    """Test error handling in conversion."""

    def test_convert_nonexistent_dataset_path(self, temp_output_dir):
        """Test error when dataset path doesn't exist."""
        datasets = {
            "waymo": {
                "path": "/nonexistent/path",
                "version": "v1.2",
            },
        }

        with pytest.raises((FileNotFoundError, ValueError)):
            pipeline.convert_raw_to_unified(
                datasets=datasets,
                output_path=str(temp_output_dir),
                num_workers=2,
                batch_size=10,
            )

    def test_convert_empty_dataset_dict(self, temp_output_dir):
        """Test error when no datasets specified."""
        datasets = {}

        with pytest.raises((ValueError, RuntimeError)):
            pipeline.convert_raw_to_unified(
                datasets=datasets,
                output_path=str(temp_output_dir),
                num_workers=2,
                batch_size=10,
            )

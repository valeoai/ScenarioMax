"""
Integration tests for Stage 3: Format unified scenarios to target formats.

Tests conversion from unified pickle format to waymax, gpudrive, and pufferdrive.
Includes CRITICAL tests for PufferDrive format which was previously untested.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scenariomax.core import pipeline


class TestFormatToWaymax:
    """Test formatting unified scenarios to Waymax TFRecord format."""

    def test_format_waymo_to_waymax(self, waymo_unified_pickles, temp_output_dir):
        """Test converting Waymo unified pickles to Waymax format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
        )

        assert stats["stage"] == "format_unified"
        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Check TFRecord files created
        waymax_dir = temp_output_dir / "waymax"
        assert waymax_dir.exists()

        tfrecords = list(waymax_dir.glob("*.tfrecord"))
        assert len(tfrecords) > 0, "Should create TFRecord files"

    def test_format_nuplan_to_waymax(self, nuplan_unified_pickles, temp_output_dir):
        """Test converting nuPlan unified pickles to Waymax format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(nuplan_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("*.tfrecord"))
        assert len(tfrecords) > 0

    def test_format_to_waymax_with_sharding(self, waymo_unified_pickles, temp_output_dir, waymax_config):
        """Test Waymax format with sharding enabled."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
            format_config=waymax_config,
        )

        assert stats["total_scenarios"] > 0

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("training*.tfrecord"))

        # Should have created sharded files
        assert len(tfrecords) >= waymax_config["num_shards"]

    def test_format_to_waymax_validate_tfrecord(self, waymo_unified_pickles, temp_output_dir):
        """Test that Waymax TFRecords are valid and readable."""
        pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
        )

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("*.tfrecord"))
        assert len(tfrecords) > 0

        # Try reading the first TFRecord
        tfrecord_path = str(tfrecords[0])
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        # Should be able to iterate over records
        count = 0
        for _ in dataset.take(1):
            count += 1

        assert count > 0, "TFRecord should contain readable records"


class TestFormatToGPUDrive:
    """Test formatting unified scenarios to GPUDrive JSON format."""

    def test_format_waymo_to_gpudrive(self, waymo_unified_pickles, temp_output_dir):
        """Test converting Waymo unified pickles to GPUDrive format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="gpudrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Check JSON files created
        gpudrive_dir = temp_output_dir / "gpudrive" / "waymo"
        assert gpudrive_dir.exists()

        json_files = list(gpudrive_dir.glob("*.json"))
        assert len(json_files) > 0, "Should create JSON files"

    def test_format_nuplan_to_gpudrive(self, nuplan_unified_pickles, temp_output_dir):
        """Test converting nuPlan unified pickles to GPUDrive format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(nuplan_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="gpudrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        gpudrive_dir = temp_output_dir / "gpudrive" / "nuplan"
        json_files = list(gpudrive_dir.glob("*.json"))
        assert len(json_files) > 0

    def test_format_to_gpudrive_json_structure(self, waymo_unified_pickles, temp_output_dir):
        """Test that GPUDrive JSON files have correct structure."""
        pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="gpudrive",
            num_workers=2,
        )

        gpudrive_dir = temp_output_dir / "gpudrive" / "waymo"
        json_files = list(gpudrive_dir.glob("*.json"))
        assert len(json_files) > 0

        # Load and validate first JSON file
        with open(json_files[0]) as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data
        assert "scenario_id" in data
        assert "objects" in data
        assert "roads" in data
        assert "metadata" in data

        # Validate objects structure
        assert isinstance(data["objects"], list)
        if len(data["objects"]) > 0:
            obj = data["objects"][0]
            assert "id" in obj
            assert "type" in obj


class TestFormatToPufferDrive:
    """
    CRITICAL: Test formatting unified scenarios to PufferDrive binary format.

    This format was previously completely untested.
    """

    def test_format_waymo_to_pufferdrive(self, waymo_unified_pickles, temp_output_dir):
        """Test converting Waymo unified pickles to PufferDrive binary format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0, "Should process scenarios"
        assert stats["errors"] == 0, "Should have no errors"

        # Check .bin files created
        pufferdrive_dir = temp_output_dir / "pufferdrive"
        assert pufferdrive_dir.exists(), "PufferDrive output directory should exist"

        bin_files = list(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0, "Should create .bin files"

    def test_format_nuplan_to_pufferdrive(self, nuplan_unified_pickles, temp_output_dir):
        """Test converting nuPlan unified pickles to PufferDrive format."""
        stats = pipeline.format_unified_to_target(
            input_path=str(nuplan_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0

    def test_format_to_pufferdrive_binary_structure(self, waymo_unified_pickles, temp_output_dir):
        """Test that PufferDrive .bin files have valid binary structure."""
        pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0

        # Check first binary file
        bin_file = bin_files[0]
        assert bin_file.stat().st_size > 0, "Binary file should not be empty"

        # Read header (first few bytes should be readable)
        with open(bin_file, "rb") as f:
            header = f.read(100)
            assert len(header) > 0, "Should be able to read binary data"

    def test_format_to_pufferdrive_file_naming(self, waymo_unified_pickles, temp_output_dir):
        """Test that PufferDrive files follow map_XXX.bin naming convention."""
        pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = sorted(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0

        # Check naming pattern: map_000.bin, map_001.bin, etc.
        for i, bin_file in enumerate(bin_files):
            expected_name = f"map_{i:03d}.bin"
            assert bin_file.name == expected_name, f"Expected {expected_name}, got {bin_file.name}"

    def test_format_multi_dataset_pufferdrive(
        self,
        waymo_unified_pickles,
        nuplan_unified_pickles,
        temp_output_dir,
    ):
        """Test PufferDrive with multiple datasets (Waymo + nuPlan)."""
        # Create combined input with both datasets
        combined_input = temp_output_dir / "combined_input"
        combined_input.mkdir()

        # Copy unified pickles to combined location
        import shutil

        waymo_dest = combined_input / "waymo"
        nuplan_dest = combined_input / "nuplan"
        shutil.copytree(waymo_unified_pickles, waymo_dest)
        shutil.copytree(nuplan_unified_pickles, nuplan_dest)

        stats = pipeline.format_unified_to_target(
            input_path=str(combined_input),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        # Should process scenarios from both datasets
        assert stats["total_scenarios"] > 0

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))

        # Should have merged and renamed all scenarios
        assert len(bin_files) > 0
        assert len(bin_files) >= 2, "Should have scenarios from both datasets"

    def test_format_pufferdrive_file_sizes_reasonable(self, waymo_unified_pickles, temp_output_dir):
        """Test that PufferDrive binary files have reasonable sizes."""
        pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))

        for bin_file in bin_files:
            size = bin_file.stat().st_size
            # Files should be between 1KB and 100MB (reasonable range)
            assert size > 1024, f"{bin_file.name} too small: {size} bytes"
            assert size < 100 * 1024 * 1024, f"{bin_file.name} too large: {size} bytes"


class TestFormatWithConfig:
    """Test format conversions with configuration overrides."""

    def test_format_with_config_override(self, waymo_unified_pickles, temp_output_dir):
        """Test format conversion with custom configuration."""
        custom_config = {
            "num_shards": 3,
            "shuffle": False,
        }

        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
            format_config=custom_config,
        )

        assert stats["total_scenarios"] > 0

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("training*.tfrecord"))

        # Should respect num_shards config
        assert len(tfrecords) >= 3


class TestFormatErrors:
    """Test error handling in format conversion."""

    def test_format_invalid_format_name(self, waymo_unified_pickles, temp_output_dir):
        """Test error when invalid format name specified."""
        with pytest.raises((ValueError, KeyError)):
            pipeline.format_unified_to_target(
                input_path=str(waymo_unified_pickles.parent),
                output_path=str(temp_output_dir),
                format="invalid_format",
                num_workers=2,
            )

    def test_format_nonexistent_input(self, temp_output_dir):
        """Test error when input path doesn't exist."""
        nonexistent = temp_output_dir / "nonexistent"

        with pytest.raises((FileNotFoundError, ValueError)):
            pipeline.format_unified_to_target(
                input_path=str(nonexistent),
                output_path=str(temp_output_dir),
                format="waymax",
                num_workers=2,
            )

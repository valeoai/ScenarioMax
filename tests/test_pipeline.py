"""
Comprehensive pipeline tests using real scenario data.

Tests all stages of the pipeline:
- Stage 1: Raw → Unified conversion
- Stage 2: Process unified scenarios (optional)
- Stage 3: Format unified → target format (tfexample/json)
- Full pipeline with validation
- Multi-dataset processing

Uses real test data from:
- tests/data/womd/*.tfrecord (Waymo)
- tests/data/nuplan/*.db (nuPlan)
"""

import os
import shutil
from pathlib import Path

import pytest

from scenariomax.core import pipeline, processor


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
WAYMO_DATA_DIR = TEST_DATA_DIR / "womd"
NUPLAN_DATA_DIR = TEST_DATA_DIR / "nuplan"
TEST_OUTPUT_DIR = Path(__file__).parent / "out"


@pytest.fixture(scope="function")
def output_dir():
    """Create and clean up output directory for each test."""
    # Create output directory
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

    yield TEST_OUTPUT_DIR

    # Cleanup after test
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)


@pytest.fixture(scope="session")
def setup_nuplan_env():
    """Setup nuPlan environment variables if not already set."""
    # Check if nuPlan maps are available
    nuplan_maps = os.getenv("NUPLAN_MAPS_ROOT")
    if not nuplan_maps or not Path(nuplan_maps).exists():
        pytest.skip("NUPLAN_MAPS_ROOT not set or not found - skipping nuPlan tests")

    nuplan_data = os.getenv("NUPLAN_DATA_ROOT")
    if not nuplan_data:
        # Try to set it to the test data directory
        os.environ["NUPLAN_DATA_ROOT"] = str(NUPLAN_DATA_DIR.parent)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 Tests: Raw → Unified
# ═══════════════════════════════════════════════════════════════════════════


class TestStage1RawToUnified:
    """Test Stage 1: Converting raw datasets to unified format."""

    def test_waymo_raw_to_unified(self, output_dir):
        """Test converting Waymo raw data to unified pickles."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        # Stage 1: Convert
        stats = pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=2,  # Process only 2 files for testing
            validate=False,
        )

        # Verify output
        assert stats["stage"] == "raw_to_unified"
        assert stats["datasets_processed"] == 1
        assert stats["total_scenarios"] > 0

        # Check that unified pickles were created
        waymo_output = unified_dir / "waymo"
        assert waymo_output.exists()

        # Pickles are in worker subdirectories
        pickle_files = list(waymo_output.rglob("*.pkl"))
        assert len(pickle_files) > 0, "No pickle files created"

        # Load and verify one scenario
        scenarios = processor.load_pickle_files(str(waymo_output))
        assert len(scenarios) > 0

        # Verify scenario structure
        scenario = scenarios[0]
        assert "id" in scenario
        assert "dynamic_agents" in scenario
        assert "static_map_elements" in scenario
        assert "dynamic_map_elements" in scenario
        assert "metadata" in scenario
        assert scenario["metadata"]["dataset_name"] == "waymo"

    def test_waymo_raw_to_unified_with_validation(self, output_dir):
        """Test Waymo conversion with soft validation enabled."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        # Stage 1: Convert with validation
        stats = pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,  # Process only 1 file
            validate=True,  # Enable validation
        )

        # Should still succeed (validation is soft)
        assert stats["datasets_processed"] == 1
        assert stats["total_scenarios"] > 0

    def test_nuplan_raw_to_unified(self, output_dir, setup_nuplan_env):
        """Test converting nuPlan raw data to unified pickles."""
        if not NUPLAN_DATA_DIR.exists():
            pytest.skip("nuPlan test data not found")

        unified_dir = output_dir / "unified"

        # Stage 1: Convert
        stats = pipeline.convert_raw_to_unified(
            datasets={"nuplan": str(NUPLAN_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,  # Process only 1 file
            validate=False,
        )

        # Verify output
        assert stats["stage"] == "raw_to_unified"
        assert stats["datasets_processed"] == 1
        assert stats["total_scenarios"] > 0

        # Check that unified pickles were created
        nuplan_output = unified_dir / "nuplan"
        assert nuplan_output.exists()

        # Pickles are in worker subdirectories
        pickle_files = list(nuplan_output.rglob("*.pkl"))
        assert len(pickle_files) > 0, "No pickle files created"

        # Load and verify one scenario
        scenarios = processor.load_pickle_files(str(nuplan_output))
        assert len(scenarios) > 0

        # Verify scenario structure
        scenario = scenarios[0]
        assert "id" in scenario
        assert scenario["metadata"]["dataset_name"] == "nuPlan"


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 Tests: Unified → Processed
# ═══════════════════════════════════════════════════════════════════════════


class TestStage2ProcessUnified:
    """Test Stage 2: Processing unified scenarios."""

    def test_process_unified_with_traffic_lights(self, output_dir):
        """Test processing unified scenarios with traffic light inference."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"
        processed_dir = output_dir / "processed"

        # Stage 1: Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Stage 2: Process with traffic lights
        from scenariomax.stage2_process import enhance_scenarios

        stats = pipeline.process_unified_scenarios(
            input_path=str(unified_dir),
            output_path=str(processed_dir),
            processors=[enhance_scenarios],
            num_workers=1,  # Use 1 worker to avoid pickling issues with complex processors
        )

        # Verify output
        assert stats["stage"] == "process_unified"
        assert stats["scenarios_processed"] > 0
        assert stats["processors_applied"] == 1

        # Check that processed pickles were created
        assert processed_dir.exists()
        pickle_files = list(processed_dir.rglob("*.pkl"))
        assert len(pickle_files) > 0

        # Load and verify scenarios were processed
        scenarios = processor.load_pickle_files(str(processed_dir))
        assert len(scenarios) > 0

    def test_process_unified_custom_processor(self, output_dir):
        """Test processing with custom processor function."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"
        processed_dir = output_dir / "processed"

        # Stage 1: Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Custom processor that adds a metadata field
        def add_custom_metadata(scenario):
            """Add custom metadata to scenario."""
            scenario["metadata"]["test_processed"] = True
            return scenario

        # Stage 2: Process with custom processor
        stats = pipeline.process_unified_scenarios(
            input_path=str(unified_dir),
            output_path=str(processed_dir),
            processors=[add_custom_metadata],
            num_workers=1,  # Use 1 worker for simple test
        )

        # Verify custom field was added
        scenarios = processor.load_pickle_files(str(processed_dir))
        assert len(scenarios) > 0
        assert scenarios[0]["metadata"]["test_processed"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 Tests: Unified → Target Format
# ═══════════════════════════════════════════════════════════════════════════


class TestStage3FormatToTarget:
    """Test Stage 3: Converting unified to target formats."""

    def test_format_to_tfexample(self, output_dir):
        """Test converting unified to TFExample format."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"
        tfrecord_dir = output_dir / "tfrecord"

        # Stage 1: Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Stage 3: Format to TFExample
        stats = pipeline.format_unified_to_target(
            input_path=str(unified_dir),
            output_path=str(tfrecord_dir),
            format="tfexample",
            num_workers=2,
            tfrecord_name="test",
        )

        # Verify output
        assert stats["stage"] == "unified_to_target"
        assert stats["format"] == "tfexample"
        assert stats["scenarios_processed"] > 0

        # Check that TFRecord file was created
        tfrecord_file = tfrecord_dir / "test.tfrecord"
        assert tfrecord_file.exists(), "TFRecord file not created"
        assert tfrecord_file.stat().st_size > 0, "TFRecord file is empty"

    def test_format_to_tfexample_with_sharding(self, output_dir):
        """Test converting unified to TFExample with sharding."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"
        tfrecord_dir = output_dir / "tfrecord"

        # Stage 1: Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=2,
            validate=False,
        )

        # Stage 3: Format to TFExample with sharding
        stats = pipeline.format_unified_to_target(
            input_path=str(unified_dir),
            output_path=str(tfrecord_dir),
            format="tfexample",
            num_workers=2,
            tfrecord_name="test",
            shard=2,  # Create 2 shards
        )

        # Check that sharded files were created
        shard_files = list(tfrecord_dir.glob("test-*.tfrecord"))
        assert len(shard_files) == 2, f"Expected 2 shard files, got {len(shard_files)}"

        # Verify both shards have data
        for shard_file in shard_files:
            assert shard_file.stat().st_size > 0, f"Shard {shard_file.name} is empty"

    def test_format_to_json(self, output_dir):
        """Test converting unified to JSON format (GPUDrive)."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"
        json_dir = output_dir / "json"

        # Stage 1: Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Stage 3: Format to JSON
        stats = pipeline.format_unified_to_target(
            input_path=str(unified_dir),
            output_path=str(json_dir),
            format="json",
            num_workers=2,
        )

        # Verify output
        assert stats["stage"] == "unified_to_target"
        assert stats["format"] == "json"
        assert stats["scenarios_processed"] > 0

        # Check that JSON files were created
        json_files = list(json_dir.rglob("*.json"))
        assert len(json_files) > 0, "No JSON files created"

        # Verify JSON is valid
        import json

        with open(json_files[0]) as f:
            data = json.load(f)
            assert isinstance(data, list), "JSON should contain a list of scenarios"


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Test the full 3-stage pipeline."""

    def test_full_pipeline_waymo_tfexample(self, output_dir):
        """Test full pipeline: Waymo → Unified → Process → TFExample."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        # Full pipeline: Raw → Unified → Processed → TFExample
        stats = pipeline.process_scenarios(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(output_dir),
            format="tfexample",
            processors=None,  # No processing
            num_workers=2,
            save_intermediate=False,  # In-memory mode
            validate=False,
            num_files=1,
            tfrecord_name="training",
        )

        # Verify statistics
        assert stats["pipeline"] == "full_3_stage"
        assert stats["mode"] == "in_memory"
        assert stats["stage1"]["datasets_processed"] == 1
        assert stats["stage1"]["total_scenarios"] > 0

        # Check output
        tfrecord_dir = output_dir / "tfexample"
        tfrecord_file = tfrecord_dir / "training.tfrecord"
        assert tfrecord_file.exists(), "TFRecord file not created"
        assert tfrecord_file.stat().st_size > 0, "TFRecord file is empty"

    def test_full_pipeline_waymo_with_validation(self, output_dir):
        """Test full pipeline with validation enabled."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        # Full pipeline with validation
        stats = pipeline.process_scenarios(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(output_dir),
            format="tfexample",
            processors=None,
            num_workers=2,
            save_intermediate=False,
            validate=True,  # Enable validation
            num_files=1,
            tfrecord_name="training",
        )

        # Should succeed (validation may filter some scenarios)
        assert stats["stage1"]["total_scenarios"] >= 0

        # Output should still be created
        tfrecord_dir = output_dir / "tfexample"
        tfrecord_file = tfrecord_dir / "training.tfrecord"
        assert tfrecord_file.exists()

    def test_full_pipeline_with_processing(self, output_dir):
        """Test full pipeline with Stage 2 processing."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        from scenariomax.stage2_process import enhance_scenarios

        # Full pipeline with processing
        stats = pipeline.process_scenarios(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(output_dir),
            format="tfexample",
            processors=[enhance_scenarios],  # Add traffic lights
            num_workers=2,
            save_intermediate=False,
            validate=False,
            num_files=1,
            tfrecord_name="training",
        )

        # Verify Stage 2 was executed
        assert stats["stage2"] is not None
        assert stats["stage2"]["processors_applied"] == 1

    def test_full_pipeline_save_intermediate(self, output_dir):
        """Test full pipeline with intermediate pickle saves."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        # Full pipeline with intermediate saves
        stats = pipeline.process_scenarios(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(output_dir),
            format="json",
            processors=None,
            num_workers=2,
            save_intermediate=True,  # Save intermediate pickles
            validate=False,
            num_files=1,
        )

        # Verify mode
        assert stats["mode"] == "disk_io"

        # Note: intermediate directories are cleaned up after pipeline completes
        # So we can't check for them here, but we can verify final output
        json_dir = output_dir / "json"
        json_files = list(json_dir.rglob("*.json"))
        assert len(json_files) > 0

    def test_full_pipeline_nuplan_json(self, output_dir, setup_nuplan_env):
        """Test full pipeline: nuPlan → JSON."""
        if not NUPLAN_DATA_DIR.exists():
            pytest.skip("nuPlan test data not found")

        # Full pipeline with nuPlan
        stats = pipeline.process_scenarios(
            datasets={"nuplan": str(NUPLAN_DATA_DIR)},
            output_path=str(output_dir),
            format="json",
            processors=None,
            num_workers=2,
            save_intermediate=False,
            validate=False,
            num_files=1,
        )

        # Verify output
        assert stats["stage1"]["datasets_processed"] == 1

        json_dir = output_dir / "json"
        json_files = list(json_dir.rglob("*.json"))
        assert len(json_files) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Dataset Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiDataset:
    """Test processing multiple datasets together."""

    def test_multi_dataset_waymo_nuplan(self, output_dir, setup_nuplan_env):
        """Test processing Waymo + nuPlan together."""
        if not WAYMO_DATA_DIR.exists() or not NUPLAN_DATA_DIR.exists():
            pytest.skip("Test data not found")

        # Process both datasets together
        stats = pipeline.process_scenarios(
            datasets={
                "waymo": str(WAYMO_DATA_DIR),
                "nuplan": str(NUPLAN_DATA_DIR),
            },
            output_path=str(output_dir),
            format="tfexample",
            processors=None,
            num_workers=2,
            save_intermediate=False,
            validate=False,
            num_files=1,  # 1 file from each dataset
            tfrecord_name="multi",
        )

        # Verify both datasets were processed
        assert stats["stage1"]["datasets_processed"] == 2

        # Check merged output
        tfrecord_file = output_dir / "tfexample" / "multi.tfrecord"
        assert tfrecord_file.exists()
        assert tfrecord_file.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Test validation functionality in pipeline."""

    def test_soft_validation_on_unified_scenarios(self, output_dir):
        """Test soft validation on converted unified scenarios."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        # Convert with validation disabled first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Load scenarios and validate manually
        from scenariomax.stage2_process.validation import soft_validate

        scenarios = processor.load_pickle_files(str(unified_dir))
        assert len(scenarios) > 0

        # Run soft validation on each scenario
        validation_results = []
        for scenario in scenarios:
            is_valid, errors, warnings = soft_validate(scenario)
            validation_results.append((is_valid, errors, warnings))

        # Most scenarios should be valid
        valid_count = sum(1 for is_valid, _, _ in validation_results if is_valid)
        assert valid_count > 0, "No valid scenarios found"

    def test_strict_validation_on_unified_scenarios(self, output_dir):
        """Test strict validation on converted unified scenarios."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        # Convert first
        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=2,
            num_files=1,
            validate=False,
        )

        # Load scenarios and validate with strict validation
        from scenariomax.stage2_process.validation import strict_validate

        scenarios = processor.load_pickle_files(str(unified_dir))
        assert len(scenarios) > 0

        # Run strict validation on first few scenarios
        for i, scenario in enumerate(scenarios[:3]):  # Test first 3
            is_valid, errors, warnings = strict_validate(scenario, validation_level=2)

            # Print results for debugging
            print(f"\nScenario {i} validation:")
            print(f"  Valid: {is_valid}")
            if errors:
                print(f"  Errors ({len(errors)}):")
                for error in errors[:5]:
                    print(f"    - {error}")
            if warnings:
                print(f"  Warnings ({len(warnings)}):")
                for warning in warnings[:5]:
                    print(f"    - {warning}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

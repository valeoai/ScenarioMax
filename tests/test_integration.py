"""
Integration tests using real scenario data from tests/data.

These tests validate:
1. Converting raw data to unified format
2. Validating unified scenarios (soft and strict)
3. Basic pipeline functionality

Uses minimal real data to keep tests fast:
- tests/data/womd/*.tfrecord (Waymo)
- tests/data/nuplan/*.db (nuPlan)
"""

import os
import shutil
from pathlib import Path

import pytest

from scenariomax.core import pipeline, processor
from scenariomax.core.validation import soft_validate, strict_validate


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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Raw → Unified Conversion Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWaymoConversion:
    """Test converting Waymo data to unified format."""

    def test_convert_waymo_to_unified(self, output_dir):
        """Test Waymo raw → unified conversion."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        # Convert
        stats = pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=1,
            num_files=1,
            validate=False,
        )

        # Verify
        assert stats["datasets_processed"] == 1
        assert stats["total_scenarios"] > 0

        # Load scenarios
        scenarios = processor.load_pickle_files(str(unified_dir))
        assert len(scenarios) > 0

        # Check first scenario structure
        scenario = scenarios[0]
        assert "id" in scenario
        assert "dynamic_agents" in scenario
        assert "static_map_elements" in scenario
        assert "dynamic_map_elements" in scenario
        assert "metadata" in scenario
        assert scenario["metadata"]["dataset_name"] == "waymo"


class TestNuPlanConversion:
    """Test converting nuPlan data to unified format."""

    def test_convert_nuplan_to_unified(self, output_dir, setup_nuplan_env):
        """Test nuPlan raw → unified conversion."""
        if not NUPLAN_DATA_DIR.exists():
            pytest.skip("nuPlan test data not found")

        unified_dir = output_dir / "unified"

        # Convert
        stats = pipeline.convert_raw_to_unified(
            datasets={"nuplan": str(NUPLAN_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=1,
            num_files=1,
            validate=False,
        )

        # Verify
        assert stats["datasets_processed"] == 1
        assert stats["total_scenarios"] > 0

        # Load scenarios
        scenarios = processor.load_pickle_files(str(unified_dir))
        assert len(scenarios) > 0

        # Check first scenario structure
        scenario = scenarios[0]
        assert "id" in scenario
        assert scenario["metadata"]["dataset_name"] == "nuPlan"


# ═══════════════════════════════════════════════════════════════════════════
# Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationWaymo:
    """Test validation on Waymo unified scenarios."""

    @pytest.fixture(scope="class")
    def waymo_scenarios(self, tmp_path_factory):
        """Convert Waymo data once for all validation tests in this class."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        tmpdir = tmp_path_factory.mktemp("waymo_validation")
        unified_dir = tmpdir / "unified"

        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=1,
            num_files=1,
            validate=False,
        )

        scenarios = processor.load_pickle_files(str(unified_dir))
        return scenarios

    def test_soft_validation_waymo(self, waymo_scenarios):
        """Test soft validation on Waymo scenarios."""
        # Run soft validation on all scenarios
        results = []
        for scenario in waymo_scenarios:
            is_valid, errors, warnings = soft_validate(scenario)
            results.append((is_valid, len(errors), len(warnings)))

        # Most scenarios should pass soft validation
        valid_count = sum(1 for is_valid, _, _ in results if is_valid)
        assert valid_count > 0, "No scenarios passed soft validation"

        # Print summary
        print(f"\nSoft Validation Results:")
        print(f"  Total scenarios: {len(results)}")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {len(results) - valid_count}")

    def test_strict_validation_waymo(self, waymo_scenarios):
        """Test strict validation on Waymo scenarios."""
        # Run strict validation on first 3 scenarios
        results = []
        for scenario in waymo_scenarios[:3]:
            is_valid, errors, warnings = strict_validate(scenario, validation_level=2)
            results.append((is_valid, len(errors), len(warnings)))

            # Print details for first scenario
            if len(results) == 1:
                print(f"\nFirst Waymo Scenario Strict Validation:")
                print(f"  Valid: {is_valid}")
                print(f"  Errors: {len(errors)}")
                print(f"  Warnings: {len(warnings)}")
                if errors:
                    print(f"  Sample errors:")
                    for error in errors[:3]:
                        print(f"    - {error}")
                if warnings:
                    print(f"  Sample warnings:")
                    for warning in warnings[:3]:
                        print(f"    - {warning}")

        # At least one scenario should have results
        assert len(results) > 0


class TestValidationNuPlan:
    """Test validation on nuPlan unified scenarios."""

    @pytest.fixture(scope="class")
    def nuplan_scenarios(self, tmp_path_factory, setup_nuplan_env):
        """Convert nuPlan data once for all validation tests in this class."""
        if not NUPLAN_DATA_DIR.exists():
            pytest.skip("nuPlan test data not found")

        tmpdir = tmp_path_factory.mktemp("nuplan_validation")
        unified_dir = tmpdir / "unified"

        pipeline.convert_raw_to_unified(
            datasets={"nuplan": str(NUPLAN_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=1,
            num_files=1,
            validate=False,
        )

        scenarios = processor.load_pickle_files(str(unified_dir))
        return scenarios

    def test_soft_validation_nuplan(self, nuplan_scenarios):
        """Test soft validation on nuPlan scenarios."""
        # Run soft validation on all scenarios
        results = []
        for scenario in nuplan_scenarios:
            is_valid, errors, warnings = soft_validate(scenario)
            results.append((is_valid, len(errors), len(warnings)))

        # Most scenarios should pass soft validation
        valid_count = sum(1 for is_valid, _, _ in results if is_valid)
        assert valid_count > 0, "No scenarios passed soft validation"

        # Print summary
        print(f"\nSoft Validation Results:")
        print(f"  Total scenarios: {len(results)}")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {len(results) - valid_count}")

    def test_strict_validation_nuplan(self, nuplan_scenarios):
        """Test strict validation on nuPlan scenarios."""
        # Run strict validation on first 3 scenarios
        results = []
        for scenario in nuplan_scenarios[:3]:
            is_valid, errors, warnings = strict_validate(scenario, validation_level=2)
            results.append((is_valid, len(errors), len(warnings)))

            # Print details for first scenario
            if len(results) == 1:
                print(f"\nFirst nuPlan Scenario Strict Validation:")
                print(f"  Valid: {is_valid}")
                print(f"  Errors: {len(errors)}")
                print(f"  Warnings: {len(warnings)}")
                if errors:
                    print(f"  Sample errors:")
                    for error in errors[:3]:
                        print(f"    - {error}")
                if warnings:
                    print(f"  Sample warnings:")
                    for warning in warnings[:3]:
                        print(f"    - {warning}")

        # At least one scenario should have results
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Statistics Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScenarioStatistics:
    """Test scenario statistics and structure."""

    def test_waymo_scenario_stats(self, output_dir):
        """Test getting statistics from Waymo scenarios."""
        if not WAYMO_DATA_DIR.exists():
            pytest.skip("Waymo test data not found")

        unified_dir = output_dir / "unified"

        pipeline.convert_raw_to_unified(
            datasets={"waymo": str(WAYMO_DATA_DIR)},
            output_path=str(unified_dir),
            num_workers=1,
            num_files=1,
            validate=False,
        )

        scenarios = processor.load_pickle_files(str(unified_dir))
        scenario = scenarios[0]

        # Get stats
        stats = scenario.get_stats()

        # Verify stats structure
        assert "num_dynamic_agents" in stats
        assert "num_static_map_elements" in stats
        assert "num_dynamic_map_elements" in stats
        assert "duration_steps" in stats
        assert "dynamic_agent_types" in stats
        assert "static_map_element_types" in stats

        # Print stats
        print(f"\nWaymo Scenario Stats:")
        print(f"  Dynamic agents: {stats['num_dynamic_agents']}")
        print(f"  Static map elements: {stats['num_static_map_elements']}")
        print(f"  Dynamic map elements: {stats['num_dynamic_map_elements']}")
        print(f"  Duration (steps): {stats['duration_steps']}")
        print(f"  Agent types: {stats['dynamic_agent_types']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

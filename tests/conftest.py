"""
Shared pytest fixtures for ScenarioMax tests.

Provides reusable test data, configurations, and pre-converted scenarios
to speed up test execution and reduce duplication.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from scenariomax.core import pipeline, types
from scenariomax.core.unified_scenario import UnifiedScenario


# ═══════════════════════════════════════════════════════════════════════════
# Test Data Paths
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def waymo_test_data() -> Path:
    """Path to Waymo test TFRecords (5 files, ~44MB)."""
    data_path = Path(__file__).parent / "data" / "womd"
    if not data_path.exists():
        pytest.skip(f"Waymo test data not found at {data_path}")
    return data_path


@pytest.fixture(scope="session")
def nuplan_test_data() -> Path:
    """Path to nuPlan test DBs (5 files, ~1.3GB)."""
    data_path = Path(__file__).parent / "data" / "nuplan"
    if not data_path.exists():
        pytest.skip(f"nuPlan test data not found at {data_path}")
    return data_path


# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def setup_nuplan_env() -> None:
    """Setup NUPLAN_MAPS_ROOT environment variable for nuPlan tests."""
    nuplan_maps = os.environ.get("NUPLAN_MAPS_ROOT")
    if not nuplan_maps:
        pytest.skip("NUPLAN_MAPS_ROOT environment variable not set")

    maps_path = Path(nuplan_maps)
    if not maps_path.exists():
        pytest.skip(f"NUPLAN_MAPS_ROOT path does not exist: {maps_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Temporary Directories
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Clean temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Configurations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_waymo_config(waymo_test_data) -> dict:
    """Waymo dataset configuration dict."""
    return {
        "path": str(waymo_test_data),
        "version": "v1.2",
        "file_limit": None,
    }


@pytest.fixture
def sample_nuplan_config(nuplan_test_data) -> dict:
    """nuPlan dataset configuration dict."""
    return {
        "path": str(nuplan_test_data),
        "metadata_dir": str(nuplan_test_data),
        "version": "v1.1",
        "file_limit": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Test Scenarios
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_unified_scenario() -> UnifiedScenario:
    """Create a valid synthetic UnifiedScenario for testing."""
    scenario = UnifiedScenario(scenario_id="test_scenario_001", dataset_name="test")

    # Add metadata
    scenario["metadata"].update(
        {
            "scenario_length": 10,
            "sdc_index": 0,
            "timesteps": np.arange(10, dtype=np.float32),
        },
    )

    # Add one agent
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

    # Add one lane
    scenario["static_map_elements"][100] = {
        "type": types.LANE_SURFACE_STREET,
        "polyline": np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32),
        "speed_limit_mph": 35.0,
        "speed_limit_kmh": 56.3,
        "entry_lanes": [],
        "exit_lanes": [],
        "left_boundaries": [],
        "right_boundaries": [],
        "left_neighbor": [],
        "right_neighbor": [],
    }

    # Add one traffic light
    scenario["dynamic_map_elements"][200] = {
        "type": types.TRAFFIC_LIGHT,
        "position": np.array([5.0, 0.0, 3.0], dtype=np.float32),
        "states": [types.TRAFFIC_LIGHT_GREEN] * 10,
        "controlled_lane": 100,
    }

    return scenario


# ═══════════════════════════════════════════════════════════════════════════
# Pre-converted Scenarios (Session-scoped for Performance)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def waymo_unified_pickles(tmp_path_factory, waymo_test_data) -> Path:
    """
    Pre-converted Waymo scenarios as unified pickles.

    Runs Stage 1 conversion once per test session and caches results.
    Significantly speeds up tests that need unified pickles.
    """
    output_dir = tmp_path_factory.mktemp("waymo_unified")

    datasets = {"waymo": {"path": str(waymo_test_data), "version": "v1.2"}}

    pipeline.convert_raw_to_unified(
        datasets=datasets,
        output_path=str(output_dir),
        num_workers=2,
        batch_size=10,
    )

    pickle_dir = output_dir / "unified" / "waymo"
    return pickle_dir


@pytest.fixture(scope="session")
def nuplan_unified_pickles(tmp_path_factory, nuplan_test_data, setup_nuplan_env) -> Path:
    """
    Pre-converted nuPlan scenarios as unified pickles.

    Runs Stage 1 conversion once per test session and caches results.
    Requires NUPLAN_MAPS_ROOT environment variable.
    """
    output_dir = tmp_path_factory.mktemp("nuplan_unified")

    datasets = {
        "nuplan": {
            "path": str(nuplan_test_data),
            "metadata_dir": str(nuplan_test_data),
            "version": "v1.1",
        },
    }

    pipeline.convert_raw_to_unified(
        datasets=datasets,
        output_path=str(output_dir),
        num_workers=2,
        batch_size=10,
    )

    pickle_dir = output_dir / "unified" / "nuplan"
    return pickle_dir


@pytest.fixture(scope="session")
def processed_pickles(tmp_path_factory, waymo_unified_pickles) -> Path:
    """
    Pre-processed Waymo scenarios with traffic_lights processor applied.

    Runs Stage 1 + Stage 2 once per session for tests needing processed pickles.
    """
    output_dir = tmp_path_factory.mktemp("processed")

    pipeline.process_unified_scenarios(
        input_path=str(waymo_unified_pickles.parent),
        output_path=str(output_dir),
        processors=["traffic_lights"],
        processor_configs=None,
        num_workers=2,
    )

    processed_dir = output_dir / "processed" / "waymo"
    return processed_dir


# ═══════════════════════════════════════════════════════════════════════════
# Processor Configurations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def traffic_lights_config() -> dict:
    """Configuration for traffic_lights processor."""
    return {
        "traffic_lights": {
            "generate_missing": True,
        },
    }


@pytest.fixture
def polyline_interpolation_config() -> dict:
    """Configuration for polyline_interpolation processor."""
    return {
        "polyline_interpolation": {
            "max_segment_length": 1.0,
        },
    }


@pytest.fixture
def validation_config() -> dict:
    """Configuration for validation processor."""
    return {
        "validation": {
            "validation_level": 2,
            "strict_keys": False,
        },
    }


@pytest.fixture
def all_processors_config(
    traffic_lights_config,
    polyline_interpolation_config,
    validation_config,
) -> dict:
    """Combined configuration for all processors."""
    config = {}
    config.update(traffic_lights_config)
    config.update(polyline_interpolation_config)
    config.update(validation_config)
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Format Configurations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def waymax_config() -> dict:
    """Configuration for waymax format."""
    return {
        "num_shards": 2,
        "shuffle": True,
    }


@pytest.fixture
def gpudrive_config() -> dict:
    """Configuration for gpudrive format."""
    return {}


@pytest.fixture
def pufferdrive_config() -> dict:
    """Configuration for pufferdrive format."""
    return {}

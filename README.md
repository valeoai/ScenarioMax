# ScenarioMax

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ScenarioMax is an extension to [ScenarioNet](https://github.com/metadriverse/scenarionet) that transforms various autonomous driving datasets into standardized formats. Like ScenarioNet, it first converts different datasets (Waymo, nuPlan, nuScenes) to a unified pickle format. ScenarioMax then extends this process with additional pipelines to convert this unified data into formats compatible with [Waymax](https://github.com/waymo-research/waymax), [V-Max](https://github.com/valeoai/V-Max), [GPUDrive](https://github.com/Emerge-Lab/gpudrive), and [PufferDrive](https://github.com/Emerge-Lab/PufferDrive).

Autonomous driving scenarios are real-world recordings from the point of view of a vehicle, represented in a Bird's Eye View (BEV) perspective. ScenarioMax provides a framework to convert and modify these scenarios for simulation and machine learning applications.

<div align="center">
  <img src="docs/scheme.png" alt="Scheme" width="100%" />
</div>

## 🎯 What Can ScenarioMax Do?

With ScenarioMax, you can:

1. **Convert datasets into specific formats**: Transform one or many dataset sources into your desired format
   - Single dataset: Convert Waymo → TFRecord/JSON/Puffer
   - Multi-dataset: Combine Waymo + nuPlan → Single unified output
   - Results saved in folders named by dataset combination (e.g., "waymo_nuplan")

2. **Process and enhance scenarios**: Modify the unified format with various processors
   - `validation`: Two-level validation (soft/strict) for data quality
   - `traffic_lights`: Traffic light inference from vehicle behavior
   - `polyline_interpolation`: Interpolate polylines for dense representation

3. **Flexible pipeline modes**:
   - **3-stage independent**: Run Convert → Process → Format separately
   - **Full pipeline**: Run all stages together with file-by-file streaming (default)

4. **Visualize scenarios**: Generate Bird's Eye View (BEV) PNG images of scenarios

**Supported datasets**: Waymo Open Motion Dataset (WOMD), nuPlan, OpenScenes, nuScenes (WIP), Argoverse2 (WIP)

**Supported output formats**:
- Unified Scenario (intermediate `.pkl` format)
- TFExample (`.tfrecord` for Waymax/V-Max simulators)
- JSON (for GPUDrive simulator)
- Puffer (binary `.bin` format for PufferDrive simulator)

## 🚀 Key Features

- **Multi-Dataset Support**: Unified interface for Waymo Open Motion Dataset, nuScenes, nuPlan, and OpenScenes
- **Flexible Output Formats**: Convert to TFExample (Waymax/V-Max), JSON (GPUDrive), Puffer (PufferDrive), or unified pickle format
- **High Performance**: File-by-file streaming architecture for TB-scale datasets with parallel processing
- **3-Stage Pipeline Architecture**: Convert → Process → Format for maximum flexibility
- **Enhanced Scenarios**: Optional scenario enhancement with traffic light inference and validation
- **Two-Level Validation**: Soft validation (structural checks) and strict validation (physics-based checks)
- **Production Features**: Checkpointing for resumable pipelines, validation reports, and automatic disk space checks
- **Visualization**: Bird's Eye View (BEV) rendering of scenarios with matplotlib
- **Memory Efficient**: File-by-file streaming ensures no batch loading into memory - scales to TB-scale datasets

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Supported Datasets](#supported-datasets)
- [Output Formats](#output-formats)
- [Architecture](#architecture)
- [Validation System](#validation-system)
- [Production Features](#production-features)
- [Development](#development)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/) for fast dependency management
- Access to at least one supported dataset (Waymo, nuPlan, or nuScenes)
- Sufficient disk space for dataset processing

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/valeoai/ScenarioMax.git
cd ScenarioMax

# Create and activate virtual environment
uv venv -p 3.10
source .venv/bin/activate

# Install ScenarioMax with dataset support
make waymo         # Waymo Open Motion Dataset
make nuplan        # nuPlan dataset
make nuscenes      # nuScenes dataset (WIP)
make all           # All datasets
make dev           # Development environment
```

### Manual Installation

```bash
# For specific datasets
uv pip install -e ".[waymo]"      # Waymo Open Motion Dataset
uv pip install -e ".[nuplan]"     # nuPlan support
uv pip install -e ".[nuscenes]"   # nuScenes support (WIP)
uv pip install -e ".[dev]"        # Development tools
uv pip install -e ".[all]"        # All datasets support
```

### Environment Setup

For nuPlan dataset, set required environment variables:

```bash
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NUPLAN_DATA_ROOT=/path/to/nuplan/data
```

For deterministic TFRecord shuffling (optional):

```bash
export SCENARIOMAX_SHUFFLE_SEED=42  # Default: 42
```

## 🚀 Quick Start

### 3-Stage Pipeline

ScenarioMax uses a flexible 3-stage pipeline architecture where each stage is independent, multiprocessed, and can be run separately or chained together:

```
Stage 1: Convert  - Raw dataset(s) → Unified pickles
Stage 2: Process  - Unified pickles → Enhanced pickles (optional)
Stage 3: Format   - Unified pickles → Target format (tfexample/json/puffer)
```

**Key Architecture Principle:**
- **File-by-file streaming**: Each worker processes files individually - load one file, process it, save it, move to next file
- **No batch loading**: Workers never load all files into memory at once
- **Worker independence**: Each worker opens and closes its own files
- **Memory efficient**: Scales to TB-scale datasets without memory issues

### Basic Usage

```bash
# Stage 1: Convert raw dataset to unified format
scenariomax command=convert datasets.waymo=/data/waymo output.dst=/output/unified execution.num_workers=16

# Stage 2: Process unified scenarios (add traffic lights, validate, etc.)
scenariomax command=process input_path=/output/unified output.dst=/output/processed \
            processing.processors=[validation,traffic_lights]

# Stage 3: Convert to target format
scenariomax command=format input_path=/output/processed output.dst=/output/tfrecord \
            output.format=tfexample output.num_shards=10
scenariomax command=format input_path=/output/processed output.dst=/output/json output.format=json
scenariomax command=format input_path=/output/processed output.dst=/output/puffer output.format=puffer

# Or run all 3 stages at once (file-by-file streaming, memory efficient)
scenariomax command=pipeline datasets.waymo=/data/waymo output.dst=/output \
            output.format=tfexample execution.num_workers=16

# Visualize unified scenarios (BEV PNG/video)
scenariomax command=viz input_path=/output/unified output.dst=/output/viz
```

## 📊 Usage Examples

### Use Case 1: Single Dataset Conversion

```bash
# Convert Waymo to TFRecord format (file-by-file streaming)
scenariomax command=pipeline datasets.waymo=/data/waymo output.dst=/output \
            output.format=tfexample execution.num_workers=8
```

### Use Case 2: Multi-Dataset Processing

```bash
# Combine Waymo and nuPlan datasets into single output
scenariomax command=pipeline datasets.waymo=/data/waymo datasets.nuplan=/data/nuplan \
            output.dst=/output output.format=tfexample output.num_shards=10 \
            execution.num_workers=16
```

### Use Case 3: Enhanced Processing with Validation

```bash
# Add traffic light processing with strict validation
scenariomax command=pipeline datasets.waymo=/data/waymo output.dst=/output \
            output.format=tfexample processing.processors=[validation,traffic_lights] \
            processing.processor_configs.validation.mode=strict execution.num_workers=8
```

### Use Case 4: Convert to Puffer Format

```bash
# Convert to Puffer simulator format
scenariomax command=pipeline datasets.waymo=/data/waymo output.dst=/output \
            output.format=puffer execution.num_workers=16
```

### Use Case 5: Validation-Only Mode

```bash
# Process scenarios with validation only
scenariomax command=process input_path=/output/unified output.dst=/tmp/validation \
            processing.processors=[validation] \
            processing.processor_configs.validation.mode=strict
```

### Use Case 6: Visualization Workflow

```bash
# Stage 1: Convert to unified format
scenariomax command=convert datasets.waymo=/data/waymo output.dst=/unified \
            execution.num_workers=8

# Visualize scenarios
scenariomax command=viz input_path=/unified output.dst=/viz

# Stage 3: Convert to target format
scenariomax command=format input_path=/unified output.dst=/output output.format=json \
            execution.num_workers=8
```

## 🗂️ Supported Datasets


| Dataset | Version | Link | Status |
|---------|---------|------|--------|
| Waymo Open Motion Dataset | v1.3.0 | [Site](https://waymo.com/open/download/) | ✅ Full Support |
| nuPlan | v1.1 | [Site](https://www.nuscenes.org/nuplan) | ✅ Full Support |
| nuScenes | v1.0 | [Site](https://www.nuscenes.org/nuscenes) | 🚧 WIP|
| Argoverse | v2.0 | [Site](https://www.argoverse.org/av2.html#forecasting-link) | 🚧 WIP |

### Dataset-Specific Options

```bash
# nuScenes with specific split
scenariomax \
  --nuscenes_src /data/nuscenes \
  --split v1.0-trainval \
  --dst /output \
  --target_format tfexample

# nuPlan with direct log parsing
scenariomax \
  --nuplan_src /data/nuplan \
  --nuplan_direct_from_logs \
  --dst /output \
  --target_format gpudrive
```

## 📤 Output Formats

### TFRecord (TensorFlow/Waymax/V-Max)

```bash
--format tfexample
```

- **Use Case**: Training neural networks with Waymax or V-Max simulators
- **Output**: `training.tfrecord` files with optional sharding support
- **Features**: TensorFlow-native format, efficient for ML training pipelines

### JSON (GPUDrive)

```bash
--format json
```

- **Use Case**: GPU-accelerated simulation and training with GPUDrive
- **Output**: JSON files compatible with GPUDrive simulator
- **Features**: Human-readable format, easy debugging

### Puffer (PufferDrive)

```bash
--format puffer
```

- **Use Case**: Simulation with PufferDrive simulator
- **Output**: Binary `.bin` files in Puffer format with roadgraph and agent data
- **Features**: In-memory binary conversion, dedicated converters for agents, roadgraph, routes, and traffic lights
- **Location**: `scenariomax/stage3_format/puffer/`

### Unified Pickle Format

```bash
# Automatically created during Stage 1 (convert)
scenariomax convert --waymo_src /data/waymo --dst /output/unified
```

- **Use Case**: Intermediate format for custom processing and debugging
- **Features**: Full scenario data preservation, Python-native
- **Output**: `.pkl` files with complete scenario information
- **Schema**: Defined in `scenariomax/core/unified_scenario.py`

## 🏗️ Architecture

ScenarioMax uses a **3-stage pipeline architecture** with file-by-file streaming:

```
Stage 1: Convert  →  Stage 2: Process  →  Stage 3: Format
Raw Data          →  Unified Format     →  Target Format
[Dataset]         →  [Enhancement]      →  [ML Ready]
```

### Pipeline Stages

1. **Stage 1 (convert)**: Raw dataset(s) → Unified pickles
   - Dataset-specific parsers convert native formats to standardized format
   - Supports multiple datasets simultaneously
   - Output: Pickle files with unified scenario data
   - Optional: `--validate` for soft validation during conversion

2. **Stage 2 (process)**: Unified → Enhanced (Optional)
   - Apply transformations, filtering, or augmentation
   - **File-by-file streaming**: Each worker loads one pkl → processes → saves → next file
   - Available processors: `validation`, `traffic_lights`, `polyline_interpolation`
   - Each processor is any function that takes/returns UnifiedScenario
   - No batch loading - memory efficient

3. **Stage 3 (format)**: Unified → Target format
   - **File-by-file streaming**: Each worker loads one pkl → formats → saves → next file
   - Converts to training-ready formats (TFRecord, JSON, Puffer)
   - Auto-detects single vs multi-dataset structure
   - Handles merging, shuffling, and sharding
   - No batch loading - memory efficient
   - Supported formats: `tfexample`, `json`, `puffer`

### Full Pipeline Mode

The `pipeline` command runs all 3 stages together:

- **Default (save_intermediate=False)**: File-by-file streaming with no intermediate saves
  - Each worker: Raw file → Unified → Process → Format → Save
  - Minimizes disk I/O and memory usage for TB-scale datasets

- **Optional (--save-intermediate)**: Saves intermediate pickles for debugging
  - Creates `_unified` and `_processed` directories
  - Useful for inspecting intermediate results

### Additional Tools

- **viz**: Visualize unified scenarios as Bird's Eye View (BEV) PNG images
- **convert**: Run Stage 1 only
- **process**: Run Stage 2 only
- **format**: Run Stage 3 only
- **pipeline**: Run all stages together

### Key Components

- **`scenariomax/core/pipeline.py`**: Main pipeline with 4 clean functions (~650 lines)
  - `convert_raw_to_unified()`: Stage 1
  - `process_unified_scenarios()`: Stage 2 (file-by-file)
  - `format_unified_to_target()`: Stage 3 (file-by-file)
  - `run_all_pipeline()`: Full pipeline
- **`scenariomax/dataset_registry.py`**: Dataset configuration registry
- **`scenariomax/stage1_convert/`**: Dataset-specific extractors and converters
  - `datasets/waymo/`: Waymo Open Motion Dataset
  - `datasets/nuplan/`: nuPlan dataset
  - `datasets/nuscenes/`: nuScenes (WIP)
  - `datasets/openscenes/`: OpenScenes dataset
- **`scenariomax/stage2_process/`**: Enhancement processors
  - `core.py`: Main processing function
  - `validate.py`: Validation processor
  - `traffic_lights/`: Traffic light inference
- **`scenariomax/stage3_format/`**: Target format converters
  - `tfexample/`: TFRecord format
  - `json/`: JSON format (GPUDrive)
  - `puffer/`: Puffer format (PufferDrive)
- **`scenariomax/visualization/`**: BEV rendering with matplotlib
- **`scenariomax/core/unified_scenario.py`**: UnifiedScenario schema definition

### UnifiedScenario Format

The intermediate format used by ScenarioMax, defined in `scenariomax/core/unified_scenario.py`:

```python
UnifiedScenario (dict subclass):
  - "id": str                                    # Unique scenario identifier
  - "dynamic_agents": {                          # Moving objects (vehicles, pedestrians, cyclists)
      obj_id: {
        type,                                    # Agent type
        states: {position, heading, velocity...} # Timestep states
      }
    }
  - "static_map_elements": {                     # Road infrastructure
      element_id: {
        type,                                    # Lane, crosswalk, etc.
        polyline,                                # Geometry
        speed_limit...                           # Properties
      }
    }
  - "dynamic_map_elements": {                    # Traffic lights, stop signs
      element_id: {
        type,                                    # Element type
        position,                                # Location
        states,                                  # Time-varying states
        lane                                     # Associated lane
      }
    }
  - "metadata": {                                # Scenario information
      dataset_name,                              # Source dataset
      dataset_version,                           # Dataset version
      length,                                    # Duration (seconds)
      timesteps,                                 # Number of timesteps
      ego_id...                                  # Ego vehicle ID
    }
```

### Processing Architecture Details

**File-by-file streaming principles** (from FEATURES.md):

1. **No batch loading**: Each worker processes files individually
   - Load one file → Process it → Save it → Move to next file
   - Workers never load all files into memory at once

2. **Worker independence**: Each worker is responsible for:
   - Opening and loading files
   - Processing/formatting data
   - Saving results to their specific folder

3. **Merging**: After all workers complete, results are merged:
   - Each worker dumps results into their specific folder
   - Post-processing merges all worker outputs
   - For TFRecord: merging, shuffling, and optional sharding

4. **Memory efficiency**: Scales to TB-scale datasets
   - No memory bottlenecks from batch loading
   - Each scenario processed independently
   - Constant memory usage per worker

**Example workflow** for Stage 2 + Stage 3:
```
Worker 1: Load pkl → Process → Format → Save → Next file
Worker 2: Load pkl → Process → Format → Save → Next file
Worker N: Load pkl → Process → Format → Save → Next file
[All workers complete]
Post-processing: Merge worker outputs → Shuffle → Shard (if TFRecord)
```

## ✅ Validation System

ScenarioMax provides a two-level validation system for UnifiedScenario objects:

### Soft Validation (Structural Checks)

Fast, lightweight validation that checks data structure:

```bash
# During conversion
scenariomax convert --waymo_src /data/waymo --dst /output --validate

# During processing
scenariomax process --src /output/unified --dst /output/processed --validate

# Validation-only mode (no output)
scenariomax process --src /output/unified --dst /tmp --validate --no-output
```

**Checks performed:**
- Required/optional keys
- Type checking for all fields
- Array shape verification
- No performance overhead

### Strict Validation (Physics-Based Checks)

Comprehensive validation that checks physical consistency:

```bash
# During processing
scenariomax process --src /output/unified --dst /output/processed --validate-strict

# In pipeline
scenariomax pipeline --waymo_src /data/waymo --dst /output --validate-strict
```

**Checks performed:**
- Trajectory coherence (position, velocity, acceleration)
- Map topology consistency
- Agent-agent interactions
- More comprehensive but slower

**Programmatic usage:**

```python
from scenariomax.core.validation import soft_validate, strict_validate
from scenariomax.core.unified_scenario import UnifiedScenario

# Soft validation
try:
    soft_validate(scenario)
    print("✓ Scenario structure is valid")
except ValidationError as e:
    print(f"✗ Validation failed: {e}")

# Strict validation
try:
    strict_validate(scenario)
    print("✓ Scenario physics is valid")
except ValidationError as e:
    print(f"✗ Validation failed: {e}")
```

**Location**: `scenariomax/core/validation.py` and `scenariomax/stage2_process/validate.py`

## 🚀 Production Features

### Checkpointing - Resume Interrupted Pipelines

```bash
# Enable checkpointing
scenariomax pipeline \
  --waymo_src /data/waymo \
  --dst /output \
  --format tfexample \
  --checkpoint

# If interrupted, re-run the same command - it will resume from checkpoint
scenariomax pipeline \
  --waymo_src /data/waymo \
  --dst /output \
  --format tfexample \
  --checkpoint
```

### Validation Reports - Detailed Error Statistics

```bash
# Generate validation_report.json with error statistics
scenariomax pipeline \
  --waymo_src /data/waymo \
  --dst /output \
  --format tfexample \
  --validation-report
```

**Report includes:**
- Total scenarios validated
- Pass/fail rates
- Top 10 most common errors
- List of failed scenarios

### Disk Space Checks - Automatic Verification

```bash
# Enabled by default - checks if sufficient disk space available
scenariomax pipeline \
  --waymo_src /data/waymo \
  --dst /output \
  --format tfexample

# Skip disk check if needed (not recommended for production)
scenariomax pipeline \
  --waymo_src /data/waymo \
  --dst /output \
  --format tfexample \
  --no-disk-check
```

### Deterministic Shuffling

Control shuffle seed for deterministic TFRecord shuffling:

```bash
export SCENARIOMAX_SHUFFLE_SEED=42
scenariomax pipeline --waymo_src /data/waymo --dst /output --format tfexample
```
```

## 🔧 Development

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_pipeline.py -v           # Pipeline tests
pytest tests/test_soft_validation.py -v    # Soft validation tests
pytest tests/test_strict_validation.py -v  # Strict validation tests
pytest tests/test_integration.py -v        # Integration tests
```

### Linting and Code Quality

```bash
# Linting with ruff
ruff check scenariomax/
ruff format scenariomax/

# Type checking
mypy scenariomax/
```

### Development Conventions

- **Package manager**: `uv` (fast Python package installer)
- **Linting**: `ruff` configured in `ruff.toml` (line length: 120)
- **Testing**: `pytest`
- **Python version**: 3.10+ (strict requirement)
- **TensorFlow**: Version 2.11.1 (for TFRecord compatibility)

### Code Style

- 2 lines after imports (enforced by ruff isort)
- Max complexity: 30 (mccabe)
- Excludes: `devkit/`, `waymo_protos/`

### Logging

- Uses custom logger: `scenariomax.logger_utils.get_logger(__name__)`
- Suppresses TensorFlow logs (set to FATAL level)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- CLI flags: `--log_level` and `--log_file`

### Cleanup

```bash
make clean  # Remove .venv, egg-info, __pycache__, *.pyc
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

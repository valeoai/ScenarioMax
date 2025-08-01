# ScenarioMax API Reference

This document provides comprehensive API documentation for ScenarioMax's core components and modules.

## Table of Contents

- [Core Pipeline API](#core-pipeline-api)
- [Dataset Registry API](#dataset-registry-api)
- [Conversion APIs](#conversion-apis)
- [Utility APIs](#utility-apis)
- [Exception Handling](#exception-handling)
- [Configuration](#configuration)

## Core Pipeline API

### `scenariomax.core.pipeline`

The main orchestrator for dataset conversion pipelines.

#### `convert_dataset()`

```python
def convert_dataset(
    source_type: str,
    source_paths: dict[str, str],
    target_format: str,
    output_path: str,
    enhancement: bool = False,
    stream_mode: bool = False,
    num_workers: int = 8,
    **kwargs
) -> dict[str, Any]
```

**Parameters:**
- `source_type` (str): Type of source data (`"raw"` or `"pickle"`)
- `source_paths` (dict): Dictionary mapping dataset names to paths
- `target_format` (str): Output format (`"pickle"`, `"tfexample"`, or `"gpudrive"`)
- `output_path` (str): Destination directory for converted data
- `enhancement` (bool): Enable scenario enhancement pipeline
- `stream_mode` (bool): Use streaming processing (memory efficient)
- `num_workers` (int): Number of parallel workers

**Returns:**
- `dict`: Processing statistics including scenario counts and timing

**Example:**
```python
from scenariomax.core import pipeline

stats = pipeline.convert_dataset(
    source_type="raw",
    source_paths={"waymo": "/data/waymo", "nuscenes": "/data/nuscenes"},
    target_format="tfexample",
    output_path="/output",
    enhancement=True,
    num_workers=16
)

print(f"Processed {stats['total_scenarios']} scenarios in {stats['processing_time']:.2f}s")
```

## Dataset Registry API

### `scenariomax.dataset_registry`

Dynamic dataset configuration and loading system.

#### `DatasetConfig`

```python
@dataclass
class DatasetConfig:
    name: str
    version: str
    load_func: Callable
    convert_func: Callable
    preprocess_func: Callable | None = None
    additional_args: dict = None
```

**Attributes:**
- `name`: Dataset identifier
- `version`: Dataset version
- `load_func`: Function to load raw scenarios
- `convert_func`: Function to convert to unified format
- `preprocess_func`: Optional preprocessing function
- `additional_args`: Additional configuration parameters

#### `get_dataset_config()`

```python
def get_dataset_config(dataset_name: str) -> DatasetConfig
```

**Parameters:**
- `dataset_name` (str): Name of the dataset (`"waymo"`, `"nuscenes"`, `"nuplan"`, `"openscenes"`)

**Returns:**
- `DatasetConfig`: Configuration object for the specified dataset

**Raises:**
- `UnsupportedDatasetError`: If dataset is not supported

**Example:**
```python
from scenariomax.dataset_registry import get_dataset_config

config = get_dataset_config("waymo")
print(f"Dataset: {config.name} v{config.version}")

# Load scenarios using the config
scenarios = config.load_func("/path/to/waymo/data")
```

## Conversion APIs

### Raw to Unified Format

#### Waymo Dataset (`scenariomax.raw_to_unified.datasets.waymo`)

```python
def get_waymo_scenarios(
    data_path: str, 
    start_index: int = 0, 
    num: int | None = None
) -> list[str]
```

```python
def convert_waymo_scenario(scenario: Any, version: str) -> dict[str, Any]
```

```python
def preprocess_waymo_scenarios(
    files: list[str], 
    worker_index: int
) -> Generator[Any, None, None]
```

#### nuScenes Dataset (`scenariomax.raw_to_unified.datasets.nuscenes`)

```python
def get_nuscenes_scenarios(
    dataroot: str, 
    version: str, 
    num_workers: int = 2
) -> list[Any]
```

```python
def convert_nuscenes_scenario(scenario: Any, version: str) -> dict[str, Any]
```

```python
def preprocess_nuscenes_scenarios(
    scenarios: list[Any], 
    worker_index: int
) -> list[Any]
```

#### nuPlan Dataset (`scenariomax.raw_to_unified.datasets.nuplan`)

```python
def get_nuplan_scenarios(
    data_path: str,
    **kwargs
) -> list[Any]
```

```python
def convert_nuplan_scenario(scenario: Any, version: str) -> dict[str, Any]
```

### Unified to Target Format

#### TFExample Conversion (`scenariomax.unified_to_tfexample`)

```python
def convert_to_tfexample(
    scenario_net_scene: dict[str, Any],
    multiagents: bool = True
) -> Any  # tf.train.Example
```

**Parameters:**
- `scenario_net_scene`: Unified scenario dictionary
- `multiagents`: Enable multi-agent processing

**Returns:**
- TensorFlow Example proto

#### GPUDrive Conversion (`scenariomax.unified_to_gpudrive`)

```python
def convert_to_json(scenario: dict[str, Any]) -> dict[str, Any]
```

**Parameters:**
- `scenario`: Unified scenario dictionary

**Returns:**
- JSON-serializable dictionary for GPUDrive

## Utility APIs

### Progress Monitoring (`scenariomax.progress_utils`)

```python
def create_progress_bar(
    iterable: Iterable,
    desc: str = "Processing",
    show_system_stats: bool = False,
    **kwargs
) -> tqdm
```

**Parameters:**
- `iterable`: Items to iterate over
- `desc`: Progress bar description
- `show_system_stats`: Show CPU/RAM usage
- `**kwargs`: Additional tqdm parameters

**Example:**
```python
from scenariomax import progress_utils

pbar = progress_utils.create_progress_bar(
    scenarios,
    desc="Converting scenarios",
    show_system_stats=True
)

for scenario in pbar:
    # Process scenario
    pass
```

### Memory Utilities (`scenariomax.raw_to_unified.datasets.utils`)

```python
def process_memory() -> dict[str, float]
```

**Returns:**
- Dictionary with memory usage statistics (RSS, VMS, percent)

```python
def get_system_memory() -> dict[str, float]
```

**Returns:**
- Dictionary with system memory information

### TensorFlow Utilities (`scenariomax.tf_utils`)

```python
def get_tensorflow() -> Any
```

**Returns:**
- TensorFlow module with optimized configuration

```python
def create_tf_feature_functions() -> dict[str, Callable]
```

**Returns:**
- Dictionary of TensorFlow feature creation functions

## Exception Handling

### Base Exception

```python
class ScenarioMaxError(Exception):
    def __init__(self, message: str, context: dict = None):
        self.message = message
        self.context = context or {}
```

### Dataset Exceptions

```python
class UnsupportedDatasetError(DatasetError):
    def __init__(self, dataset_name: str, supported_datasets: list[str] = None):
        # Dataset not supported
        pass

class DatasetLoadError(DatasetError):
    def __init__(self, dataset_name: str, dataset_path: str, reason: str = None):
        # Failed to load dataset
        pass

class EmptyDatasetError(DatasetError):
    def __init__(self, dataset_name: str, dataset_path: str):
        # Dataset contains no scenarios
        pass
```

### Conversion Exceptions

```python
class UnsupportedFormatError(ConversionError):
    def __init__(self, format_name: str, supported_formats: list[str] = None):
        # Format not supported
        pass

class ScenarioConversionError(ConversionError):
    def __init__(self, scenario_id: str, dataset_name: str, reason: str = None):
        # Individual scenario conversion failed
        pass

class WorkerProcessingError(ConversionError):
    def __init__(self, worker_index: int, reason: str = None):
        # Worker failed during parallel processing
        pass
```

### Validation Exceptions

```python
class InvalidScenarioDataError(ValidationError):
    def __init__(self, scenario_id: str, field_name: str, reason: str = None):
        # Scenario data validation failed
        pass

class OverpassException(ValidationError):
    def __init__(self, scenario_id: str = None, message: str = "Overpass detected"):
        # Overpass detected in roadgraph
        pass

class NotEnoughValidObjectsException(ValidationError):
    def __init__(self, scenario_id: str = None, message: str = "Not enough valid objects"):
        # Insufficient objects for multi-agent processing
        pass
```

**Example Error Handling:**
```python
from scenariomax.core.exceptions import (
    DatasetLoadError,
    ScenarioConversionError,
    UnsupportedFormatError
)

try:
    stats = pipeline.convert_dataset(
        source_type="raw",
        source_paths={"waymo": "/invalid/path"},
        target_format="tfexample",
        output_path="/output"
    )
except DatasetLoadError as e:
    print(f"Failed to load dataset: {e}")
    print(f"Context: {e.context}")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except ScenarioConversionError as e:
    print(f"Scenario conversion failed: {e}")
```

## Configuration

### Environment Variables

```bash
# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=3          # Suppress TensorFlow logs
TF_ENABLE_ONEDNN_OPTS=0         # Disable oneDNN optimizations
CUDA_VISIBLE_DEVICES=""         # Disable GPU for data processing

# nuPlan Configuration
NUPLAN_MAPS_ROOT=/path/to/maps  # nuPlan maps directory
NUPLAN_DATA_ROOT=/path/to/data  # nuPlan data directory
```

### Command Line Interface

The main CLI is accessible via `scenariomax.convert_dataset`:

```python
import argparse

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    # Returns configured ArgumentParser
```

**Key Arguments:**
- `--waymo_src`, `--nuplan_src`, `--nuscenes_src`: Dataset source paths
- `--dst`: Output directory (required)
- `--target_format`: Output format (`pickle`, `tfexample`, `gpudrive`)
- `--num_workers`: Number of parallel workers (default: 8)
- `--shard`: Number of output shards (default: 1)
- `--enable_enhancement`: Enable scenario enhancement
- `--log_level`: Logging verbosity
- `--split`: nuScenes data split

## Type Definitions

### Core Types (`scenariomax.core.types`)

```python
# Scenario description constants
class ScenarioDescription:
    VERSION: str = "version"
    ID: str = "id"
    METADATA: str = "metadata"
    DYNAMIC_AGENTS: str = "dynamic_agents"
    STATIC_AGENTS: str = "static_agents"
    MAP_ELEMENTS: str = "map_elements"
    DYNAMIC_MAP_ELEMENTS: str = "dynamic_map_elements"
    TIMESTEPS: str = "timesteps"

# Agent types
class AgentType:
    VEHICLE: str = "vehicle"
    PEDESTRIAN: str = "pedestrian"
    CYCLIST: str = "cyclist"
    UNKNOWN: str = "unknown"

# Map element types
class MapElementType:
    LANE: str = "lane"
    CROSSWALK: str = "crosswalk"
    SPEED_BUMP: str = "speed_bump"
    DRIVEWAY: str = "driveway"
    STOP_SIGN: str = "stop_sign"
```

### Unified Scenario (`scenariomax.core.unified_scenario`)

```python
class UnifiedScenario(dict):
    """
    Unified scenario representation with validation and utility methods.
    """
    
    def sanity_check(self) -> None:
        """Validate scenario data integrity."""
        pass
    
    @property
    def export_file_name(self) -> str:
        """Generate export filename for scenario."""
        pass
    
    def get_scenario_length(self) -> int:
        """Get number of timesteps in scenario."""
        pass
```

## Advanced Usage Examples

### Custom Dataset Integration

```python
from scenariomax.dataset_registry import DatasetConfig

def load_custom_scenarios(data_path: str) -> list[Any]:
    # Custom loading logic
    pass

def convert_custom_scenario(scenario: Any, version: str) -> dict[str, Any]:
    # Custom conversion logic
    pass

# Register custom dataset
custom_config = DatasetConfig(
    name="custom",
    version="1.0",
    load_func=load_custom_scenarios,
    convert_func=convert_custom_scenario
)
```

### Parallel Processing with Custom Functions

```python
from scenariomax.core.write import write_to_directory

def custom_convert_func(scenario, dataset_version, dataset_name, **kwargs):
    # Custom conversion logic
    return converted_scenario

def custom_postprocess_func(
    worker_index: int,
    scenarios: list,
    output_path: str,
    dataset_name: str,
    dataset_version: str,
    convert_func: Callable,
    **kwargs
):
    # Custom postprocessing logic
    pass

write_to_directory(
    convert_func=custom_convert_func,
    scenarios=scenario_list,
    output_path="/output",
    dataset_version="1.0",
    dataset_name="custom",
    num_workers=8,
    postprocess_func=custom_postprocess_func
)
```

### Stream Processing

```python
from scenariomax.core import pipeline

# Enable streaming for memory-efficient processing
stats = pipeline.convert_dataset(
    source_type="raw",
    source_paths={"waymo": "/large/dataset"},
    target_format="tfexample",
    output_path="/output",
    stream_mode=True,  # Process without intermediate pickle files
    num_workers=8
)
```

This API reference provides comprehensive coverage of ScenarioMax's public interfaces. For implementation details and examples, refer to the source code and additional documentation.
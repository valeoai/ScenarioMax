# Dataset Specifications

This document provides detailed information about each supported dataset in ScenarioMax, including data formats, setup instructions, and conversion specifics.

## Table of Contents

- [Overview](#overview)
- [Waymo Open Motion Dataset](#waymo-open-motion-dataset)
- [nuScenes Dataset](#nuscenes-dataset)
- [nuPlan Dataset](#nuplan-dataset)
- [OpenScenes Dataset](#openscenes-dataset)
- [Dataset Comparison](#dataset-comparison)
- [Common Issues](#common-issues)

## Overview

ScenarioMax supports four major autonomous driving datasets, each with unique characteristics and data formats. The following sections provide comprehensive setup and usage information for each dataset.

### Supported Formats

| Dataset | Raw Format | Unified Schema Support | TFExample Output | GPUDrive Output |
|---------|------------|----------------------|------------------|-----------------|
| Waymo | TFRecord | ✅ | ✅ | ✅ |
| nuScenes | JSON + Binary | ✅ | ✅ | ✅ |
| nuPlan | Database | ✅ | ✅ | ✅ |
| OpenScenes | Pickle | ✅ | ✅ | ✅ |

## Waymo Open Motion Dataset

### Overview

The Waymo Open Motion Dataset (WOMD) contains real-world autonomous driving scenarios with high-definition maps, agent trajectories, and traffic light states.

**Key Features**:
- 103,354 scenarios with 9 seconds of trajectory data
- High-definition maps with lane connectivity
- Multi-agent interactions (vehicles, pedestrians, cyclists)
- Traffic light states and stop sign information
- Rich metadata including weather and time of day

### Data Structure

```python
# Raw Waymo scenario (TFRecord format)
{
    "scenario/id": bytes,                    # Unique scenario identifier
    "state/x": [timesteps, agents],         # Agent x positions
    "state/y": [timesteps, agents],         # Agent y positions  
    "state/heading": [timesteps, agents],   # Agent headings
    "state/velocity_x": [timesteps, agents], # Agent x velocities
    "state/velocity_y": [timesteps, agents], # Agent y velocities
    "state/valid": [timesteps, agents],     # Validity mask
    "state/type": [agents],                 # Agent types
    "roadgraph_samples/xyz": [samples, 3],  # Map polylines
    "roadgraph_samples/type": [samples],    # Map element types
    "traffic_light_state/current_state": [...], # Traffic light states
    "tracks_to_predict": [...],             # Prediction targets
}
```

### Setup Instructions

1. **Download Dataset**:
   ```bash
   # Visit https://waymo.com/open/download/
   # Download training/validation TFRecord files
   ```

2. **Install Dependencies**:
   ```bash
   make womd
   # OR
   uv pip install -e ".[womd]"
   ```

3. **Environment Setup**:
   ```bash
   # No special environment variables required
   export TF_CPP_MIN_LOG_LEVEL=3  # Optional: suppress TF logs
   ```

### Usage Examples

```bash
# Basic conversion
python -m scenariomax.convert_dataset \
  --waymo_src /path/to/waymo/tfrecords \
  --dst /output \
  --target_format tfexample \
  --num_workers 8

# Convert specific number of files
python -m scenariomax.convert_dataset \
  --waymo_src /path/to/waymo/tfrecords \
  --dst /output \
  --target_format gpudrive \
  --num_files 100

# Convert to unified format first
python -m scenariomax.convert_dataset \
  --waymo_src /path/to/waymo/tfrecords \
  --dst /intermediate \
  --target_format pickle
```

### Memory Optimization

Waymo datasets can be large. ScenarioMax includes specific optimizations:

```python
def preprocess_waymo_scenarios(files, worker_index):
    """Generator-based processing for memory efficiency."""
    for file in files:
        # Process one file at a time
        yield from load_waymo_scenarios_from_file(file)
```

### Waymo-Specific Configuration

```python
# Access Waymo configuration
from scenariomax.dataset_registry import get_dataset_config

config = get_dataset_config("waymo")
# config.name = "womd"
# config.version = "v1.3"
```

## nuScenes Dataset

### Overview

The nuScenes dataset provides full autonomous vehicle sensor suite data with 1000 scenes in diverse conditions.

**Key Features**:
- 1000 scenes with 20-second duration each
- Full sensor suite (6 cameras, 1 lidar, 5 radars)
- Dense 3D object annotations
- Rich map data with semantic segmentation
- Diverse weather and lighting conditions

### Data Structure

```python
# Raw nuScenes scenario structure
{
    "scene": {
        "token": str,           # Unique scene identifier
        "name": str,            # Scene name
        "nbr_samples": int,     # Number of samples
        "first_sample_token": str,
        "last_sample_token": str
    },
    "samples": [
        {
            "token": str,       # Sample identifier
            "timestamp": int,   # Unix timestamp
            "scene_token": str, # Reference to scene
            "data": {...}       # Sensor data references
        }
    ],
    "sample_annotations": [
        {
            "token": str,           # Annotation identifier
            "instance_token": str,  # Object instance
            "category_name": str,   # Object category
            "translation": [x, y, z], # 3D position
            "size": [w, l, h],      # 3D dimensions
            "rotation": [w, x, y, z] # Quaternion
        }
    ],
    "maps": [...]  # Map data
}
```

### Setup Instructions

1. **Download Dataset**:
   ```bash
   # Visit https://www.nuscenes.org/nuscenes
   # Download Full dataset (v1.0)
   # Extract to /data/nuscenes/
   ```

2. **Install Dependencies**:
   ```bash
   make nuscenes
   # OR
   uv pip install -e ".[nuscenes]"
   ```

3. **Verify Installation**:
   ```python
   from nuscenes.nuscenes import NuScenes
   nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes')
   print(f"Loaded {len(nusc.scene)} scenes")
   ```

### Usage Examples

```bash
# Convert with specific split
python -m scenariomax.convert_dataset \
  --nuscenes_src /data/nuscenes \
  --dst /output \
  --target_format tfexample \
  --split v1.0-trainval

# Convert mini dataset for testing
python -m scenariomax.convert_dataset \
  --nuscenes_src /data/nuscenes \
  --dst /output \
  --target_format pickle \
  --split v1.0-mini

# Convert to GPUDrive format
python -m scenariomax.convert_dataset \
  --nuscenes_src /data/nuscenes \
  --dst /output \
  --target_format gpudrive \
  --split v1.0-trainval
```

### nuScenes-Specific Features

**Prediction Split**: nuScenes includes a prediction challenge split:

```python
def get_nuscenes_prediction_split(dataroot, version, past, future, num_workers=2):
    """Get scenarios formatted for prediction tasks."""
    # Implementation in nuscenes/load.py
```

**Map Integration**: Rich HD map data with lane connectivity:

```python
# Access map data
from nuscenes.map_expansion.map_api import NuScenesMap

nusc_map = NuScenesMap(dataroot='/data/nuscenes', map_name='singapore-onenorth')
```

### nuScenes Configuration

```python
# Dataset configuration
config = get_dataset_config("nuscenes")
# config.name = "nuscenes"
# config.version = "v1.0"
# config.preprocess_func = nuscenes.preprocess_nuscenes_scenarios
```

## nuPlan Dataset

### Overview

nuPlan is a large-scale planning-centric dataset with 1500+ hours of human driving data from 4 cities.

**Key Features**:
- 1500+ hours of driving data
- 4 cities: Las Vegas, Boston, Pittsburgh, Singapore  
- Rich planning annotations
- Complex urban scenarios
- Database format with efficient querying

### Data Structure

```python
# nuPlan database schema (simplified)
{
    "scenario": {
        "token": str,           # Scenario identifier
        "map_version": str,     # Map version
        "initial_lidar_token": str,
        "scenario_type": str,   # Scenario classification
        "duration": float       # Scenario duration (seconds)
    },
    "lidar_pc": {
        "token": str,           # Lidar pointcloud identifier
        "timestamp": int,       # Unix timestamp
        "ego_pose_token": str,  # Reference to ego pose
        "filename": str         # Path to pointcloud file
    },
    "ego_pose": {
        "token": str,           # Ego pose identifier
        "timestamp": int,       # Unix timestamp
        "x": float, "y": float, "z": float,        # Position
        "qw": float, "qx": float, "qy": float, "qz": float  # Quaternion
    },
    "tracks": [...]  # Object tracks
}
```

### Setup Instructions

1. **Download Dataset**:
   ```bash
   # Visit https://www.nuplan.org/
   # Download sensor_blobs, maps, and nuplan-v1.1
   ```

2. **Install Dependencies**:
   ```bash
   make nuplan
   # OR
   uv pip install -e ".[nuplan]"
   ```

3. **Environment Setup**:
   ```bash
   # Required environment variables
   export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset
   export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
   
   # Optional: Disable GPU for data processing
   export CUDA_VISIBLE_DEVICES=""
   ```

4. **Verify Setup**:
   ```python
   import os
   from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
   
   data_root = os.environ['NUPLAN_DATA_ROOT']
   db_files = [f for f in os.listdir(data_root) if f.endswith('.db')]
   print(f"Found {len(db_files)} database files")
   ```

### Usage Examples

```bash
# Basic conversion
python -m scenariomax.convert_dataset \
  --nuplan_src $NUPLAN_DATA_ROOT \
  --dst /output \
  --target_format tfexample

# Direct log parsing (alternative method)
python -m scenariomax.convert_dataset \
  --nuplan_src $NUPLAN_DATA_ROOT \
  --dst /output \
  --target_format gpudrive \
  --nuplan_direct_from_logs

# Convert specific scenarios
python -m scenariomax.convert_dataset \
  --nuplan_src $NUPLAN_DATA_ROOT \
  --dst /output \
  --target_format pickle \
  --num_files 50
```

### nuPlan-Specific Features

**Scenario Filtering**: nuPlan supports advanced scenario filtering:

```python
# Configure scenario filters
scenario_filter_config = {
    "scenario_types": ["following_lane_with_lead", "stopping_with_lead"],
    "num_scenarios_per_type": 100,
    "limit_total_scenarios": 1000
}
```

**Map Integration**: Rich HD maps with semantic information:

```python
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

maps_db = GPKGMapsDB(maps_root_path, map_version="nuplan-maps-v1.0")
```

### nuPlan Configuration

```python
# Dataset configuration includes complex setup
config = get_dataset_config("nuplan")
# Uses nuPlan's scenario builder infrastructure
# Supports both database queries and direct log parsing
```

## OpenScenes Dataset

### Overview

OpenScenes provides synthetic scenarios based on nuPlan infrastructure, designed for scalable scenario generation.

**Key Features**:
- nuPlan-compatible format
- Synthetic scenario generation
- Pickle-based storage for efficiency
- Compatible with nuPlan tools and infrastructure

### Data Structure

```python
# OpenScenes scenario (pickle format)
{
    "scenario_id": str,          # Unique identifier
    "metadata": {
        "dataset_name": "openscenes",
        "source": "nuplan_derived",
        "generation_method": str
    },
    "nuplan_scenario": NuPlanScenario  # Native nuPlan scenario object
}
```

### Setup Instructions

1. **Prerequisites**:
   ```bash
   # Requires nuPlan installation
   make nuplan
   ```

2. **Environment Setup**:
   ```bash
   # Same as nuPlan
   export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset  
   export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
   ```

3. **OpenScenes Metadata**:
   ```bash
   # If using OpenScenes-specific metadata
   python -m scenariomax.convert_dataset \
     --nuplan_src $NUPLAN_DATA_ROOT \
     --openscenes_metadata_src /path/to/openscenes/metadata \
     --dst /output \
     --target_format tfexample
   ```

### Usage Examples

```bash
# Convert OpenScenes scenarios
python -m scenariomax.convert_dataset \
  --nuplan_src $NUPLAN_DATA_ROOT \
  --openscenes_metadata_src /path/to/metadata \
  --dst /output \
  --target_format gpudrive

# Process as standard nuPlan (if no special metadata)
python -m scenariomax.convert_dataset \
  --nuplan_src $NUPLAN_DATA_ROOT \
  --dst /output \
  --target_format pickle
```

### OpenScenes Configuration

```python
# OpenScenes uses nuPlan converter with different loading
config = get_dataset_config("openscenes")
# config.load_func = openscenes.get_openscenes_scenarios  
# config.convert_func = nuplan.convert_nuplan_scenario  # Reuses nuPlan converter
```

## Dataset Comparison

### Scale and Coverage

| Dataset | Scenarios | Duration | Cities/Locations | Real/Synthetic |
|---------|-----------|----------|------------------|----------------|
| **Waymo** | 103K+ | 9s each | Multiple US cities | Real |
| **nuScenes** | 1K scenes | 20s each | Boston, Singapore | Real |
| **nuPlan** | 10K+ hours | Variable | 4 cities (US, Asia) | Real |
| **OpenScenes** | Variable | Variable | nuPlan-based | Synthetic |

### Data Characteristics

| Feature | Waymo | nuScenes | nuPlan | OpenScenes |
|---------|-------|----------|--------|------------|
| **HD Maps** | ✅ | ✅ | ✅ | ✅ |
| **Traffic Lights** | ✅ | ❌ | ✅ | ✅ |
| **Multi-Agent** | ✅ | ✅ | ✅ | ✅ |
| **Planning Focus** | ❌ | ❌ | ✅ | ✅ |
| **Sensor Data** | ❌ | ✅ | ✅ | ❌ |
| **Weather Variety** | ✅ | ✅ | ✅ | Variable |

### Computational Requirements

| Dataset | Memory Usage | Storage Size | Processing Time |
|---------|--------------|--------------|-----------------|
| **Waymo** | High | ~1TB | Fast |
| **nuScenes** | Medium | ~350GB | Medium |
| **nuPlan** | Low-Medium | ~5TB | Slow |
| **OpenScenes** | Low | Variable | Fast |

## Common Issues

### Dataset-Specific Problems

#### Waymo
```bash
# Issue: TensorFlow version conflicts
# Solution: Use exact TensorFlow version
uv pip install tensorflow==2.11.1

# Issue: Memory errors with large files
# Solution: Reduce workers or use streaming
--num_workers 4
--stream_mode
```

#### nuScenes
```bash
# Issue: Missing sensor data
# Solution: Download complete dataset
wget https://www.nuscenes.org/data/v1.0-trainval01_blobs.tgz

# Issue: Map loading errors  
# Solution: Verify map files
ls /data/nuscenes/maps/
```

#### nuPlan
```bash
# Issue: Environment variables not set
# Solution: Set required variables
export NUPLAN_DATA_ROOT=/path/to/data
export NUPLAN_MAPS_ROOT=/path/to/maps

# Issue: Database connection errors
# Solution: Check file permissions
chmod 644 /path/to/nuplan/*.db

# Issue: Ray multiprocessing errors
# Solution: Use single machine mode
--num_workers 1
```

#### OpenScenes
```bash  
# Issue: nuPlan dependency missing
# Solution: Install nuPlan support
make nuplan

# Issue: Metadata format mismatch
# Solution: Verify metadata structure
python -c "import pickle; print(pickle.load(open('metadata.pkl', 'rb')).keys())"
```

### General Troubleshooting

#### Memory Issues
```bash
# Monitor memory usage
python -c "
from scenariomax.raw_to_unified.datasets.utils import get_system_memory
print(get_system_memory())
"

# Reduce memory usage
--num_workers 2
export TF_ENABLE_ONEDNN_OPTS=0
```

#### Performance Issues
```bash
# Profile conversion
time python -m scenariomax.convert_dataset --waymo_src /data --dst /output --num_files 10

# Use faster storage
# Move datasets to SSD for better I/O performance

# Optimize workers
--num_workers $(nproc)  # Use all CPU cores
```

#### Validation Errors
```bash
# Enable detailed validation
python -c "
import logging
from scenariomax import logger_utils
logger_utils.setup_logger(log_level=logging.DEBUG)
"

# Check scenario structure
python -m scenariomax.scripts.debug --scenario_path scenario.pkl
```

This comprehensive dataset guide should help you successfully work with all supported datasets in ScenarioMax. For additional support, refer to the individual dataset documentation or the main ScenarioMax documentation.
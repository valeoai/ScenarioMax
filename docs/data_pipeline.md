# Understanding the ScenarioMax Data Pipeline

ScenarioMax provides a comprehensive pipeline for transforming autonomous driving datasets into formats compatible with various simulation environments. This document explains the data flow and transformation processes.

## Overview

The ScenarioMax pipeline consists of two main stages:

1. **Raw to Unified**: Converting raw dataset formats to a standardized pickle format
2. **Unified to Target Format**: Converting the unified format to specific output formats (TFExample or GPUDrive)

```
Raw Dataset → [raw_to_unified] → Unified Format (pkl) → [unified_to_tfexample / unified_to_gpudrive] → Target Format
```

## Stage 1: Raw to Unified

This stage converts dataset-specific formats into a standardized representation:

- **Input**: Raw dataset files (Waymo, nuPlan, nuScenes, etc.)
- **Process**:
  - Parse dataset-specific structures
  - Extract relevant features (agents, road network, etc.)
  - Standardize coordinate systems and units
  - Apply type corrections and data reordering
- **Output**: Pickle (.pkl) files with unified scenario representations

While similar to ScenarioNet, ScenarioMax's unified format includes specific adjustments to ensure compatibility with downstream processes.

## Stage 2: Unified to Target Format

This stage converts the unified format into specific output formats:

### TFExample Pipeline

- **Input**: Unified pickle files
- **Process**:
  - Convert Python objects to TensorFlow feature representations
  - Apply feature engineering for Waymax/V-Max compatibility
  - Optimize data layout for TensorFlow performance
- **Output**: TFRecord files compatible with Waymax and V-Max

### GPUDrive Pipeline

- **Input**: Unified pickle files
- **Process**:
  - Convert Python objects to GPUDrive-compatible JSON structures
  - Adjust coordinate systems and entity representations
- **Output**: JSON files compatible with GPUDrive

## Implementation Details

### Key Modules

- `raw_to_unified`: Dataset-specific converters that transform raw data to the unified format
- `unified_to_tfexample`: Components that convert unified data to TFExample format
- `unified_to_gpudrive`: Components that convert unified data to GPUDrive format

### Conversion Flow

1. `convert_dataset.py` orchestrates the entire process
2. Dataset-specific converters handle the first stage transformation
3. Format-specific converters handle the second stage transformation
4. Post-processing steps (sharding, merging) optimize the final output

## Advanced Usage

The pipeline is designed to be modular, allowing you to:

- Add support for new datasets by creating new dataset converters
- Create new output formats by implementing new format converters
- Customize post-processing steps for specific requirements

See the [Advanced Configuration Guide](docs/advanced_configuration.md) for more details.

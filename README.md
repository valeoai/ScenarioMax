# ScenarioMax

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ScenarioMax is an extension to [ScenarioNet](https://github.com/metadriverse/scenarionet) that transforms various autonomous driving datasets into standardized formats. Like ScenarioNet, it first converts different datasets (Waymo, nuPlan, nuScenes) to a unified pickle format. ScenarioMax then extends this process with additional pipelines to convert this unified data into formats compatible with [Waymax](https://github.com/waymo-research/waymax), [V-Max](https://github.com/valeoai/V-Max), and [GPUDrive](https://github.com/Emerge-Lab/gpudrive).

## Features

- **Data Unification Pipeline**: Similar to ScenarioNet, converts raw datasets to a standardized pickle format with type corrections and data reordering
- **Multiple Output Formats**:
  - **TFRecord/TFExample**: Generate optimized TFRecord files for Waymax and V-Max
  - **JSON**: Create compatible output for GPUDrive
- **SDC Path Support**: Add and evaluate multiple potential trajectories for self-driving vehicles
- **Extensible Architecture**: Easily add support for new datasets or output formats

## Data Processing Pipeline

ScenarioMax provides a complete data transformation workflow:

1. **Raw to Unified** (`raw_to_unified`):
   - Converts various datasets into a standardized pickle format
   - Applies type corrections and data reordering to ensure compatibility
   - Similar to ScenarioNet but with adjustments for downstream compatibility

2. **Unified to Target Format**:
   - **TFExample** (`unified_to_tfexample`): Converts unified data to TFRecord format for Waymax/V-Max
   - **GPUDrive** (`unified_to_gpudrive`): Converts unified data to JSON format for GPUDrive

This two-stage pipeline allows for flexible processing of autonomous driving datasets across different simulation platforms.

## Supported Datasets

| Dataset | Version | Link | Status |
|---------|---------|------|--------|
| Waymo Open Motion Dataset | v1.3.0 | [Site](https://waymo.com/open/download/) | ✅ Full Support |
| nuPlan | v1.1 | [Site](https://www.nuscenes.org/nuplan) | ✅ Full Support |
| nuScenes | v1.0 | [Site](https://www.nuscenes.org/nuscenes) | ✅ Full Support |
| Argoverse | v2.0 | [Site](https://www.argoverse.org/av2.html#forecasting-link) | 🚧 WIP |

For dataset setup, you can see the complete [ScenarioNet documention](https://scenarionet.readthedocs.io/en/latest/).

# Quick Start Guide

## Prerequisites

- Python 3.10 or newer
- Access to at least one supported dataset (Waymo, nuPlan, or nuScenes)
- Sufficient disk space for dataset processing

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scenariomax.git
cd scenariomax

# Install the package
pip install -e .
```

### With nuPlan Support

If you need to work with nuPlan data:

```bash
pip install -e devkit/nuplan-devkit
pip install -r devkit/nuplan-devkit/requirements.txt
```

## Basic Usage

### Converting Waymo Open Motion Dataset

```bash
python scenariomax/convert_dataset.py \
  --waymo_src /path/to/waymo/data \
  --dst /path/to/output/directory \
  --target_format tfexample \
  --num_workers 8 \
  --tfrecord_name training
```

### Converting nuScenes Dataset

```bash
python scenariomax/convert_dataset.py \
  --nuscenes_src /path/to/nuscenes/data \
  --dst /path/to/output/directory \
  --split v1.0-trainval \
  --target_format tfexample \
  --num_workers 8 \
  --tfrecord_name training
```

### Converting to GPUDrive Format

```bash
python scenariomax/convert_dataset.py \
  --waymo_src /path/to/waymo/data \
  --dst /path/to/output/directory \
  --target_format gpudrive \
  --num_workers 8
```

## Understanding the Output

After running the conversion process, you'll have:

- For `tfexample` format: TFRecord files compatible with Waymax and V-Max
- For `gpudrive` format: JSON files compatible with GPUDrive
- For `pickle` format: Standardized pickle files with unified scenario data. Pickle can be converted to a tfexample or a gpudrive afterwards.

## Documentation

- Check the [Advanced Configuration Guide](docs/advanced_configuration.md) for more options
- See [Dataset-Specific Notes](docs/dataset_notes.md) for details on handling each dataset

## Release Notes

### Version 1.0
- Initial implementation with support for Waymo, nuPlan, and nuScenes
- TFRecord conversion pipeline for efficient data handling
- SDC paths support
- GPUDrive format support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


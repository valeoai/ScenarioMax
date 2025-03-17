# ScenarioMax

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ScenarioMax is an extension to [ScenarioNet](https://github.com/metadriverse/scenarionet) that transforms converted data from [ScenarioNet](https://github.com/metadriverse/scenarionet) into a standardized TFRecord format. It currently supports real-world datasets like Waymo, nuPlan, and nuScenes through a unified interface, making the data compatible with both [Waymax](https://github.com/waymo-research/waymax) and [V-Max](https://github.com/valeoai/V-Max).

## Features

- **Unified Data Interface**: Access multiple autonomous driving datasets through a single API
- **Data Transformation Pipeline**: Convert raw scenario data into a standardized format
- **TFRecord Conversion**: Generate TFRecord files for efficient processing in Waymax and V-Max
- **Multi-path SDC Support**: Add and evaluate multiple potential trajectories for self-driving vehicles
- **Extensible Architecture**: Easily add support for new datasets

## Supported Datasets

| Dataset | Version | Link | Status |
|---------|---------|------|--------|
| Waymo Open Motion Dataset | v1.3.0 | [Site](https://waymo.com/open/download/) | ✅ Full Support |
| nuPlan | v1.1 | [Site](https://www.nuscenes.org/nuplan) | ✅ Full Support |
| nuScenes | v1.0 | [Site](https://www.nuscenes.org/nuscenes) | ✅ Full Support |
| Argoverse | v2.0 | [Site](https://www.argoverse.org/av2.html#forecasting-link) | 🚧 WIP |


For dataset setup, you can see the complete [ScenarioNet documention](https://scenarionet.readthedocs.io/en/latest/).

## Installation && Usage

Support 3.10

### Basic Installation
```bash
pip install -e .
```

### With nuPlan Support
```bash
pip install -e devkit/nuplan-devkit
pip install -r devkit/nuplan-devkit/requirements.txt
```

### Usage

```bash
python scenariomax/convert_dataset.py --waymo_src /data/datasets/womd/training/ --dst /data/test --log-level DEBUG --shard 10 --num-workers 10 --format tfexample
```

## Data Pipeline

ScenarioMax provides a streamlined workflow for:

1. **Loading** raw scenario data from various sources
2. **Processing** the data into a unified format
3. **Transforming** scenarios with additional features
4. **Exporting** to TFRecord format for use in simulation environments

## Release Notes

### Version 1.0
- Initial implementation with support for Waymo, nuPlan, and nuScenes
- TFRecord conversion pipeline for efficient data handling
- SDC multi-path augmentation capabilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For issues or feature requests, please open an issue on the repository or contact the project maintainers.

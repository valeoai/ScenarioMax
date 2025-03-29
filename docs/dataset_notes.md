# Dataset-Specific Notes

This guide provides important details and considerations for each of the supported datasets in ScenarioMax.

## Waymo Open Motion Dataset (WOMD)

### Dataset Overview
- Contains real-world driving scenarios collected by Waymo's autonomous vehicles
- Features high-quality trajectories and map data
- Version supported: v1.1, v1.2, v1.3

### Dataset Structure
The dataset is organized into TFRecord files containing serialized protobuf messages:
```
waymo/
├── training/
│   ├── scenario_0.tfrecord
│   ├── scenario_1.tfrecord
│   └── ...
├── validation/
└── testing/
```

### Conversion Notes
- To be filled

### Common Issues
- To be filled

## nuPlan

### Dataset Overview
- Real-world driving data collected in various cities worldwide
- Rich semantic map information
- Version supported: v1.1

### Dataset Structure
The dataset uses a SQLite database structure:
```
nuplan/
├── maps/
│   ├── sg-one-north/
│   ├── us-boston/
│   └── ...
├── nuplan-v1.1/
│   ├── mini/
│   ├── trainval/
│   └── test/
```

### Environment Variables
Ensure you set the following environment variable for map access:
```bash
export NUPLAN_DATA_ROOT=<path>/nuplan/dataset
export NUPLAN_MAPS_ROOT=<path>/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=<path>/nuplan/exp
```

### Conversion Notes
- Requires the nuPlan devkit for proper data loading
- Map data is converted to a standardized polyline format
- Agent types are standardized according to the unified schema

### Common Issues
- Map loading requires significant memory

## nuScenes

### Dataset Overview
- Dataset with multimodal sensor data and annotations
- Contains data from Boston and Singapore
- Version supported: v1.0

### Dataset Structure
```
nuscenes/
├── samples/
├── sweeps/
├── maps/
├── v1.0-trainval/
└── v1.0-test/
```

### Conversion Notes
- To be filled

### Common Issues
- To be filled

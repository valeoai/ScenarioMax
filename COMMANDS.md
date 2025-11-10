# ScenarioMax Testing Commands

Quick reference for testing all pipeline features. Copy-paste commands directly into terminal.

**Prerequisites:**
```bash
source .venv/bin/activate
cd /home/o-vcharrau/Workspace/ScenarioMax
```

---

## Stage 1: Convert (Raw → Unified)

### Waymo (WOMD)
```bash
# Basic conversion (1 file)
scenariomax command=convert datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/unified execution.num_workers=1 dataset_options.num_files=1

# Multi-worker conversion (10 files)
scenariomax command=convert datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/unified_batch execution.num_workers=4 dataset_options.num_files=10
```

### nuPlan
```bash
# Basic conversion (5 files)
scenariomax command=convert datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/nuplan/unified execution.num_workers=1 dataset_options.num_files=5

# Multi-worker conversion (20 files)
scenariomax command=convert datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/nuplan/unified_batch execution.num_workers=4 dataset_options.num_files=20
```

### Multi-Dataset Conversion
```bash
# Combine Waymo + nuPlan
scenariomax command=convert datasets.waymo=/data/datasets/womd/training/ datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/multi/unified execution.num_workers=8 dataset_options.num_files=10
```

---

## Stage 2: Process (Unified → Enhanced)

**Note:** All processors take unified pickles as input.

### Validation Processor

```bash
# Soft validation (structural checks only, no output save)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/tmp/validation_soft processing.processors=[validation] processing.processor_configs.validation.strict=false processing.save_output=false execution.num_workers=1

# Strict validation (physics checks, no output save)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/tmp/validation_strict processing.processors=[validation] processing.processor_configs.validation.strict=true processing.save_output=false execution.num_workers=1

# Soft validation with output save
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/validated processing.processors=[validation] processing.processor_configs.validation.strict=false execution.num_workers=1
```

### Traffic Lights Processor

```bash
# Add traffic lights to WOMD scenarios
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/with_traffic_lights processing.processors=[traffic_lights] execution.num_workers=1

# Add traffic lights to nuPlan scenarios
scenariomax command=process input_path=/data/debug/nuplan/unified output.dst=/data/debug/nuplan/with_traffic_lights processing.processors=[traffic_lights] execution.num_workers=2
```

### Polyline Interpolation Processor

```bash
# Interpolate polylines (default: 2.0m max segment length)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/interpolated processing.processors=[polyline_interpolation] processing.processor_configs.polyline_interpolation.max_segment_length=2.0 execution.num_workers=1

# Interpolate with 1.0m segments (denser)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/interpolated_dense processing.processors=[polyline_interpolation] processing.processor_configs.polyline_interpolation.max_segment_length=1.0 execution.num_workers=1

# Interpolate with 5.0m segments (sparser)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/interpolated_sparse processing.processors=[polyline_interpolation] processing.processor_configs.polyline_interpolation.max_segment_length=5.0 execution.num_workers=1
```

### Combined Processors

```bash
# Validation + Traffic Lights
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/val_and_tl processing.processors=[validation,traffic_lights] processing.processor_configs.validation.strict=false execution.num_workers=1

# All 3 processors (validation + traffic_lights + polyline_interpolation)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/fully_processed processing.processors=[validation,traffic_lights,polyline_interpolation] processing.processor_configs.validation.strict=true processing.processor_configs.polyline_interpolation.max_segment_length=2.0 execution.num_workers=1

# Traffic Lights + Polyline Interpolation only
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/tl_and_interp processing.processors=[traffic_lights,polyline_interpolation] processing.processor_configs.polyline_interpolation.max_segment_length=2.0 execution.num_workers=2
```

---

## Stage 3: Format (Unified → Target Format)

### Waymax Format (for Waymax/V-Max)

```bash
# Basic Waymax conversion
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/tfexample formatting.target_format=waymax execution.num_workers=1

# Waymax with sharding (2 shards)
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/tfexample_sharded formatting.target_format=waymax output.num_shards=2 execution.num_workers=4

# Waymax with 10 shards (production-like)
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/tfexample_10shards formatting.target_format=waymax output.num_shards=10 execution.num_workers=8
```

### GPUDrive Format (JSON)

```bash
# Basic GPUDrive conversion
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/json formatting.target_format=gpudrive execution.num_workers=1

# GPUDrive with multi-worker
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/json_batch formatting.target_format=gpudrive execution.num_workers=4

# GPUDrive from nuPlan
scenariomax command=format input_path=/data/debug/nuplan/unified output.dst=/data/debug/nuplan/json formatting.target_format=gpudrive execution.num_workers=2
```

### PufferDrive Format (Binary)

```bash
# Basic PufferDrive conversion (outputs .bin files)
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/puffer formatting.target_format=pufferdrive execution.num_workers=1

# PufferDrive with multi-worker
scenariomax command=format input_path=/data/debug/womd/unified output.dst=/data/debug/womd/puffer_batch formatting.target_format=pufferdrive execution.num_workers=4

# PufferDrive from nuPlan
scenariomax command=format input_path=/data/debug/nuplan/unified output.dst=/data/debug/nuplan/puffer formatting.target_format=pufferdrive execution.num_workers=2

# PufferDrive from multi-dataset
scenariomax command=format input_path=/data/debug/multi/unified output.dst=/data/debug/multi/puffer formatting.target_format=pufferdrive execution.num_workers=4
```

---

## Visualization

### PNG Images (Single Frame)

```bash
# Generate PNG images (first 5 scenarios)
scenariomax command=viz input_path=/data/debug/womd/unified output.dst=/data/debug/womd/viz_png visualization.output_format=png visualization.max_scenarios=5

# PNG with scatter map view
scenariomax command=viz input_path=/data/debug/womd/unified output.dst=/data/debug/womd/viz_scatter visualization.output_format=png visualization.scatter_map=true visualization.max_scenarios=5

# PNG from nuPlan
scenariomax command=viz input_path=/data/debug/nuplan/unified output.dst=/data/debug/nuplan/viz_png visualization.output_format=png visualization.max_scenarios=10
```

### Video (Animated)

```bash
# Generate videos (first 2 scenarios, 10 FPS)
scenariomax command=viz input_path=/data/debug/womd/unified output.dst=/data/debug/womd/viz_video visualization.output_format=video visualization.fps=10 visualization.max_scenarios=2

# Video with scatter map
scenariomax command=viz input_path=/data/debug/womd/unified output.dst=/data/debug/womd/viz_video_scatter visualization.output_format=video visualization.fps=10 visualization.scatter_map=true visualization.max_scenarios=2

# Video from nuPlan (20 FPS, faster playback)
scenariomax command=viz input_path=/data/debug/nuplan/unified output.dst=/data/debug/nuplan/viz_video visualization.output_format=video visualization.fps=20 visualization.max_scenarios=3
```

---

## Full Pipeline (All 3 Stages)

**Note:** Pipeline runs all stages in memory (file-by-file streaming, no intermediate saves).

### Basic Pipeline Tests

```bash
# WOMD → Waymax (simplest test)
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_tfexample formatting.target_format=waymax execution.num_workers=1 dataset_options.num_files=2

# WOMD → GPUDrive
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_json formatting.target_format=gpudrive execution.num_workers=1 dataset_options.num_files=2

# WOMD → PufferDrive
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_puffer formatting.target_format=pufferdrive execution.num_workers=1 dataset_options.num_files=2

# nuPlan → Waymax
scenariomax command=pipeline datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/nuplan/pipeline_tfexample formatting.target_format=waymax execution.num_workers=2 dataset_options.num_files=5

# nuPlan → Puffer
scenariomax command=pipeline datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/nuplan/pipeline_puffer formatting.target_format=pufferdrive execution.num_workers=2 dataset_options.num_files=5
```

### Pipeline with Processors

```bash
# WOMD → Waymax with validation
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_validated formatting.target_format=waymax processing.processors=[validation] processing.processor_configs.validation.strict=false execution.num_workers=1 dataset_options.num_files=2

# WOMD → PufferDrive with traffic lights
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_puffer_tl formatting.target_format=pufferdrive processing.processors=[traffic_lights] execution.num_workers=1 dataset_options.num_files=2

# WOMD → Waymax with polyline interpolation
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_interpolated formatting.target_format=waymax processing.processors=[polyline_interpolation] processing.processor_configs.polyline_interpolation.max_segment_length=2.0 execution.num_workers=1 dataset_options.num_files=2

# WOMD → Waymax with ALL processors
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_full formatting.target_format=waymax processing.processors=[validation,traffic_lights,polyline_interpolation] processing.processor_configs.validation.strict=true processing.processor_configs.polyline_interpolation.max_segment_length=2.0 execution.num_workers=1 dataset_options.num_files=2

# nuPlan → Puffer with all processors
scenariomax command=pipeline datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/nuplan/pipeline_puffer_full formatting.target_format=pufferdrive processing.processors=[validation,traffic_lights,polyline_interpolation] processing.processor_configs.validation.strict=false execution.num_workers=2 dataset_options.num_files=5
```

### Multi-Dataset Pipeline

```bash
# Waymo + nuPlan → TFExample (combined)
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/multi/pipeline_tfexample formatting.target_format=waymax execution.num_workers=4 dataset_options.num_files=10

# Waymo + nuPlan → Puffer with validation
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/multi/pipeline_puffer formatting.target_format=pufferdrive processing.processors=[validation] execution.num_workers=4 dataset_options.num_files=10

# Waymo + nuPlan → TFExample with all processors
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/multi/pipeline_full formatting.target_format=waymax processing.processors=[validation,traffic_lights,polyline_interpolation] execution.num_workers=8 dataset_options.num_files=20
```

### Pipeline with Sharding

```bash
# WOMD → Waymax with 5 shards
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/pipeline_sharded formatting.target_format=waymax output.num_shards=5 execution.num_workers=4 dataset_options.num_files=20

# Multi-dataset → TFExample with 10 shards (production-like)
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/multi/pipeline_production formatting.target_format=waymax output.num_shards=10 execution.num_workers=16 dataset_options.num_files=100
```

---

## Performance Testing

### Worker Scaling Tests

```bash
# 1 worker
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/1worker formatting.target_format=waymax execution.num_workers=1 dataset_options.num_files=10

# 4 workers
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/4workers formatting.target_format=waymax execution.num_workers=4 dataset_options.num_files=10

# 8 workers
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/8workers formatting.target_format=waymax execution.num_workers=8 dataset_options.num_files=10

# 16 workers
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/16workers formatting.target_format=waymax execution.num_workers=16 dataset_options.num_files=10
```

### Batch Size Tests

```bash
# Default batch size (10)
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/batch10 formatting.target_format=waymax execution.num_workers=4 execution.batch_size=10 dataset_options.num_files=20

# Small batch size (5)
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/perf/batch5 formatting.target_format=waymax execution.num_workers=4 execution.batch_size=5 dataset_options.num_files=20

# Large batch size (50)
scenariomax command=pipeline datasets.nuplan=/data/datasets/nuplan/dataset/nuplan-v1.1/splits/mini/ output.dst=/data/debug/perf/batch50 formatting.target_format=waymax execution.num_workers=4 execution.batch_size=50 dataset_options.num_files=100
```

---

## Edge Cases & Special Scenarios

### Validation Edge Cases

```bash
# Strict validation with high tolerance
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/tmp/validation_tolerant processing.processors=[validation] processing.processor_configs.validation.strict=true processing.processor_configs.validation.speed_limit_tolerance=0.5 processing.processor_configs.validation.position_jump_threshold=100.0 processing.save_output=false execution.num_workers=1

# Strict validation with low tolerance (catch more errors)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/tmp/validation_strict_low processing.processors=[validation] processing.processor_configs.validation.strict=true processing.processor_configs.validation.speed_limit_tolerance=0.05 processing.processor_configs.validation.position_jump_threshold=20.0 processing.save_output=false execution.num_workers=1
```

### Empty Processor List

```bash
# Process with no processors (identity copy)
scenariomax command=process input_path=/data/debug/womd/unified output.dst=/data/debug/womd/identity_copy processing.processors=[] execution.num_workers=1
```

### Single File Processing

```bash
# Process exactly 1 file through entire pipeline
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/womd/single_file formatting.target_format=waymax dataset_options.num_files=1 execution.num_workers=1
```

---

## Verification Commands

### Check Outputs

```bash
# Count unified pickle files
find /data/debug/womd/unified -name "*.pkl" | wc -l

# Count TFExample files
find /data/debug/womd/tfexample -name "*.tfrecord" | wc -l

# Count JSON files
find /data/debug/womd/json -name "*.json" | wc -l

# Count Puffer binary files
find /data/debug/womd/puffer -name "*.bin" | wc -l

# Check file sizes
du -sh /data/debug/womd/*

# List Puffer output structure
ls -lh /data/debug/womd/puffer/
```

### Inspect Pickles

```bash
# Python REPL to inspect unified scenarios
python3 << EOF
from scenariomax.core.utils import load_pickle
scenario = load_pickle("/data/debug/womd/unified/waymo/scenario_001.pkl")
print(f"Scenario ID: {scenario['id']}")
print(f"Length: {scenario['metadata']['length']} timesteps")
print(f"Agents: {len(scenario['dynamic_agents'])}")
print(f"Map elements: {len(scenario['static_map_elements'])}")
print(f"Traffic lights: {len(scenario.get('dynamic_map_elements', {}))}")
EOF
```

### Validate TFRecords

```bash
# Check TFRecord integrity (requires TensorFlow)
python3 << EOF
import tensorflow as tf
import glob

files = glob.glob("/data/debug/womd/tfexample/*.tfrecord")
print(f"Found {len(files)} TFRecord files")

for f in files[:3]:
    count = sum(1 for _ in tf.data.TFRecordDataset(f))
    print(f"{f}: {count} examples")
EOF
```

---

## Quick Test Suite

Run these in sequence for a comprehensive test:

```bash
# 1. Stage 1: Convert
scenariomax command=convert datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/test/unified execution.num_workers=2 dataset_options.num_files=3

# 2. Stage 2: Process with all processors
scenariomax command=process input_path=/data/debug/test/unified output.dst=/data/debug/test/processed processing.processors=[validation,traffic_lights,polyline_interpolation] execution.num_workers=1

# 3. Stage 3: Format to all targets
scenariomax command=format input_path=/data/debug/test/processed output.dst=/data/debug/test/tfexample formatting.target_format=waymax execution.num_workers=1
scenariomax command=format input_path=/data/debug/test/processed output.dst=/data/debug/test/json formatting.target_format=gpudrive execution.num_workers=1
scenariomax command=format input_path=/data/debug/test/processed output.dst=/data/debug/test/puffer formatting.target_format=pufferdrive execution.num_workers=1

# 4. Visualize
scenariomax command=viz input_path=/data/debug/test/unified output.dst=/data/debug/test/viz visualization.output_format=png visualization.max_scenarios=3

# 5. Full pipeline test
scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/test/pipeline formatting.target_format=waymax processing.processors=[validation,traffic_lights,polyline_interpolation] execution.num_workers=2 dataset_options.num_files=3
```

---

## Cleanup

```bash
# Remove all debug outputs
rm -rf /data/debug/*

# Remove specific test outputs
rm -rf /data/debug/womd/
rm -rf /data/debug/nuplan/
rm -rf /data/debug/multi/
```


scenariomax command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/test/pipeline formatting.target_format=pufferdrive processing.processors=[validation,traffic_lights,polyline_interpolation, overpass_filtering] execution.num_workers=1 dataset_options.num_files=1

python -m cProfile -o program.prof main.py command=pipeline datasets.waymo=/data/datasets/womd/training/ output.dst=/data/debug/test/pipeline formatting.target_format=pufferdrive processing.processors=[validation,traffic_lights,polyline_interpolation] execution.num_workers=1 dataset_options.num_files=1


python viz_bad_routes.py /data/debug/womd/pipeline_puffer_tl --threshold 4.0 --output-dir bad_routes_viz

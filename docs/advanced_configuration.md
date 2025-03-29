# Advanced Configuration Guide

This guide covers advanced configuration options and techniques for ScenarioMax to optimize your data processing workflow.

## Command Line Options

### Output Format Selection

ScenarioMax supports multiple output formats:

```bash
# For TFRecord output (Waymax/V-Max compatible)
--target_format tfexample

# For JSON output (GPUDrive compatible)
--target_format gpudrive

# For unified pickle format (intermediate representation)
--target_format pickle
```

### Parallel Processing

Control the level of parallelism:

```bash
# Use 16 workers for parallel processing
--num_workers 16
```

For large datasets, increasing the number of workers can significantly reduce processing time, but be mindful of memory usage.

### TFRecord Sharding

**Only works with TFExample**

For large datasets, you can split the output into multiple TFRecord files:

```bash
# Split output into 1000 separate TFRecord files
--shard 1000 --tfrecord_name training
```

This is particularly useful when working with Waymax, as it enables more efficient data loading.

### Logging Configuration

Control the verbosity of logging:

```bash
# Enable detailed debug logging
--log_level DEBUG

# Log to a file instead of stdout
--log_file /path/to/conversion.log
```

### Dataset Limits

Process only a subset of data for testing:

```bash
# Process only 100 scenarios
--num_files 100
```

## Advanced Pipeline Customization

### Chaining Multiple Conversions

For complex transformations, you can chain multiple conversions:

```bash
# Step 1: Convert from raw to unified format
python scenariomax/convert_dataset.py --waymo_src /data/waymo --dst /data/intermediate --target_format pickle

# Step 2: Convert from unified to TFExample
python scenariomax/scripts/convert_scenarionet_to_tfexample.py -d waymo --input_dir /data/intermediate --output_dir /data/final
```

### Multi-Dataset Combination

**Multi-Dataset has only been tested for the TFExample format**

ScenarioMax allows you to combine multiple datasets from different sources into a unified dataset. This is particularly useful for training models on more diverse driving data
and cross-domain validation sets.

When combining datasets, ScenarioMax:
- Preserves dataset-specific identifiers in the output
- Automatically shuffles data
- Applies the same preprocessing to all sources

Example of combining Waymo and nuPlan datasets:

```bash
python scenariomax/convert_dataset.py \
  --waymo_src /path/to/waymo/data \
  --nuplan_src /path/to/nuplan/data \
  --dst /path/to/output/directory \
  --target_format tfexample \
  --num_workers 8 \
  --shard 100 \
```

You can add as many source datasets as needed by including their corresponding source flags.


## Advanced Usage Examples

## Utility Scripts

ScenarioMax provides several utility scripts for specific conversion tasks and debugging. Here's an overview of the key scripts:

### Debug Script

The debug script allows you to visualize converted data as PNG images to validate your conversion pipeline:

```bash
python scenariomax/scripts/debug.py \
  --src /path/to/input/pickle/files \
  --dst /path/to/output/directory \
  --dataset dataset_name \
  --num-files 10
```

This script:
- Takes pickle files as input and converts them to TFRecord format
- Generates PNG visualizations of each scenario's data representation
- Saves visualizations to a debug directory for inspection
- Helps validate that the conversion process maintains spatial relationships and essential data

The visualizations are particularly useful for ensuring that the converted data accurately represents the original scenario geometry, agent positions, and trajectories.

### Convert to TFExample Script

For batch conversion of pickle files to TensorFlow TFRecord format:

```bash
python scenariomax/scripts/convert_scenarionet_to_tfexample.py \
  -d nuplan \
  --num_files 1000 \
  --n_jobs 16
```

This script:
- Converts pickle files to TFRecord format compatible with TensorFlow pipelines (like Waymax/V-Max)
- Processes files in parallel for efficiency
- Automatically merges individual TFRecord files into a single training file
- Supports batch processing of large datasets

### Convert to GPUDrive Script

For converting datasets to GPUDrive JSON format:

```bash
python scenariomax/scripts/convert_scenarionet_to_gpudrive.py \
  --input_dir /path/to/pickle/files \
  --dataset nuplan \
  --split train \
  --num_files 1000 \
  --n_jobs 16 \
  --output_dir /path/to/output
```

This script:
- Converts pickle files to GPUDrive-compatible JSON format
- Organizes output files by dataset and split
- Processes files in parallel
- Creates individual JSON files for each scenario
- Maintains scenario IDs and structure needed for GPUDrive

Each conversion script can be customized with command-line arguments to control processing parameters, input/output locations, and parallelism.


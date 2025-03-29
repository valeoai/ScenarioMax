# Adding Support for a New Dataset

This tutorial explains how to extend ScenarioMax to support additional autonomous driving datasets.

## Overview

Adding a new dataset requires implementing:

1. A dataset loader to read the raw data
2. A converter to transform the raw data into the unified format
3. (Optional) Dataset-specific pre/post-processing functions

## Step 1: Create Dataset Converter Module

Create a new Python module in the `scenariomax/raw_to_unified/converter` directory:

```python
# scenariomax/raw_to_unified/converter/my_dataset.py

def get_my_dataset_scenarios(dataset_path, **kwargs):
    """
    Load scenarios from your dataset format.

    Args:
        dataset_path: Path to the dataset
        **kwargs: Additional arguments

    Returns:
        List of scenario objects in the dataset's native format
    """
    # Implementation details will depend on your dataset's format
    scenarios = []

    # Load and parse dataset files
    # ...

    return scenarios

def preprocess_my_dataset_scenarios(scenarios):
    """
    Preprocess scenarios before conversion (optional).

    Args:
        scenarios: List of scenarios in native format

    Returns:
        Preprocessed scenarios
    """
    # Apply any necessary preprocessing
    # ...

    return processed_scenarios

def convert_my_dataset_scenario(scenario, **kwargs):
    """
    Convert a single scenario to unified format.

    Args:
        scenario: A scenario in the dataset's native format
        **kwargs: Additional arguments

    Returns:
        Dictionary containing the scenario in unified format
    """
    unified_scenario = {
        "metadata": {
            "dataset_name": "my_dataset",
            "dataset_version": "v1.0",
            "scenario_id": extract_scenario_id(scenario),
            # Other metadata fields
        },
        "dynamic_map_states": convert_map_states(scenario),
        "timestamps": extract_timestamps(scenario),
        "tracks": convert_tracks(scenario),
        # Other required fields
    }

    return unified_scenario
```

## Step 2: Implement Conversion Functions

Implement the necessary helper functions to convert dataset-specific elements:

```python
def extract_scenario_id(scenario):
    # Extract a unique identifier for the scenario
    pass

def convert_map_states(scenario):
    # Convert map information to unified format
    pass

def extract_timestamps(scenario):
    # Extract timestamp information
    pass

def convert_tracks(scenario):
    # Convert vehicle/agent tracks to unified format
    pass
```

## Step 3: Update the Main Conversion Script

Modify `scenariomax/convert_dataset.py` to include your new dataset:

1. Add a new command line argument:

```python
parser.add_argument(
    "--my_dataset_src",
    type=str,
    default=None,
    help="The directory storing the raw My Dataset data",
)
```

2. Add your dataset to the processing logic:

```python
if args.my_dataset_src is not None:
    datasets_to_process.append("my_dataset")
```

3. Add your dataset to the conversion function:

```python
elif dataset == "my_dataset":
    from scenariomax.raw_to_unified.converter import my_dataset

    logger.info(f"Loading My Dataset scenarios from {args.my_dataset_src}")
    scenarios = my_dataset.get_my_dataset_scenarios(args.my_dataset_src, num=args.num_files)

    preprocess_func = my_dataset.preprocess_my_dataset_scenarios
    convert_func = my_dataset.convert_my_dataset_scenario

    dataset_name = "my_dataset"
    dataset_version = "v1.0"
    additional_args = {}
```

## Step 4: Test Your Implementation

Test your implementation with:

```bash
python scenariomax/convert_dataset.py \
  --my_dataset_src /path/to/my/dataset \
  --dst /path/to/output \
  --target_format pickle
```

## Unified Format Requirements

Your converter must produce a dictionary with these key elements:

- `metadata`: Dataset and scenario information
- `dynamic_map_states`: Map elements like lanes, road boundaries
- `timestamps`: Scenario timesteps
- `tracks`: Agent states at each timestep

Refer to existing dataset converters (waymo.py, nuplan.py) for detailed examples of the expected structure.

## Advanced Features

For more complex datasets, you may need to implement:

- Custom preprocessing for handling dataset-specific quirks
- Specialized map conversion for different map formats
- Custom agent filtering logic

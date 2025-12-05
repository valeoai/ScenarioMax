# ScenarioMax Installation Guide

Comprehensive installation guide for ScenarioMax with dataset-specific setup instructions.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Dataset-Specific Setup](#dataset-specific-setup)
- [Installation Methods](#installation-methods)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.10.x (strict requirement)
- **Memory**: 16 GB RAM (32 GB recommended for large datasets)
- **Storage**: 100+ GB free space (depends on dataset size)
- **OS**: Linux (Ubuntu 20.04+), macOS (limited support)

### Software Dependencies

- [uv](https://docs.astral.sh/uv/) - Fast Python package manager (required)
- Git 2.20+ for cloning repositories

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify installation
uv --version
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/valeoai/ScenarioMax.git
cd ScenarioMax
```

### 2. Install with Dataset Support

**Recommended method** - uses modern uv workflow:

```bash
# Waymo Open Motion Dataset
make waymo

# nuPlan dataset
make nuplan

# All compatible datasets (Waymo + nuPlan)
make all

# Development environment (includes all tools)
make dev
```

**What `make` does**:
- Automatically creates `.venv` directory with Python 3.10
- Installs ScenarioMax and dependencies
- Configures dataset-specific requirements
- No manual venv activation needed!

### 3. Verify Installation

```bash
make status
```

You should see:
```
✅ uv.lock file exists
✅ Virtual environment exists
   ScenarioMax version: 0.1.0
✅ waymo (or nuplan, etc.)
```

---

## Dataset-Specific Setup

### Waymo Open Motion Dataset

#### Installation

```bash
make waymo
```

#### Dataset Download

Download Waymo Open Motion Dataset from: https://waymo.com/open/download/

```bash
# Example directory structure
/data/waymo/
├── training/
│   ├── training.tfrecord-00000-of-01000
│   ├── training.tfrecord-00001-of-01000
│   └── ...
└── validation/
    └── ...
```

#### Usage

```bash
source .venv/bin/activate
scenariomax command=convert \
    datasets.waymo.path=/data/waymo/training \
    output_dir=/output/waymo
```

---

### nuPlan Dataset

#### Installation

```bash
make nuplan
```

#### Prerequisites

1. **Request access** to nuPlan dataset: https://www.nuscenes.org/nuplan
2. **Download dataset** following nuPlan instructions
3. **Download map data** (required)

#### Environment Variables

Set these in your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"
export NUPLAN_EXP_ROOT="/path/to/experiments"  # Optional
```

#### Directory Structure

```bash
$NUPLAN_DATA_ROOT/
├── nuplan-v1.1/
│   ├── splits/
│   ├── sensor_blobs/
│   └── ...

$NUPLAN_MAPS_ROOT/
├── nuplan-maps-v1.0/
│   ├── sg-one-north/
│   ├── us-ma-boston/
│   ├── us-nv-las-vegas-strip/
│   └── us-pa-pittsburgh-hazelwood/
```

#### Usage

```bash
source .venv/bin/activate
scenariomax command=convert \
    datasets.nuplan.path=$NUPLAN_DATA_ROOT/nuplan-v1.1 \
    output_dir=/output/nuplan
```

---

### nuScenes Dataset (Limited Support)

⚠️ **Warning**: nuScenes conflicts with nuPlan due to incompatible Shapely versions.

#### Installation (Separate Environment)

```bash
# Option 1: New directory
cd ..
git clone https://github.com/valeoai/ScenarioMax.git ScenarioMax-nuscenes
cd ScenarioMax-nuscenes
make nuscenes

# Option 2: Clean existing install
make clean
make nuscenes  # Will show warning
```

#### Why Separate Environment?

- nuPlan requires `Shapely>=2.0.0`
- nuScenes requires `Shapely<2.0.0`
- Cannot install both simultaneously

#### Dataset Download

Download from: https://www.nuscenes.org/nuscenes

```bash
/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/
└── v1.0-trainval/
```

---

## Installation Methods

### Method 1: Makefile (Recommended)

**Best for**: Quick setup, standard configurations

```bash
make waymo    # or nuplan, all, dev
make status   # Check installation
make lock     # Generate lockfile
make clean    # Clean up
```

### Method 2: Direct uv Commands

**Best for**: Custom workflows, CI/CD pipelines

```bash
# Install with extras
uv sync --extra waymo
uv sync --extra nuplan
uv sync --extra all

# With lockfile
uv lock                    # Generate uv.lock
uv sync --frozen          # Install from lock (reproducible)

# Activate and use
source .venv/bin/activate
scenariomax --help
```

### Method 3: Development Mode

**Best for**: Contributing to ScenarioMax

```bash
make dev

# Installs additional tools:
# - pytest, pytest-cov (testing)
# - ruff (linting)
# - black, isort (formatting)
# - mypy (type checking)
```

---

## Troubleshooting

### Problem: "uv: command not found"

**Solution**: Install uv package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal or source profile
source ~/.bashrc  # or ~/.zshrc
```

### Problem: Python 3.10 not found

**Solution**: Install Python 3.10

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# macOS (using pyenv)
brew install pyenv
pyenv install 3.10.13
pyenv global 3.10.13
```

### Problem: "Shapely version conflict"

**Symptoms**: Error when installing `[all]` or mixing nuPlan + nuScenes

**Solution**:
- For nuPlan: Use `make nuplan` (Shapely>=2.0.0)
- For nuScenes: Use `make nuscenes` in separate environment
- Do NOT install both together

### Problem: nuPlan environment variables not set

**Symptoms**: `KeyError: 'NUPLAN_DATA_ROOT'` or similar

**Solution**: Set environment variables

```bash
# Temporary (current session)
export NUPLAN_DATA_ROOT="/path/to/nuplan"
export NUPLAN_MAPS_ROOT="/path/to/maps"

# Permanent (add to ~/.bashrc)
echo 'export NUPLAN_DATA_ROOT="/path/to/nuplan"' >> ~/.bashrc
echo 'export NUPLAN_MAPS_ROOT="/path/to/maps"' >> ~/.bashrc
source ~/.bashrc
```

### Problem: TensorFlow not finding CUDA

**Symptoms**: TensorFlow warnings about CUDA/GPU

**Solution**: TensorFlow 2.11.1 used for CPU-only (TFRecord I/O). GPU not required.

If you need GPU support:
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled TensorFlow (modify requirements)
# Note: This is beyond standard ScenarioMax support
```

### Problem: "Failed to build nuplan-devkit"

**Symptoms**: Installation fails during nuplan-devkit compilation

**Solution**: Install system dependencies

```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install
```

### Problem: Out of memory during conversion

**Symptoms**: Process killed during large dataset conversion

**Solution**: Reduce parallel workers

```bash
scenariomax command=convert \
    datasets.waymo.path=/data/waymo \
    execution.num_workers=4  # Reduce from default 8
```

---

## Advanced Configuration

### Using uv.lock for Reproducible Builds

Generate lockfile:
```bash
make lock  # Creates uv.lock
```

Install from lock:
```bash
uv sync --frozen --extra waymo
```

### Custom Installation Location

```bash
# Create venv in custom location
uv venv --python 3.10 /path/to/custom/venv

# Install to custom venv
uv pip install --python /path/to/custom/venv/bin/python -e ".[waymo]"
```

### Offline Installation

1. Generate lockfile on online machine:
```bash
uv lock
```

2. Transfer repository + lockfile to offline machine

3. Install with frozen dependencies:
```bash
uv sync --frozen --extra waymo
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: |
    uv sync --frozen --extra dev

- name: Run tests
  run: |
    source .venv/bin/activate
    pytest tests/
```

---

## Verifying Dataset Installation

### Check Waymo

```bash
source .venv/bin/activate
python -c "from waymo_open_dataset import dataset_pb2; print('✅ Waymo OK')"
```

### Check nuPlan

```bash
source .venv/bin/activate
python -c "import nuplan; print('✅ nuPlan OK')"
python -c "import os; print(f'Data: {os.getenv(\"NUPLAN_DATA_ROOT\")}')"
```

### Check ScenarioMax

```bash
source .venv/bin/activate
scenariomax --help
python -c "import scenariomax; print('✅ ScenarioMax OK')"
```

---

## Getting Help

- **Documentation**: See [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
- **Issues**: https://github.com/valeoai/ScenarioMax/issues
- **Dataset docs**:
  - Waymo: https://waymo.com/open/data/motion/
  - nuPlan: https://www.nuscenes.org/nuplan
  - nuScenes: https://www.nuscenes.org/nuscenes

---

## Summary of Commands

```bash
# Installation
make waymo          # Waymo dataset
make nuplan         # nuPlan dataset
make nuscenes       # nuScenes (conflicts with nuPlan)
make all            # Waymo + nuPlan
make dev            # Development tools

# Utilities
make status         # Check installation
make lock           # Generate uv.lock
make clean          # Remove venv

# Direct uv commands
uv sync --extra waymo     # Install Waymo support
uv sync --frozen          # Install from lockfile
uv lock                   # Generate lockfile
```

---

**Last Updated**: 2025-11-20
**ScenarioMax Version**: 0.1.0

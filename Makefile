# ScenarioMax Makefile for uv-based installation and management

.PHONY: help setup womd nuplan nuscenes all dev clean

# Default target
help:
	@echo "ScenarioMax - uv-based installation and management"
	@echo "================================================="
	@echo ""
	@echo "Quick installation targets:"
	@echo "  make womd          Install with Waymo dataset support"
	@echo "  make nuplan        Install with nuPlan dataset support"
	@echo "  make nuscenes      Install with nuScenes dataset support"
	@echo "  make all           Install every datasets"
	@echo "  make dev           Install development environment"
	@echo ""
	@echo "Utility commands:"
	@echo "  make clean         Remove virtual environment"
	@echo "  make lock          Generate lock file"



# Create virtual environment
.venv/pyvenv.cfg:
	uv venv --python 3.10


womd: .venv/pyvenv.cfg
	uv pip install -e ".[womd]"

nuplan: .venv/pyvenv.cfg
	uv pip install -e ".[nuplan]"

nuscenes: .venv/pyvenv.cfg
	uv pip install -e ".[nuscenes]"

all: .venv/pyvenv.cfg
	uv pip install -e ".[all]"

dev: .venv/pyvenv.cfg
	uv pip install -e ".[dev]"


# Utility commands
clean:
	rm -rf .venv
	rm -rf *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

lock: .venv/pyvenv.cfg
	uv pip compile pyproject.toml -o requirements.lock

# Show current installation
status:
	@echo "Current installation status:"
	@echo "=========================="
	@if [ -d .venv ]; then \
		echo "✅ Virtual environment exists"; \
		.venv/bin/python -c "import scenariomax; print(f'ScenarioMax version: {scenariomax.__version__ if hasattr(scenariomax, \"__version__\") else \"dev\"}')" 2>/dev/null || echo "❌ ScenarioMax not installed"; \
	else \
		echo "❌ No virtual environment found"; \
	fi

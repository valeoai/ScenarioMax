# ScenarioMax Makefile - Modern uv workflow
#
# This Makefile uses uv's native sync command which:
# - Automatically creates/updates the virtual environment
# - Installs dependencies with proper resolution
# - Uses uv.lock for reproducible builds

.PHONY: help waymo nuplan nuscenes all dev clean lock status

# Default target
help:
	@echo "ScenarioMax - Modern uv Installation"
	@echo "====================================="
	@echo ""
	@echo "Quick installation (auto-creates venv):"
	@echo "  make waymo          Install with Waymo dataset support"
	@echo "  make nuplan         Install with nuPlan dataset support"
	@echo "  make all            Install all datasets (Waymo + nuPlan)"
	@echo "  make dev            Install development environment"
	@echo ""
	@echo "Utility commands:"
	@echo "  make lock           Generate/update uv.lock file"
	@echo "  make status         Show installation status"
	@echo "  make clean          Remove virtual environment and artifacts"
	@echo ""
	@echo "Note: Commands use 'uv sync' which handles venv creation automatically"

# Dataset installations using uv sync
waymo:
	uv sync --extra waymo

nuplan:
	uv sync --extra nuplan

all:
	uv sync --extra all

dev:
	uv sync --extra dev

# Lock file generation using uv's native locking
lock:
	uv lock

# Status check
status:
	@echo "Current installation status:"
	@echo "============================"
	@if [ -f uv.lock ]; then \
		echo "✅ uv.lock file exists"; \
	else \
		echo "⚠️  No uv.lock file (run 'make lock')"; \
	fi
	@if [ -d .venv ]; then \
		echo "✅ Virtual environment exists"; \
		.venv/bin/python -c "import scenariomax; print(f'   ScenarioMax version: {scenariomax.__version__ if hasattr(scenariomax, \"__version__\") else \"dev\"}')" 2>/dev/null || echo "❌ ScenarioMax not installed in venv"; \
	else \
		echo "❌ No virtual environment found"; \
	fi
	@echo ""
	@echo "Installed extras:"
	@if [ -d .venv ]; then \
		.venv/bin/python -c "try:\n    import waymo_open_dataset; print('  ✅ waymo')\nexcept: print('  ❌ waymo')" 2>/dev/null || true; \
		.venv/bin/python -c "try:\n    import nuplan; print('  ✅ nuplan')\nexcept: print('  ❌ nuplan')" 2>/dev/null || true; \
	fi

# Clean up
clean:
	rm -rf .venv
	rm -rf *.egg-info
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned up virtual environment and artifacts"

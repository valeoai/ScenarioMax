"""ScenarioMax: A toolkit for scenario-based autonomous vehicle testing."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("scenariomax")
    except PackageNotFoundError:
        __version__ = "dev"
except ImportError:
    __version__ = "dev"


__all__ = ["__version__"]

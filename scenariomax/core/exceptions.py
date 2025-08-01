"""
ScenarioMax custom exception hierarchy for better error handling and debugging.

This module defines a comprehensive exception hierarchy that covers common error
scenarios in the dataset conversion pipeline, making error handling more specific
and debugging more straightforward.
"""


class ScenarioMaxError(Exception):
    """Base exception class for all ScenarioMax-related errors."""

    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# ═══════════════════════════════════════════════════════
# Dataset-specific Errors
# ═══════════════════════════════════════════════════════


class DatasetError(ScenarioMaxError):
    """Base class for dataset-related errors."""


class UnsupportedDatasetError(DatasetError):
    """Raised when an unsupported dataset is requested."""

    def __init__(self, dataset_name: str, supported_datasets: list[str] = None):
        supported = ", ".join(supported_datasets) if supported_datasets else "check dataset_registry.py"
        message = f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}"
        super().__init__(message, {"dataset_name": dataset_name, "supported_datasets": supported_datasets})


class DatasetLoadError(DatasetError):
    """Raised when dataset loading fails."""

    def __init__(self, dataset_name: str, dataset_path: str, reason: str = None):
        message = f"Failed to load dataset '{dataset_name}' from '{dataset_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"dataset_name": dataset_name, "dataset_path": dataset_path, "reason": reason})


class EmptyDatasetError(DatasetError):
    """Raised when a dataset contains no scenarios."""

    def __init__(self, dataset_name: str, dataset_path: str):
        message = f"Dataset '{dataset_name}' at '{dataset_path}' contains no scenarios"
        super().__init__(message, {"dataset_name": dataset_name, "dataset_path": dataset_path})


# ═══════════════════════════════════════════════════════
# Conversion Pipeline Errors
# ═══════════════════════════════════════════════════════


class ConversionError(ScenarioMaxError):
    """Base class for conversion pipeline errors."""


class UnsupportedFormatError(ConversionError):
    """Raised when an unsupported format is requested."""

    def __init__(self, format_name: str, supported_formats: list[str] = None):
        supported = ", ".join(supported_formats) if supported_formats else "pickle, tfexample, gpudrive"
        message = f"Unsupported format: {format_name}. Supported formats: {supported}"
        super().__init__(message, {"format_name": format_name, "supported_formats": supported_formats})


class ScenarioConversionError(ConversionError):
    """Raised when individual scenario conversion fails."""

    def __init__(self, scenario_id: str, dataset_name: str, reason: str = None):
        message = f"Failed to convert scenario '{scenario_id}' from dataset '{dataset_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"scenario_id": scenario_id, "dataset_name": dataset_name, "reason": reason})


class WorkerProcessingError(ConversionError):
    """Raised when a worker fails during parallel processing."""

    def __init__(self, worker_index: int, reason: str = None):
        message = f"Worker {worker_index} failed during processing"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"worker_index": worker_index, "reason": reason})


# ═══════════════════════════════════════════════════════
# Data Validation Errors
# ═══════════════════════════════════════════════════════


class ValidationError(ScenarioMaxError):
    """Base class for data validation errors."""


class InvalidScenarioDataError(ValidationError):
    """Raised when scenario data fails validation."""

    def __init__(self, scenario_id: str, field_name: str, reason: str = None):
        message = f"Invalid data in scenario '{scenario_id}' field '{field_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"scenario_id": scenario_id, "field_name": field_name, "reason": reason})


class MissingRequiredFieldError(ValidationError):
    """Raised when required fields are missing from scenario data."""

    def __init__(self, scenario_id: str, missing_fields: list[str]):
        fields_str = ", ".join(missing_fields)
        message = f"Missing required fields in scenario '{scenario_id}': {fields_str}"
        super().__init__(message, {"scenario_id": scenario_id, "missing_fields": missing_fields})


class InvalidAgentTypeError(ValidationError):
    """Raised when an invalid agent type is encountered."""

    def __init__(self, agent_type: str, valid_types: list[str] = None):
        valid = ", ".join(valid_types) if valid_types else "check types.py"
        message = f"Invalid agent type: {agent_type}. Valid types: {valid}"
        super().__init__(message, {"agent_type": agent_type, "valid_types": valid_types})


# ═══════════════════════════════════════════════════════
# TFExample-specific Errors (maintain compatibility)
# ═══════════════════════════════════════════════════════


class OverpassException(ValidationError):
    """Exception raised when an overpass is detected in the input data."""

    def __init__(self, scenario_id: str = None, message: str = "Overpass detected in the roadgraph. Skip scenario."):
        super().__init__(message, {"scenario_id": scenario_id})


class NotEnoughValidObjectsException(ValidationError):
    """Exception raised when there are not enough valid objects in the input data."""

    def __init__(
        self,
        scenario_id: str = None,
        message: str = "Not enough valid objects in the scenario for multi-agents. Skip scenario.",
    ):
        super().__init__(message, {"scenario_id": scenario_id})


# ═══════════════════════════════════════════════════════
# Map Processing Errors
# ═══════════════════════════════════════════════════════


class MapProcessingError(ScenarioMaxError):
    """Base class for map processing errors."""


class UnsupportedGeometryError(MapProcessingError):
    """Raised when unsupported geometry types are encountered."""

    def __init__(self, geometry_type: str, supported_types: list[str] = None):
        supported = ", ".join(supported_types) if supported_types else "LineString, MultiLineString"
        message = f"Unsupported geometry type: {geometry_type}. Supported types: {supported}"
        super().__init__(message, {"geometry_type": geometry_type, "supported_types": supported_types})


class MapExtractionError(MapProcessingError):
    """Raised when map extraction fails."""

    def __init__(self, map_layer: str, reason: str = None):
        message = f"Failed to extract map layer '{map_layer}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"map_layer": map_layer, "reason": reason})


# ═══════════════════════════════════════════════════════
# Environment and Configuration Errors
# ═══════════════════════════════════════════════════════


class ConfigurationError(ScenarioMaxError):
    """Base class for configuration-related errors."""


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when required environment variables are missing."""

    def __init__(self, env_var: str, description: str = None):
        message = f"Missing required environment variable: {env_var}"
        if description:
            message += f" ({description})"
        super().__init__(message, {"env_var": env_var, "description": description})


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, config_key: str, config_value: str, reason: str = None):
        message = f"Invalid configuration for '{config_key}': {config_value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"config_key": config_key, "config_value": config_value, "reason": reason})

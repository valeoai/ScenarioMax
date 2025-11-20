import logging
import os

import absl.logging


def get_logger(name: str = "scenariomax") -> logging.Logger:
    """Get a configured logger instance.

    Uses standard Python logging with per-module loggers for better control.

    Args:
        name: The name for the logger (typically __name__ from calling module)

    Returns:
        Configured Logger instance with appropriate handlers
    """
    return logging.getLogger(name)


def setup_logger(log_level: int | None = None, log_file: str | None = None):
    """Set up the logger with proper configuration.

    Args:
        log_level: Logging level (if None, uses INFO or level from env var)
        log_file: Optional file path to write logs to
    """
    absl.logging.use_absl_handler()
    absl.logging.set_verbosity(absl.logging.ERROR)

    # Allow log level to be set via environment variable
    if log_level is None:
        log_level_env = os.getenv("SCENARIOMAX_LOG_LEVEL", "INFO")
        log_level = getattr(logging, log_level_env.upper(), logging.INFO)

    logger = get_logger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%d-%m %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

import logging
import os

import absl.logging


class Logger(logging.Logger):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, name: str, level=logging.NOTSET):
        if self.__initialized:
            return
        super().__init__(name, level)
        self.__initialized = True

    def debug(self, msg, *args, **kwargs):
        super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)


def get_logger(name: str = "scenariomax") -> Logger:
    """Get a configured logger instance.

    Args:
        name: The name for the logger

    Returns:
        Configured Logger instance
    """
    return Logger(name)


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

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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

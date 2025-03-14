from scenariomax.logger_utils import setup_logger


def setup_logging(log_file="scenariomax.raw_to_unified.log"):
    """
    Setup logging configuration to log to a file.

    Args:
        log_file: Path to the log file
    """
    return setup_logger(log_file=log_file)

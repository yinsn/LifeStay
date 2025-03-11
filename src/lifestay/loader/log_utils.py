import logging
from typing import Optional


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up and configure a logger

    Args:
        name: Logger name (typically module name)
        level: Optional logging level (defaults to INFO if None)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # We're no longer adding handlers here since the root configuration in __init__.py
    # will propagate to all child loggers

    return logger

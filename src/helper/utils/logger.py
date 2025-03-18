"""
Centralized logging configuration for the application.
Provides consistent logging format and levels across different modules.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from ..config import settings

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Define log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define different log levels based on environment
LOG_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the specified module name.

    Args:
        name: Name of the module requesting the logger

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

        # File handler for all logs
        file_handler = RotatingFileHandler(
            logs_dir / "app.log",
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

        # File handler for errors only
        error_handler = RotatingFileHandler(
            logs_dir / "error.log",
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(error_handler)

    return logger
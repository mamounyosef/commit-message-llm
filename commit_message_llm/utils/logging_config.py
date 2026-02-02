"""Logging configuration for commit message LLM training."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Logger:
    """
    A simple logger wrapper that provides consistent logging across the package.

    This class wraps Python's logging module to provide a cleaner interface
    and ensure consistent log formatting.
    """

    _loggers: dict[str, logging.Logger] = {}
    _configured: bool = False

    @classmethod
    def configure(
        cls,
        level: str | int = logging.INFO,
        log_file: Optional[str | Path] = None,
        log_to_console: bool = True,
        format_string: str = DEFAULT_FORMAT,
    ) -> None:
        """
        Configure the root logger and all future loggers.

        Args:
            level: Logging level (e.g., "INFO", "DEBUG", or logging.INFO).
            log_file: Optional path to a log file. If provided, logs will be written to this file.
            log_to_console: Whether to log to console.
            format_string: Format string for log messages.
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if requested
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.

        Args:
            name: Name for the logger (typically __name__ of the calling module).

        Returns:
            A logging.Logger instance.
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)

        # Ensure the logger is configured
        if not cls._configured:
            cls.configure()

        return cls._loggers[name]


def setup_logging(
    level: str | int = logging.INFO,
    log_file: Optional[str | Path] = None,
    log_to_console: bool = True,
    format_string: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """
    Set up logging for the application.

    This is a convenience function that configures logging and returns a logger.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG", or logging.INFO).
        log_file: Optional path to a log file.
        log_to_console: Whether to log to console.
        format_string: Format string for log messages.

    Returns:
        A logging.Logger instance for the root logger.
    """
    Logger.configure(level=level, log_file=log_file, log_to_console=log_to_console, format_string=format_string)
    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name for the logger (typically __name__ of the calling module).

    Returns:
        A logging.Logger instance.
    """
    return Logger.get_logger(name)


__all__ = [
    "setup_logging",
    "get_logger",
    "Logger",
    "DEFAULT_FORMAT",
    "DEFAULT_DATE_FORMAT",
]

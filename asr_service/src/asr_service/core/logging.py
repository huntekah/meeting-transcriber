"""
Logging configuration using Loguru for beautiful, structured, and zero-boilerplate logging.

Features:
- Environment variable support (LOG_LEVEL)
- Colorized console output
- File logging with automatic rotation and retention
- Exception catching with full context and variable inspection
- Thread-safe by default

Environment Variables:
- LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  Example: export LOG_LEVEL=DEBUG && python main.py
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from loguru import logger


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_LEVELS: tuple[LogLevel, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _normalize_level(raw_level: str, default: LogLevel) -> LogLevel:
    normalized = raw_level.upper()
    if normalized in LOG_LEVELS:
        return cast(LogLevel, normalized)
    return default


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: LogLevel = "INFO"
    console_format: str = (
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    file_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    log_dir: str = "logs"
    file_retention: str = "5 days"
    file_rotation: str = "10 MB"
    colorize: bool = True

    @classmethod
    def from_env(cls) -> "LoggingSettings":
        """
        Load logging settings from environment variables.

        Environment Variables:
            LOG_LEVEL: Logging level (default: INFO)

        Returns:
            LoggingSettings instance with values from environment
        """
        return cls(
            level=_normalize_level(os.getenv("LOG_LEVEL", "INFO"), "INFO"),
        )


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """
    Configure loguru with console and file sinks.

    Args:
        settings: LoggingSettings instance. If None, loads from environment.

    Notes:
        - Removes default loguru handler to avoid duplicates
        - Adds colorized console output
        - Adds file output with automatic rotation and retention
        - Makes loguru the default logger for the application
    """
    if settings is None:
        settings = LoggingSettings.from_env()

    # Remove default handler
    logger.remove()

    # Console sink - colorized, human-readable
    logger.add(
        sys.stdout,
        level=settings.level,
        format=settings.console_format,
        colorize=settings.colorize,
        backtrace=True,
        diagnose=True,
    )

    # File sink - detailed, with rotation and retention
    log_path = Path(settings.log_dir)
    log_path.mkdir(exist_ok=True)

    logger.add(
        log_path / "asr_service.log",
        level=settings.level,
        format=settings.file_format,
        rotation=settings.file_rotation,
        retention=settings.file_retention,
        backtrace=True,
        diagnose=True,
        compression="zip",  # Compress rotated logs
    )

    # Log initialization
    logger.info(f"Logging initialized at level: {settings.level}")
    logger.debug(f"Log directory: {log_path.resolve()}")


# Initialize logging on import
setup_logging()

# Export logger for use throughout application
__all__ = ["logger", "setup_logging", "LoggingSettings"]

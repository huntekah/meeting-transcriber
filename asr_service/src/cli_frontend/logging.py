"""
Logging configuration for CLI frontend using Loguru.

Features:
- Logs to file since Textual apps can't use stdout
- Environment variable support (LOG_LEVEL)
- Automatic rotation and cleanup of old logs
- Beautiful formatted output with context

Environment Variables:
- LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  Default: DEBUG (CLI is for debugging)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger


@dataclass
class CLILoggingSettings:
    """CLI-specific logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    log_file: Path = Path.home() / ".meeting_scribe" / "cli.log"
    file_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    file_retention: str = "7 days"
    file_rotation: str = "5 MB"

    @classmethod
    def from_env(cls) -> "CLILoggingSettings":
        """
        Load logging settings from environment variables.

        Environment Variables:
            LOG_LEVEL: Logging level (default: DEBUG for CLI)

        Returns:
            CLILoggingSettings instance
        """
        return cls(
            level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
        )


def setup_logging(settings: CLILoggingSettings | None = None) -> None:
    """
    Configure loguru for CLI frontend.

    Logs to file only (not stdout) since Textual UI uses stdout.

    Args:
        settings: CLILoggingSettings instance. If None, loads from environment.

    Notes:
        - Removes default handler to avoid duplicates
        - Adds file sink with rotation and retention
        - Creates log directory if it doesn't exist
    """
    if settings is None:
        settings = CLILoggingSettings.from_env()

    # Remove default handler
    logger.remove()

    # Create log directory
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)

    # File sink - only sink for CLI (no stdout, as Textual owns it)
    logger.add(
        str(settings.log_file),
        level=settings.level,
        format=settings.file_format,
        rotation=settings.file_rotation,
        retention=settings.file_retention,
        backtrace=True,
        diagnose=True,
        compression="zip",
    )

    # Log initialization (this will go to the file)
    logger.info(f"CLI Logging initialized at level: {settings.level}")
    logger.debug(f"Log file: {settings.log_file.resolve()}")


# Initialize logging on import
setup_logging()

# Export logger and settings for use throughout CLI frontend
__all__ = ["logger", "setup_logging", "CLILoggingSettings"]

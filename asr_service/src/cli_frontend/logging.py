"""
Logging configuration for CLI frontend.

Logs to file since Textual apps can't use stdout.
"""

import logging
from pathlib import Path

# Log file in user's home directory
LOG_FILE = Path.home() / ".asr_cli_debug.log"

def setup_logging():
    """Setup file logging for CLI debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),  # Overwrite each run
        ]
    )

    logger = logging.getLogger("cli_frontend")
    logger.setLevel(logging.DEBUG)
    logger.info(f"CLI logging initialized - writing to {LOG_FILE}")
    return logger

# Global logger instance
logger = setup_logging()

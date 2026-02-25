# Logging Configuration Guide

This project uses **Loguru** for professional, zero-boilerplate logging throughout the application.

## Quick Start

### Backend Service

```bash
# Run with default INFO level
make run

# Run with DEBUG level
export LOG_LEVEL=DEBUG && make run

# Run with WARNING level
export LOG_LEVEL=WARNING && make run

# Run with CRITICAL only
export LOG_LEVEL=CRITICAL && make run
```

### CLI Frontend

```bash
# Run with default DEBUG level (CLI is for debugging)
make run-cli

# Run with INFO level
export LOG_LEVEL=INFO && make run-cli
```

## Architecture

### Backend (`src/asr_service/core/logging.py`)

- **Console output**: Colorized, human-readable format
- **File output**: `logs/asr_service.log` with rotation (10 MB) and retention (5 days)
- **Environment**: `LOG_LEVEL` (default: INFO)
- **Features**: Full backtrace and variable inspection on errors

```python
from asr_service.core.logging import logger

# Usage examples
logger.info("Service started")
logger.debug("Debug information: {}", variable)
logger.warning("Warning message")
logger.error("Error occurred: {}", error)

# With exception context
logger.opt(exception=True).error("Error with full traceback")

# With context binding
logger.bind(user_id=123).info("User action")
```

### CLI Frontend (`src/cli_frontend/logging.py`)

- **File output only**: `~/.meeting_scribe/cli.log` (stdout reserved for Textual UI)
- **Rotation**: 5 MB per file, retained for 7 days
- **Environment**: `LOG_LEVEL` (default: DEBUG for CLI debugging)
- **Features**: Compressed rotated logs

```python
from cli_frontend.logging import logger

logger.info("CLI message")
logger.debug("Debug for CLI")
```

## Configuration

Both logging systems use dataclass-based configuration that loads from environment variables:

### Backend LoggingSettings

```python
from asr_service.core.logging import LoggingSettings, setup_logging

# Load from environment
settings = LoggingSettings.from_env()
setup_logging(settings)

# Or use custom settings
custom_settings = LoggingSettings(
    level="DEBUG",
    log_dir="custom_logs",
    file_rotation="20 MB",
    file_retention="10 days"
)
setup_logging(custom_settings)
```

### CLI CLILoggingSettings

```python
from cli_frontend.logging import CLILoggingSettings, setup_logging

# Load from environment
settings = CLILoggingSettings.from_env()
setup_logging(settings)
```

## Log Levels

Loguru supports standard Python logging levels:

| Level | Usage | Environment Value |
|-------|-------|-------------------|
| DEBUG | Detailed debugging info | `LOG_LEVEL=DEBUG` |
| INFO | General information | `LOG_LEVEL=INFO` (backend default) |
| WARNING | Warning messages | `LOG_LEVEL=WARNING` |
| ERROR | Error messages | `LOG_LEVEL=ERROR` |
| CRITICAL | Critical failures only | `LOG_LEVEL=CRITICAL` |

## Output Locations

### Backend
- **Console**: Colorized output to stdout
- **File**: `logs/asr_service.log` (auto-rotated)

### CLI Frontend
- **File Only**: `~/.meeting_scribe/cli.log` (Textual UI owns stdout)

## Advanced Features

### Exception Catching with Full Context

```python
@logger.catch(level="ERROR")
def risky_function():
    # If any exception occurs, it will be logged with full traceback and variable values
    return 1 / 0

# Or use opt() for explicit exception logging
try:
    do_something()
except Exception:
    logger.opt(exception=True).error("Operation failed")
```

### Context Binding

```python
# Bind context for a specific logger instance
session_logger = logger.bind(session_id="abc123", user="john")
session_logger.info("Session event")  # session_id and user will be in logs
```

### Lazy Evaluation

```python
# Expensive function only called if DEBUG level is active
logger.opt(lazy=True).debug("Data: {}", expensive_calculation)
```

## Log Rotation

Both backend and CLI logs automatically rotate:

- **Backend**: When file reaches 10 MB, rotates and compresses
- **CLI**: When file reaches 5 MB, rotates and compresses
- **Retention**: Old logs automatically cleaned up (5-7 days)

## Integration with Existing Code

Existing imports of `logger` from `asr_service.core.logging` and `cli_frontend.logging` work seamlessly:

```python
from asr_service.core.logging import logger  # Now uses loguru

logger.info("Works the same!")
```

## Benefits of Loguru

✅ **Zero boilerplate**: No handler/formatter configuration needed
✅ **Beautiful output**: Colorized with context information
✅ **Thread-safe**: Built-in thread safety, no manual lock management
✅ **Exception handling**: Full variable inspection on errors
✅ **Lazy evaluation**: Only formats strings when needed
✅ **Rotation**: Simple, built-in log rotation and compression
✅ **Context binding**: Easy contextual information tracking
✅ **Structured logging**: Works with JSON, custom formats
✅ **Modern**: Uses modern Python features (dataclasses, type hints)

## Troubleshooting

### Logs not appearing
- Check `LOG_LEVEL` environment variable
- Check `logs/` directory exists and is writable
- Check file permissions

### Too many logs
- Increase `LOG_LEVEL` (e.g., set to WARNING or ERROR)
- Adjust retention policy if disk space is an issue

### Need to see only specific module logs
- Can add additional sinks in code for module-specific logging:
  ```python
  logger.add("specific_module.log", filter=lambda record: record["name"] == "asr_service.services.session")
  ```

## References

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Python Logging Standards](https://docs.python.org/3/library/logging.html)

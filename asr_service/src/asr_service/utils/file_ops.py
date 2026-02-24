import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from asr_service.core.logging import logger


def get_project_root() -> Path:
    """
    Find the project root directory by searching for pyproject.toml.

    Searches upwards from the current file's directory until pyproject.toml is found.

    Returns:
        Path to the project root directory

    Raises:
        FileNotFoundError: If pyproject.toml is not found in any parent directory
    """
    current_dir = Path(__file__).resolve().parent

    # Check current dir and all parents
    for directory in [current_dir, *current_dir.parents]:
        if (directory / "pyproject.toml").exists():
            return directory

    raise FileNotFoundError(
        "Could not find project root (pyproject.toml not found)."
    )


@contextmanager
def create_temp_file(suffix: str = ".tmp") -> Generator:
    """
    Create a temporary file that is automatically cleaned up.

    Args:
        suffix: File suffix/extension

    Yields:
        Temporary file object
    """
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        yield temp_file
    finally:
        if temp_file:
            temp_file.close()
            temp_path = Path(temp_file.name)
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")


def validate_mime_type(mime_type: str) -> bool:
    """
    Validate if the MIME type is an acceptable audio/video format.

    Args:
        mime_type: MIME type string

    Returns:
        True if valid, False otherwise
    """
    valid_types = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/flac",
        "audio/ogg",
        "video/mp4",  # Can contain audio
        "video/quicktime",  # .mov files
    }

    return mime_type in valid_types

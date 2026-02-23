from pathlib import Path
from asr_service.utils.file_ops import create_temp_file, validate_mime_type


def test_temporary_file_creation_and_cleanup():
    """Test that temporary files are created and cleaned up properly."""
    temp_path = None

    with create_temp_file(suffix=".mp3") as temp_file:
        temp_path = Path(temp_file.name)
        assert temp_path.exists()
        assert temp_path.suffix == ".mp3"

    # After context manager exits, file should be cleaned up
    assert not temp_path.exists()


def test_file_validation_utilities():
    """Test MIME type validation for audio files."""
    assert validate_mime_type("audio/mpeg") is True
    assert validate_mime_type("audio/wav") is True
    assert validate_mime_type("audio/mp4") is True
    assert validate_mime_type("video/mp4") is True  # Can contain audio
    assert validate_mime_type("text/plain") is False
    assert validate_mime_type("application/json") is False

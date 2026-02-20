import pytest
from pathlib import Path
from asr_service.services.audio_processor import AudioProcessor


def test_audio_file_validation_valid_mp3(test_audio_file):
    """Test that valid MP3 files are correctly validated."""
    processor = AudioProcessor()

    is_valid = processor.validate_audio_file(test_audio_file)

    assert is_valid is True


def test_audio_conversion_to_16khz_mono(test_audio_file):
    """Test that audio is converted to 16kHz mono format."""
    processor = AudioProcessor()

    audio_array, sample_rate = processor.load_and_convert_audio(test_audio_file)

    assert sample_rate == 16000
    assert len(audio_array.shape) == 1  # Mono (1D array)
    assert audio_array.dtype.name.startswith("float")


def test_invalid_audio_file_handling():
    """Test that invalid audio files raise appropriate exceptions."""
    processor = AudioProcessor()
    invalid_file = Path("/tmp/nonexistent_file.mp3")

    with pytest.raises((FileNotFoundError, ValueError)):
        processor.load_and_convert_audio(invalid_file)

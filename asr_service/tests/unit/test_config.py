from pathlib import Path
from asr_service.core.config import Settings


def test_settings_loads_from_environment(monkeypatch):
    """Test that Settings correctly loads values from environment variables."""
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("SAMPLE_RATE", "48000")

    settings = Settings()

    assert settings.WHISPER_MODEL == "base"
    assert settings.DEVICE == "cuda"
    assert settings.SAMPLE_RATE == 48000


def test_settings_defaults_are_correct():
    """Test that Settings has correct default values."""
    settings = Settings()

    # Model defaults
    assert settings.WHISPER_MODEL  # Should have a default model
    assert "whisper" in settings.MLX_WHISPER_MODEL.lower()
    assert settings.DEVICE in ["cpu", "cuda", "mps", "auto"]

    # Audio settings
    assert settings.SAMPLE_RATE == 16000
    assert settings.CHUNK_SIZE > 0
    assert 0.0 <= settings.VAD_THRESHOLD <= 1.0
    assert settings.SILENCE_CHUNKS > 0
    assert settings.MIN_AUDIO_LENGTH > 0

    # Server settings
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000

    # Output directory
    assert isinstance(settings.OUTPUT_DIR, Path)


def test_get_device_returns_valid_device():
    """Test that get_device() returns a valid device string."""
    settings = Settings()

    device = settings.get_device()
    assert device in ["cpu", "cuda", "mps"]


def test_get_device_respects_explicit_setting(monkeypatch):
    """Test that get_device() returns explicit DEVICE setting when not 'auto'."""
    monkeypatch.setenv("DEVICE", "cpu")

    settings = Settings()
    assert settings.get_device() == "cpu"

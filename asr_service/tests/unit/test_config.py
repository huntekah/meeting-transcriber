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


def test_new_audio_defaults():
    """New two-tier silence constants must have the expected default values."""
    s = Settings()

    assert s.BREATH_SILENCE_CHUNKS == 15
    assert s.SEMANTIC_SILENCE_CHUNKS == 45
    assert s.MAX_UTTERANCE_SECONDS == 15.0
    assert s.MIN_VALID_AUDIO_SECONDS == 1.0


def test_known_hallucinations_is_non_empty_list():
    """KNOWN_HALLUCINATIONS must be a non-empty list of strings."""
    s = Settings()

    assert isinstance(s.KNOWN_HALLUCINATIONS, list)
    assert len(s.KNOWN_HALLUCINATIONS) > 0
    assert all(isinstance(item, str) for item in s.KNOWN_HALLUCINATIONS)


def test_silence_chunks_property_alias():
    """SILENCE_CHUNKS property must return BREATH_SILENCE_CHUNKS."""
    s = Settings()
    assert s.SILENCE_CHUNKS == s.BREATH_SILENCE_CHUNKS


def test_min_audio_length_property_alias():
    """MIN_AUDIO_LENGTH property must return MIN_VALID_AUDIO_SECONDS."""
    s = Settings()
    assert s.MIN_AUDIO_LENGTH == s.MIN_VALID_AUDIO_SECONDS


def test_semantic_silence_greater_than_breath():
    """SEMANTIC_SILENCE_CHUNKS must always be > BREATH_SILENCE_CHUNKS to make
    the two-tier logic meaningful."""
    s = Settings()
    assert s.SEMANTIC_SILENCE_CHUNKS > s.BREATH_SILENCE_CHUNKS


def test_provisional_min_less_than_valid_audio():
    """PROVISIONAL_MIN_AUDIO_SECONDS must be < MIN_VALID_AUDIO_SECONDS.
    Provisional is speculative (lower latency); final commits need more evidence."""
    s = Settings()
    assert s.PROVISIONAL_MIN_AUDIO_SECONDS < s.MIN_VALID_AUDIO_SECONDS

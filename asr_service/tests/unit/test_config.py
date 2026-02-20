import pytest
import os
from asr_service.core.config import Settings


def test_settings_loads_from_environment(monkeypatch):
    """Test that Settings correctly loads values from environment variables."""
    monkeypatch.setenv("MODEL_ID", "openai/whisper-base")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("USE_FLASH_ATTENTION", "true")

    settings = Settings()

    assert settings.MODEL_ID == "openai/whisper-base"
    assert settings.DEVICE == "cuda"
    assert settings.USE_FLASH_ATTENTION is True


def test_settings_defaults_are_correct():
    """Test that Settings has correct default values."""
    settings = Settings()

    assert settings.PROJECT_NAME == "ASR Service"
    assert "whisper" in settings.MODEL_ID.lower()
    assert settings.DEVICE in ["cpu", "cuda", "mps", "auto"]
    assert isinstance(settings.USE_FLASH_ATTENTION, bool)
    assert settings.FINAL_BEAM_SIZE >= 1
    assert settings.LIVE_CHUNK_LENGTH_S > 0
    assert settings.LIVE_BATCH_SIZE > 0

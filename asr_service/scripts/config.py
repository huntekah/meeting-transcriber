"""
Unified configuration manager for ASR test scripts.

Loads all model configuration from .env file to avoid hardcoding.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class ScriptConfig(BaseSettings):
    """Configuration for test scripts, loaded from parent .env"""

    model_config = ConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        case_sensitive=True,
        extra='ignore',  # Allow extra fields in .env that aren't defined here
    )

    # Whisper models
    WHISPER_MODEL: str = "large-v3-turbo"  # For faster-whisper
    MLX_WHISPER_MODEL: str = "mlx-community/whisper-large-v3-turbo"  # For mlx-whisper

    # Diarization
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    HF_TOKEN: str | None = None

    # Device
    DEVICE: str = "auto"

    # Performance tuning
    COLD_PATH_PARALLEL_WORKERS: int = 4  # Number of parallel transcription workers


# Global config instance
config = ScriptConfig()

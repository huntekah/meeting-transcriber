"""
Configuration settings for ASR service.

Uses Pydantic Settings to load from environment variables with .env file support.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from ..utils.file_ops import get_project_root

# Compute .env file path using project root utility
try:
    _PROJECT_ROOT = get_project_root()
    _ENV_FILE = _PROJECT_ROOT / ".env"
except FileNotFoundError:
    # Fallback if project root not found
    _ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application configuration settings loaded from environment."""

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        case_sensitive=True,
        extra="ignore",  # Allow extra fields in .env
    )

    # Whisper models
    WHISPER_MODEL: str = "large-v3-turbo"  # For faster-whisper (legacy)
    MLX_WHISPER_MODEL: str = "mlx-community/whisper-large-v3-turbo"  # For MLX

    # Diarization
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    HF_TOKEN: str | None = None

    # Device configuration
    DEVICE: str = "auto"  # auto, mps, cuda, cpu

    # Audio settings
    SAMPLE_RATE: int = 16000  # Hz
    CHUNK_SIZE: int = 512  # Samples per chunk (32ms at 16kHz)
    VAD_THRESHOLD: float = 0.5  # Speech probability threshold
    SILENCE_CHUNKS: int = 15  # ~480ms silence triggers finalization
    MIN_AUDIO_LENGTH: float = 0.5  # Minimum seconds before transcription
    SCREENCAPTURE_MAX_DURATION_SECONDS: int = 86400  # 24 hours

    # Cold path settings
    COLD_PATH_CHUNK_DURATION: int = 300  # 5 minutes in seconds
    COLD_PATH_OVERLAP: int = 5  # 5 seconds overlap between chunks
    COLD_PATH_PARALLEL: bool = True  # Run diarization + transcription in parallel
    GLOBAL_DIARIZATION_ENABLED: bool = (
        True  # Run diarization once on full audio for chunks
    )

    # Server settings
    HOST: str = (
        "127.0.0.1"  # Bind to localhost by default (set to 0.0.0.0 to expose publicly)
    )
    PORT: int = 8000

    # Output
    OUTPUT_DIR: Path = Path.home() / ".meeting_scribe" / "meetings"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Performance
    COLD_PATH_PARALLEL_WORKERS: int = 4  # Number of parallel workers

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_device(self) -> str:
        """
        Auto-detect optimal device if DEVICE="auto".

        Returns:
            Device string: "mps", "cuda", or "cpu"
        """
        if self.DEVICE != "auto":
            return self.DEVICE

        # Auto-detection
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"


# Global settings instance
settings = Settings()

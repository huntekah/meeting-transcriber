import torch
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    # Project
    PROJECT_NAME: str = "ASR Service"

    # Model configuration
    MODEL_ID: str = "openai/whisper-tiny"  # Default to tiny for testing
    WHISPER_MODEL: str = "tiny"  # Whisper model for faster-whisper (tiny, base, small, medium, large-v3, large-v3-turbo)
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"  # Pyannote diarization model
    DEVICE: str = "auto"  # Will be determined at runtime
    USE_FLASH_ATTENTION: bool = False  # Flash attention for speed (requires compatible GPU)

    # Final transcription settings (high precision)
    FINAL_BEAM_SIZE: int = 5
    FINAL_CONDITION_ON_PREV: bool = True

    # Live transcription settings (fast)
    LIVE_CHUNK_LENGTH_S: int = 30
    LIVE_BATCH_SIZE: int = 16

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # HuggingFace token for accessing gated models (e.g., Pyannote)
    HF_TOKEN: str | None = None

    def get_device(self) -> str:
        """Determine the best available device."""
        if self.DEVICE != "auto":
            return self.DEVICE

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_torch_dtype(self):
        """Get appropriate torch dtype based on device."""
        device = self.get_device()
        if device in ["cuda", "mps"]:
            return torch.float16
        return torch.float32


settings = Settings()

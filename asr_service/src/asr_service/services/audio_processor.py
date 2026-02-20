import librosa
import numpy as np
from pathlib import Path
from typing import Tuple
from asr_service.core.logging import logger


class AudioProcessor:
    """Handle audio file validation and conversion."""

    TARGET_SAMPLE_RATE = 16000

    def validate_audio_file(self, file_path: Path) -> bool:
        """
        Validate that the file is a valid audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if valid, False otherwise
        """
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return False

        valid_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov"}
        if file_path.suffix.lower() not in valid_extensions:
            logger.error(f"Invalid audio file extension: {file_path.suffix}")
            return False

        return True

    def load_and_convert_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to 16kHz mono format.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load audio and resample to 16kHz mono
            audio, sample_rate = librosa.load(
                str(file_path),
                sr=self.TARGET_SAMPLE_RATE,
                mono=True
            )

            logger.info(f"Loaded audio: {len(audio)} samples at {sample_rate}Hz")
            return audio, sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise ValueError(f"Invalid audio file: {e}")

    def convert_bytes_to_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes (Float32 PCM) to numpy array.

        Args:
            audio_bytes: Raw PCM audio bytes

        Returns:
            Numpy array of audio samples
        """
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            return audio_array
        except Exception as e:
            logger.error(f"Failed to convert audio bytes: {e}")
            raise ValueError(f"Invalid audio bytes: {e}")

    def chunk_audio(self, audio: np.ndarray, chunk_size: int) -> list:
        """
        Split audio into chunks for streaming.

        Args:
            audio: Audio array
            chunk_size: Size of each chunk in samples

        Returns:
            List of audio chunks
        """
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

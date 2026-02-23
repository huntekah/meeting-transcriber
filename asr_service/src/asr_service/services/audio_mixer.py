"""
Audio mixer service.

Mixes multiple audio sources into a single file.
Supports both mono mixing and multi-channel output.
"""

from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf

from ..core.config import settings
from ..core.logging import logger


class AudioMixer:
    """
    Mix multiple audio sources into a single file.

    Provides static methods for audio mixing operations.
    """

    @staticmethod
    def mix_to_mono(
        audio_sources: List[np.ndarray],
        output_path: Path | str,
        sample_rate: int | None = None,
    ) -> Path:
        """
        Mix N audio arrays to mono by averaging.

        All sources are padded to the same length before mixing.

        Args:
            audio_sources: List of numpy arrays (one per source)
            output_path: Path to save mixed audio
            sample_rate: Sample rate in Hz (default from settings)

        Returns:
            Path to saved file

        Raises:
            ValueError: If audio_sources is empty
        """
        if not audio_sources:
            raise ValueError("Cannot mix empty audio_sources list")

        output_path = Path(output_path)
        sample_rate = sample_rate or settings.SAMPLE_RATE

        logger.info(
            f"Mixing {len(audio_sources)} sources to mono (output: {output_path.name})"
        )

        # Pad all sources to same length
        max_length = max(len(audio) for audio in audio_sources)
        logger.debug(
            f"Max length: {max_length} samples ({max_length / sample_rate:.2f}s)"
        )

        padded = []
        for idx, audio in enumerate(audio_sources):
            if len(audio) < max_length:
                padding = np.zeros(max_length - len(audio), dtype=audio.dtype)
                audio = np.concatenate([audio, padding])
                logger.debug(
                    f"Padded source {idx}: {len(audio_sources[idx])} → {len(audio)} samples"
                )
            padded.append(audio)

        # Stack as channels and average
        multi_channel = np.stack(padded, axis=1)  # Shape: (samples, channels)
        mono = multi_channel.mean(axis=1)  # Average across channels

        # Save mono WAV file
        sf.write(output_path, mono, sample_rate)

        logger.info(
            f"Saved mono mix to {output_path} "
            f"({len(mono)} samples, {len(mono) / sample_rate:.2f}s)"
        )

        return output_path

    @staticmethod
    def save_multi_channel(
        audio_sources: List[np.ndarray],
        output_path: Path | str,
        sample_rate: int | None = None,
    ) -> Path:
        """
        Save multi-channel version for debugging.

        Each source becomes a separate channel in the output file.

        Args:
            audio_sources: List of numpy arrays (one per source)
            output_path: Path to save multi-channel audio
            sample_rate: Sample rate in Hz (default from settings)

        Returns:
            Path to saved file

        Raises:
            ValueError: If audio_sources is empty
        """
        if not audio_sources:
            raise ValueError("Cannot save empty audio_sources list")

        output_path = Path(output_path)
        sample_rate = sample_rate or settings.SAMPLE_RATE

        logger.info(
            f"Saving {len(audio_sources)} sources as multi-channel "
            f"(output: {output_path.name})"
        )

        # Pad all sources to same length
        max_length = max(len(audio) for audio in audio_sources)

        padded = []
        for audio in audio_sources:
            if len(audio) < max_length:
                padding = np.zeros(max_length - len(audio), dtype=audio.dtype)
                audio = np.concatenate([audio, padding])
            padded.append(audio)

        # Stack as channels
        multi_channel = np.stack(padded, axis=1)  # Shape: (samples, channels)

        # Save multi-channel WAV file
        sf.write(output_path, multi_channel, sample_rate)

        logger.info(
            f"Saved multi-channel to {output_path} "
            f"({multi_channel.shape[0]} samples, {multi_channel.shape[1]} channels)"
        )

        return output_path

    @staticmethod
    def get_duration(audio: np.ndarray, sample_rate: int | None = None) -> float:
        """
        Get duration of audio array in seconds.

        Args:
            audio: Audio numpy array
            sample_rate: Sample rate in Hz (default from settings)

        Returns:
            Duration in seconds
        """
        sample_rate = sample_rate or settings.SAMPLE_RATE
        return len(audio) / sample_rate

"""
Shared VAD streaming state helper for provisional updates.

Encapsulates buffering, VAD state machine, and segment finalization.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Any

import numpy as np
import torch

from ..core.config import settings
from ..core.logging import logger

# Global lock to serialize VAD model access across all sources
_VAD_MODEL_LOCK = threading.Lock()


class VADStreamingBuffer:
    """Shared streaming buffer and VAD state machine."""

    def __init__(
        self,
        *,
        source_id: int,
        sample_rate: int,
        vad_model: torch.nn.Module,
        vad_threshold: float | None = None,
        silence_chunks: int | None = None,
        min_audio_length: float | None = None,
    ):
        self.source_id = source_id
        self.sample_rate = sample_rate
        self.vad_model = vad_model
        self.vad_threshold = (
            settings.VAD_THRESHOLD if vad_threshold is None else vad_threshold
        )
        self.silence_chunks = (
            settings.SILENCE_CHUNKS if silence_chunks is None else silence_chunks
        )
        self.min_audio_length = (
            settings.MIN_AUDIO_LENGTH if min_audio_length is None else min_audio_length
        )

        self._buffer_lock = threading.Lock()
        self._growing_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_counter = 0
        self._commit_ready = False
        self._speech_start_time: float | None = None

    def append_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any] | None:
        """Append audio, run VAD, and return a finalized segment if ready."""
        with self._buffer_lock:
            self._growing_buffer.append(audio_chunk)

        try:
            audio_tensor = torch.from_numpy(audio_chunk)
            with _VAD_MODEL_LOCK:
                vad_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        except (RuntimeError, ValueError) as e:
            logger.error(
                f"VAD error for source {self.source_id}: {e}", exc_info=True
            )
            return None

        if vad_prob > self.vad_threshold:
            if not self._is_speaking and not self._commit_ready:
                logger.debug(
                    f"[Source {self.source_id}] Speech START (VAD prob: {vad_prob:.3f})"
                )
                self._speech_start_time = time.time()
                self._is_speaking = True
            if not self._commit_ready:
                self._silence_counter = 0
        else:
            if self._is_speaking and not self._commit_ready:
                self._silence_counter += 1
                if self._silence_counter >= self.silence_chunks:
                    logger.debug(
                        f"[Source {self.source_id}] Speech END (silence chunks: {self._silence_counter})"
                    )
                    self._commit_ready = True
                    segment = self._finalize_segment()
                    self._is_speaking = False
                    self._silence_counter = 0
                    return segment

        return None

    def finalize_pending(self) -> Dict[str, Any] | None:
        """Force finalize any pending audio in the buffer."""
        if not self._growing_buffer:
            return None
        self._commit_ready = True
        return self._finalize_segment()

    def _finalize_segment(self) -> Dict[str, Any] | None:
        with self._buffer_lock:
            if len(self._growing_buffer) == 0:
                self._commit_ready = False
                return None
            audio_np = np.concatenate(self._growing_buffer)
            self._growing_buffer.clear()

        min_samples = int(self.min_audio_length * self.sample_rate)
        if len(audio_np) < min_samples:
            logger.debug(
                f"Skipping short segment for source {self.source_id}: "
                f"{len(audio_np)} samples < {min_samples}"
            )
            self._commit_ready = False
            self._speech_start_time = None
            return None

        current_time = time.time()
        segment_start_time = self._speech_start_time or current_time
        self._speech_start_time = None

        return {
            "audio": audio_np,
            "timestamp": current_time,
            "start_time": segment_start_time,
            "source_id": self.source_id,
            "sample_rate": self.sample_rate,
        }

    def get_streaming_snapshot(self) -> Dict[str, Any]:
        """Return a safe snapshot of the growing buffer and state."""
        with self._buffer_lock:
            audio_np = (
                np.concatenate(self._growing_buffer)
                if self._growing_buffer
                else None
            )

        return {
            "audio": audio_np,
            "speech_start_time": self._speech_start_time,
            "is_speaking": self._is_speaking,
            "commit_ready": self._commit_ready,
            "sample_rate": self.sample_rate,
        }

    def clear_commit_ready(self) -> None:
        """Clear commit_ready after finalization is consumed."""
        self._commit_ready = False

    def get_stats(self) -> Dict[str, Any]:
        """Get current streaming buffer stats."""
        with self._buffer_lock:
            buffer_size = len(self._growing_buffer)
        return {
            "buffer_size": buffer_size,
            "is_speaking": self._is_speaking,
            "silence_counter": self._silence_counter,
            "commit_ready": self._commit_ready,
        }

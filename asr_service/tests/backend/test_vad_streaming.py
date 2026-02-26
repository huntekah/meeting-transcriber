"""
Tests for shared VAD streaming buffer.
"""

import numpy as np
import torch

from asr_service.services.vad_streaming import VADStreamingBuffer


class DummyVAD:
    """Simple VAD stub returning preset probabilities."""

    def __init__(self, probs: list[float]):
        self._probs = iter(probs)

    def __call__(self, _tensor: torch.Tensor, _sample_rate: int) -> torch.Tensor:
        return torch.tensor(next(self._probs))


def test_vad_streaming_finalizes_on_silence():
    vad = DummyVAD([0.9, 0.1])
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        silence_chunks=1,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)

    assert buffer.append_chunk(chunk) is None
    segment = buffer.append_chunk(chunk)

    assert segment is not None
    assert segment["source_id"] == 0
    assert segment["sample_rate"] == 16000
    assert len(segment["audio"]) == 1024

    snapshot = buffer.get_streaming_snapshot()
    assert snapshot["commit_ready"] is True

    buffer.clear_commit_ready()
    snapshot = buffer.get_streaming_snapshot()
    assert snapshot["commit_ready"] is False


def test_finalize_pending_clears_commit_ready_on_short_segment():
    vad = DummyVAD([0.9])
    buffer = VADStreamingBuffer(
        source_id=1,
        sample_rate=16000,
        vad_model=vad,
        silence_chunks=1,
        min_audio_length=10.0,
    )

    chunk = np.ones(512, dtype=np.float32)
    buffer.append_chunk(chunk)

    segment = buffer.finalize_pending()
    assert segment is None
    snapshot = buffer.get_streaming_snapshot()
    assert snapshot["commit_ready"] is False

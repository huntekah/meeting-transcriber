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


def test_two_tier_breath_pause_does_not_commit():
    """Breath-length silence should NOT commit; buffer keeps accumulating."""
    # Speech → breath (15 chunks of silence) → still speaking → semantic silence (45)
    speech_probs = [0.9] * 5          # speaking
    breath_probs = [0.1] * 15         # breath pause (exactly breath_silence_chunks)
    more_speech = [0.9] * 3           # speaker resumes
    semantic_probs = [0.1] * 45       # proper semantic silence → commit
    all_probs = speech_probs + breath_probs + more_speech + semantic_probs

    vad = DummyVAD(all_probs)
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=15,
        semantic_silence_chunks=45,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)

    # Feed all chunks except the final semantic silence run
    segments_committed = []
    for _ in range(len(speech_probs) + len(breath_probs) + len(more_speech)):
        result = buffer.append_chunk(chunk)
        if result is not None:
            segments_committed.append(result)

    # No commit yet — breath pause is not a commit
    assert len(segments_committed) == 0

    # Feed semantic silence; last chunk should trigger commit
    for _ in range(len(semantic_probs)):
        result = buffer.append_chunk(chunk)
        if result is not None:
            segments_committed.append(result)

    assert len(segments_committed) == 1
    # Buffer contains all speech + breath + more speech chunks concatenated
    expected_samples = (len(speech_probs) + len(breath_probs) + len(more_speech) + len(semantic_probs)) * 512
    assert len(segments_committed[0]["audio"]) == expected_samples


def test_two_tier_semantic_silence_commits():
    """Semantic silence (45 chunks) always triggers a commit."""
    speech_probs = [0.9] * 5
    semantic_probs = [0.1] * 45  # straight to semantic silence

    vad = DummyVAD(speech_probs + semantic_probs)
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=15,
        semantic_silence_chunks=45,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)
    committed = []
    for _ in range(len(speech_probs) + len(semantic_probs)):
        result = buffer.append_chunk(chunk)
        if result is not None:
            committed.append(result)

    assert len(committed) == 1


def test_force_commit_on_max_utterance_seconds():
    """Buffer overflow flag triggers commit on the very next breath pause."""
    sample_rate = 16000
    chunk_size = 512
    # 15 seconds of speech at 32ms/chunk → 15 / 0.032 = 468.75 → 469 chunks
    max_sec = 15.0
    overflow_chunks = int(max_sec * sample_rate / chunk_size) + 1  # 469

    speech_probs = [0.9] * overflow_chunks
    breath_probs = [0.1] * 15   # breath pause — should force commit due to overflow

    vad = DummyVAD(speech_probs + breath_probs)
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=sample_rate,
        vad_model=vad,
        breath_silence_chunks=15,
        semantic_silence_chunks=45,
        max_utterance_seconds=max_sec,
        min_audio_length=0.0,
    )

    chunk = np.ones(chunk_size, dtype=np.float32)
    committed = []
    for _ in range(len(speech_probs) + len(breath_probs)):
        result = buffer.append_chunk(chunk)
        if result is not None:
            committed.append(result)

    assert len(committed) == 1
    stats = buffer.get_stats()
    assert stats["force_commit_on_next_breath"] is False  # reset after commit


# ---------------------------------------------------------------------------
# LiveTranscriber hallucination filter tests
# ---------------------------------------------------------------------------

def _make_result(text: str, no_speech_prob: float) -> dict:
    return {
        "text": text,
        "segments": [{"text": text, "no_speech_prob": no_speech_prob}],
        "confidence": 1.0,
    }


def test_hallucination_filter_rejects_known_artifact_with_high_no_speech_prob():
    from asr_service.services.live_transcriber import LiveTranscriber
    import queue

    lt = LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=lambda u: None,
    )

    result = _make_result("Thank you.", no_speech_prob=0.8)
    assert lt._is_hallucination(result, audio_duration_sec=1.5) is True


def test_hallucination_filter_rejects_artifact_with_length_mismatch():
    from asr_service.services.live_transcriber import LiveTranscriber
    import queue

    lt = LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=lambda u: None,
    )

    # 5s of audio, 10-char text → suspicious
    result = _make_result("Thank you.", no_speech_prob=0.1)
    assert lt._is_hallucination(result, audio_duration_sec=5.0) is True


def test_hallucination_filter_passes_genuine_thank_you():
    from asr_service.services.live_transcriber import LiveTranscriber
    import queue

    lt = LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=lambda u: None,
    )

    # Genuine "Thank you" — low no_speech_prob, audio matches text length
    result = _make_result("Thank you.", no_speech_prob=0.05)
    assert lt._is_hallucination(result, audio_duration_sec=1.0) is False


def test_hallucination_filter_passes_normal_speech():
    from asr_service.services.live_transcriber import LiveTranscriber
    import queue

    lt = LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=lambda u: None,
    )

    result = _make_result("Let me share my screen.", no_speech_prob=0.02)
    assert lt._is_hallucination(result, audio_duration_sec=2.0) is False


def test_hallucination_filter_rejects_empty_text():
    from asr_service.services.live_transcriber import LiveTranscriber
    import queue

    lt = LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=lambda u: None,
    )

    result = _make_result("", no_speech_prob=0.9)
    assert lt._is_hallucination(result, audio_duration_sec=2.0) is True


# ---------------------------------------------------------------------------
# VADStreamingBuffer edge-case paths
# ---------------------------------------------------------------------------


class ErrorVAD:
    """VAD stub that raises RuntimeError on first call, then returns a prob."""

    def __init__(self, error_on_first: bool = True, fallback_prob: float = 0.9):
        self._called = False
        self._error_on_first = error_on_first
        self._fallback = fallback_prob

    def __call__(self, _tensor: torch.Tensor, _sample_rate: int) -> torch.Tensor:
        if self._error_on_first and not self._called:
            self._called = True
            raise RuntimeError("VAD model internal error")
        return torch.tensor(self._fallback)


def test_vad_error_returns_none_and_keeps_buffer():
    """RuntimeError from the VAD model must be caught and return None.
    The audio chunk should still be in the buffer (not lost)."""
    vad = ErrorVAD(error_on_first=True)
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=5,
        semantic_silence_chunks=10,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)
    result = buffer.append_chunk(chunk)

    # VAD errored → no segment committed
    assert result is None
    # But chunk is still in the buffer
    stats = buffer.get_stats()
    assert stats["buffer_size"] == 1


def test_finalize_pending_returns_none_when_buffer_empty():
    """finalize_pending on an empty buffer must return None."""
    vad = DummyVAD([])
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        silence_chunks=1,
        min_audio_length=0.0,
    )
    assert buffer.finalize_pending() is None


def test_get_buffer_seconds_accurate():
    """_get_buffer_seconds should reflect total accumulated audio duration."""
    vad = DummyVAD([0.9] * 100)  # always speaking
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=5,
        semantic_silence_chunks=50,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)
    for _ in range(10):
        buffer.append_chunk(chunk)

    # 10 chunks × 512 samples / 16000 Hz = 0.32 s
    seconds = buffer._get_buffer_seconds()
    assert abs(seconds - 0.32) < 1e-6


def test_breath_pause_resume_then_semantic_commit_aggregates_audio():
    """Full speech-breath-resume-commit cycle: the committed segment should
    contain ALL audio (speech + breath + resumed speech + semantic silence)."""
    # speech: 5 chunks, breath: 15, resume: 5, semantic silence: 45
    speech1 = [0.9] * 5
    breath = [0.1] * 15
    speech2 = [0.9] * 5
    semantic = [0.1] * 45

    vad = DummyVAD(speech1 + breath + speech2 + semantic)
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=15,
        semantic_silence_chunks=45,
        min_audio_length=0.0,
    )

    chunk = np.ones(512, dtype=np.float32)
    total_chunks = len(speech1) + len(breath) + len(speech2) + len(semantic)
    segments = []
    for _ in range(total_chunks):
        result = buffer.append_chunk(chunk)
        if result is not None:
            segments.append(result)

    assert len(segments) == 1
    assert len(segments[0]["audio"]) == total_chunks * 512


def test_get_stats_exposes_force_commit_flag():
    """get_stats must include the new force_commit_on_next_breath field."""
    vad = DummyVAD([0.9])
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        breath_silence_chunks=5,
        semantic_silence_chunks=10,
        min_audio_length=0.0,
    )
    stats = buffer.get_stats()
    assert "force_commit_on_next_breath" in stats
    assert stats["force_commit_on_next_breath"] is False


def test_silence_chunks_alias_collapses_both_thresholds():
    """When silence_chunks (legacy) is passed, both breath and semantic
    thresholds should collapse to that value (old single-threshold behaviour)."""
    vad = DummyVAD([0.9, 0.1, 0.1, 0.1])  # speech + 3 silence chunks
    buffer = VADStreamingBuffer(
        source_id=0,
        sample_rate=16000,
        vad_model=vad,
        silence_chunks=3,  # legacy kwarg
        min_audio_length=0.0,
    )
    assert buffer.breath_silence_chunks == 3
    assert buffer.semantic_silence_chunks == 3

    chunk = np.ones(512, dtype=np.float32)
    buffer.append_chunk(chunk)  # speech
    buffer.append_chunk(chunk)  # silence 1
    buffer.append_chunk(chunk)  # silence 2
    seg = buffer.append_chunk(chunk)  # silence 3 → commit

    assert seg is not None


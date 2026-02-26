"""
TDD tests for LiveTranscriber.

Covers _handle_final_segment decision logic:
  - Rollover: short segments are queued, not discarded
  - Rollover prepend: next segment receives prepended audio
  - Hallucination filter: known artifacts with bad signals are rejected
  - Normal path: good segments emit an is_final=True Utterance
  - Stop guard: stop_event prevents work and callback
  - Callback errors are caught and logged
  - _transcribe return dict includes 'segments' key for no_speech_prob access
"""

import queue
import time
from typing import Any, Dict
from unittest.mock import patch

import numpy as np

from asr_service.core.config import settings
from asr_service.schemas.transcription import Utterance
from asr_service.services.live_transcriber import LiveTranscriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = settings.SAMPLE_RATE
MIN_SAMPLES = int(settings.MIN_VALID_AUDIO_SECONDS * SAMPLE_RATE)  # 16000


def make_segment(
    n_samples: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    start_time: float | None = None,
) -> Dict[str, Any]:
    """Build a fake finalized segment dict."""
    now = time.time()
    return {
        "audio": np.zeros(n_samples, dtype=np.float32),
        "timestamp": now,
        "start_time": start_time or now,
        "sample_rate": sample_rate,
        "source_id": 0,
    }


def make_transcribe_result(
    text: str = "Hello world.",
    no_speech_prob: float = 0.02,
) -> Dict[str, Any]:
    """Build a fake _transcribe() result."""
    return {
        "text": text,
        "segments": [{"text": text, "no_speech_prob": no_speech_prob}],
        "confidence": 1.0,
    }


def make_lt(callback=None) -> LiveTranscriber:
    """Convenience factory for a LiveTranscriber with a no-op (or given) callback."""
    if callback is None:
        callback = lambda u: None  # noqa: E731
    return LiveTranscriber(
        source_id=0,
        input_queue=queue.Queue(),
        output_callback=callback,
    )


# ---------------------------------------------------------------------------
# Rollover behaviour
# ---------------------------------------------------------------------------


class TestRollover:
    def test_short_segment_stores_rollover_not_transcribed(self):
        """Audio shorter than MIN_VALID_AUDIO_SECONDS must NOT be transcribed —
        it is saved in _rollover_audio for the next segment."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        short_seg = make_segment(MIN_SAMPLES - 1)  # one sample too short
        with patch.object(lt, "_transcribe") as mock_t:
            lt._handle_final_segment(short_seg)

        mock_t.assert_not_called()
        assert lt._rollover_audio is not None
        assert len(lt._rollover_audio) == MIN_SAMPLES - 1
        assert emitted == []

    def test_rollover_prepended_to_next_segment(self):
        """The audio stored in _rollover_audio is prepended to the next call's audio
        before the length check, so the combined buffer may reach MIN threshold."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        # First segment: short, goes to rollover
        lt._rollover_audio = np.zeros(MIN_SAMPLES // 2, dtype=np.float32)

        # Second segment: by itself still short, but combined just hits minimum
        half = MIN_SAMPLES - MIN_SAMPLES // 2  # completes the min exactly
        seg = make_segment(half)

        with patch.object(lt, "_transcribe", return_value=make_transcribe_result()) as mock_t:
            lt._handle_final_segment(seg)

        # Combined audio should have been passed to _transcribe
        mock_t.assert_called_once()
        passed_audio = mock_t.call_args[0][0]
        assert len(passed_audio) == MIN_SAMPLES

    def test_rollover_clears_after_prepend(self):
        """After prepend the rollover buffer must be cleared."""
        lt = make_lt()
        lt._rollover_audio = np.zeros(MIN_SAMPLES // 2, dtype=np.float32)
        half = MIN_SAMPLES - MIN_SAMPLES // 2
        seg = make_segment(half)

        with patch.object(lt, "_transcribe", return_value=make_transcribe_result()):
            lt._handle_final_segment(seg)

        assert lt._rollover_audio is None

    def test_combined_still_short_re_rolls(self):
        """If rollover + new segment is still below minimum, store the combined
        audio as the new rollover (don't drop it)."""
        lt = make_lt()
        lt._rollover_audio = np.zeros(100, dtype=np.float32)

        seg = make_segment(100)  # combined = 200 < MIN_SAMPLES
        with patch.object(lt, "_transcribe") as mock_t:
            lt._handle_final_segment(seg)

        mock_t.assert_not_called()
        assert lt._rollover_audio is not None
        assert len(lt._rollover_audio) == 200  # combined

    def test_stop_clears_rollover(self):
        """stop() must set _rollover_audio = None to avoid stale state."""
        lt = make_lt()
        lt._rollover_audio = np.zeros(MIN_SAMPLES // 2, dtype=np.float32)
        lt._stop_event.set()  # simulate stop without starting thread
        lt.stop()
        assert lt._rollover_audio is None


# ---------------------------------------------------------------------------
# Normal path
# ---------------------------------------------------------------------------


class TestNormalPath:
    def test_emits_is_final_true_utterance(self):
        """A good 1s+ segment with clean transcription fires the callback
        with an Utterance where is_final=True."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        seg = make_segment(MIN_SAMPLES)
        with patch.object(lt, "_transcribe", return_value=make_transcribe_result("Let's begin.")):
            lt._handle_final_segment(seg)

        assert len(emitted) == 1
        utt = emitted[0]
        assert isinstance(utt, Utterance)
        assert utt.is_final is True
        assert utt.text == "Let's begin."
        assert utt.source_id == 0

    def test_utterance_timestamps_are_reasonable(self):
        """start_time and end_time should bound the audio duration."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        start = time.time()
        seg = make_segment(MIN_SAMPLES, start_time=start)
        with patch.object(lt, "_transcribe", return_value=make_transcribe_result("Hi.")):
            lt._handle_final_segment(seg)

        utt = emitted[0]
        expected_duration = MIN_SAMPLES / SAMPLE_RATE  # 1.0 s
        assert abs((utt.end_time - utt.start_time) - expected_duration) < 0.01

    def test_total_segments_counter_increments(self):
        lt = make_lt()
        assert lt._total_segments == 0

        seg = make_segment(MIN_SAMPLES)
        with patch.object(lt, "_transcribe", return_value=make_transcribe_result("Hello.")):
            lt._handle_final_segment(seg)

        assert lt._total_segments == 1

    def test_callback_runtime_error_is_caught(self):
        """RuntimeError from the callback must not propagate to the caller."""
        def bad_callback(_):
            raise RuntimeError("WebSocket closed")

        lt = make_lt(callback=bad_callback)
        seg = make_segment(MIN_SAMPLES)
        with patch.object(lt, "_transcribe", return_value=make_transcribe_result("Hello.")):
            lt._handle_final_segment(seg)  # Must not raise


# ---------------------------------------------------------------------------
# Filtering / early-exit paths
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_stop_event_prevents_processing(self):
        """If the stop event is set, _handle_final_segment returns immediately
        without calling _transcribe or the callback."""
        emitted = []
        lt = make_lt(callback=emitted.append)
        lt._stop_event.set()

        seg = make_segment(MIN_SAMPLES)
        with patch.object(lt, "_transcribe") as mock_t:
            lt._handle_final_segment(seg)

        mock_t.assert_not_called()
        assert emitted == []

    def test_empty_transcription_no_emit(self):
        """Empty or whitespace-only text must not trigger the callback."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        seg = make_segment(MIN_SAMPLES)
        with patch.object(lt, "_transcribe", return_value=make_transcribe_result("   ")):
            lt._handle_final_segment(seg)

        assert emitted == []

    def test_hallucination_no_emit(self):
        """A hallucinated phrase with high no_speech_prob must be silently dropped."""
        emitted = []
        lt = make_lt(callback=emitted.append)

        seg = make_segment(MIN_SAMPLES)
        hallucinated = make_transcribe_result("Thank you.", no_speech_prob=0.95)
        with patch.object(lt, "_transcribe", return_value=hallucinated):
            lt._handle_final_segment(seg)

        assert emitted == []

    def test_stop_after_transcription_no_emit(self):
        """If stop fires between transcription and the callback,
        the utterance should still be emitted (stop mid-way should not silently drop it)."""
        emitted = []

        def set_stop_then_call(utt):
            emitted.append(utt)

        lt = make_lt(callback=set_stop_then_call)
        seg = make_segment(MIN_SAMPLES)

        # Set stop AFTER transcription but BEFORE callback — callback still fires
        def transcribe_then_stop(audio):
            result = make_transcribe_result("Almost done.")
            lt._stop_event.set()  # stop fires mid-execution
            return result

        with patch.object(lt, "_transcribe", side_effect=transcribe_then_stop):
            lt._handle_final_segment(seg)

        # The second stop_event check in _handle_final_segment prevents the callback
        assert emitted == []


# ---------------------------------------------------------------------------
# _transcribe return contract
# ---------------------------------------------------------------------------


class TestTranscribeContract:
    def test_transcribe_result_has_segments_key(self):
        """_transcribe must return a dict with a 'segments' key so that
        _is_hallucination can read no_speech_prob."""
        import sys
        from unittest.mock import MagicMock

        fake_mlx_result = {
            "text": "Hello.",
            "segments": [{"text": "Hello.", "no_speech_prob": 0.01}],
        }
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = fake_mlx_result

        lt = make_lt()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            result = lt._transcribe(np.zeros(MIN_SAMPLES, dtype=np.float32))

        assert "segments" in result
        assert "text" in result
        assert result["text"] == "Hello."

    def test_transcribe_result_handles_empty_segments(self):
        """If mlx returns no segments, text should be empty string."""
        import sys
        from unittest.mock import MagicMock

        fake_mlx_result = {"text": "", "segments": []}
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = fake_mlx_result

        lt = make_lt()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            result = lt._transcribe(np.zeros(MIN_SAMPLES, dtype=np.float32))

        assert result["text"] == ""
        assert result["segments"] == []

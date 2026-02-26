"""
Live transcriber service.

Consumer thread that transcribes audio segments from VADAudioProducer.
Adapted from scripts/live_test_v2.py consumer_thread pattern.
"""

import time
import threading
import queue
import contextlib
import os
from typing import Callable, Dict, Any, Protocol
import numpy as np

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import TranscriptionError
from ..schemas.transcription import Utterance

# Global lock to serialize MLX Whisper calls across all sources
# MLX might not be fully thread-safe for concurrent transcriptions
_MLX_WHISPER_LOCK = threading.Lock()


class StreamingSource(Protocol):
    """Minimal interface for producers that support streaming snapshots."""

    sample_rate: int

    def get_streaming_snapshot(self) -> Dict[str, Any]:
        ...

    def clear_commit_ready(self) -> None:
        ...


class LiveTranscriber:
    """
    Consumer thread that transcribes audio segments.

    Reads finalized audio segments from queue and runs MLX-Whisper inference.
    Sends results via callback for real-time updates.

    No provisional transcription (too resource-intensive for N sources).
    """

    def __init__(
        self,
        source_id: int,
        input_queue: queue.Queue,
        output_callback: Callable[[Utterance], None],
        whisper_model_name: str | None = None,
        language: str = "en",
        streaming_source: "StreamingSource | None" = None,
    ):
        """
        Initialize live transcriber.

        Args:
            source_id: Source identifier
            input_queue: Queue receiving audio segments from VADAudioProducer
            output_callback: Callback to send transcribed utterances
            whisper_model_name: MLX-Whisper model name (default from settings)
            language: Language code for transcription
        """
        self.source_id = source_id
        self.input_queue = input_queue
        self.output_callback = output_callback
        self.whisper_model_name = whisper_model_name or settings.MLX_WHISPER_MODEL
        self.language = language
        self.streaming_source = streaming_source

        # Thread lifecycle
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self._total_segments = 0
        self._total_inference_time = 0.0
        self._error_count = 0
        self._last_provisional_text = ""
        self._last_provisional_time = 0.0
        self._rollover_audio: np.ndarray | None = None  # Prepended to next final segment

    def start(self):
        """Start consumer thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning(
                f"LiveTranscriber {self.source_id} already running, ignoring start"
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info(f"LiveTranscriber {self.source_id} started")

    def stop(self):
        """Gracefully stop consumer thread."""
        self._rollover_audio = None  # Always clear to avoid stale state

        if self._thread is None:
            return

        logger.info(f"Stopping LiveTranscriber {self.source_id}...")
        self._stop_event.set()

        # Wait for thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info(
            f"LiveTranscriber {self.source_id} stopped "
            f"(segments={self._total_segments}, errors={self._error_count})"
        )

    def _inference_loop(self):
        """
        Main inference loop (runs in thread).

        Reads segments from queue and transcribes them.
        """
        if self.streaming_source is None:
            self._final_only_loop()
        else:
            self._streaming_loop()

    def _final_only_loop(self):
        """Inference loop for producers without streaming state."""
        while not self._stop_event.is_set():
            try:
                segment = self.input_queue.get(timeout=0.5)
                self._handle_final_segment(segment)
            except queue.Empty:
                continue
            except TranscriptionError as e:
                self._handle_transcription_error(e)

    def _streaming_loop(self):
        """Inference loop with provisional updates while speaking."""
        min_samples = int(settings.PROVISIONAL_MIN_AUDIO_SECONDS * settings.SAMPLE_RATE)
        while not self._stop_event.is_set():
            try:
                segment = self.input_queue.get_nowait()
                self._handle_final_segment(segment)
                if self.streaming_source is not None:
                    self.streaming_source.clear_commit_ready()
                self._last_provisional_text = ""
                continue
            except queue.Empty:
                pass
            except TranscriptionError as e:
                self._handle_transcription_error(e)
                if self.streaming_source is not None:
                    self.streaming_source.clear_commit_ready()
                    self._last_provisional_text = ""
                continue

            snapshot = self.streaming_source.get_streaming_snapshot()
            if snapshot["commit_ready"] or not snapshot["is_speaking"]:
                if not snapshot["is_speaking"]:
                    self._last_provisional_text = ""
                time.sleep(0.05)
                continue

            if time.time() - self._last_provisional_time < settings.PROVISIONAL_INTERVAL:
                time.sleep(0.05)
                continue

            audio_np = snapshot["audio"]
            if audio_np is None or len(audio_np) < min_samples:
                time.sleep(0.05)
                continue

            result = self._transcribe(audio_np)
            text = result["text"]
            if not text.strip() or text == self._last_provisional_text:
                self._last_provisional_time = time.time()
                time.sleep(0.05)
                continue

            sample_rate = snapshot["sample_rate"]
            duration = len(audio_np) / sample_rate
            start_time = snapshot["speech_start_time"] or (time.time() - duration)
            end_time = start_time + duration
            utterance = Utterance(
                source_id=self.source_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
                confidence=result.get("confidence", 1.0),
                is_final=False,
                overlaps_with=[],
            )

            try:
                self.output_callback(utterance)
                self._last_provisional_text = text
                self._last_provisional_time = time.time()
            except RuntimeError as callback_error:
                logger.error(
                    f"Callback error for source {self.source_id}: {callback_error}"
                )
                time.sleep(0.05)

    def _handle_final_segment(self, segment: Dict[str, Any]) -> None:
        """Process a finalized segment from the producer."""
        if self._stop_event.is_set():
            logger.debug(
                f"LiveTranscriber {self.source_id} stopping, skipping segment"
            )
            return

        audio_np: np.ndarray = segment["audio"]
        capture_timestamp: float = segment["timestamp"]
        segment_start_time: float = segment.get("start_time", capture_timestamp)

        # Prepend any rolled-over audio from a previous short segment
        if self._rollover_audio is not None:
            audio_np = np.concatenate([self._rollover_audio, audio_np])
            self._rollover_audio = None

        min_samples = int(settings.MIN_VALID_AUDIO_SECONDS * settings.SAMPLE_RATE)
        if len(audio_np) < min_samples:
            # Too short for reliable Whisper output — roll over to next segment
            logger.debug(
                f"[Source {self.source_id}] Short segment "
                f"({len(audio_np) / settings.SAMPLE_RATE:.2f}s < "
                f"{settings.MIN_VALID_AUDIO_SECONDS}s), rolling over"
            )
            self._rollover_audio = audio_np
            return

        sample_rate = segment.get("sample_rate", settings.SAMPLE_RATE)
        audio_duration = len(audio_np) / sample_rate

        start_time = time.time()
        result = self._transcribe(audio_np)
        inference_time = time.time() - start_time

        if not result["text"].strip():
            logger.debug(f"Empty transcription for source {self.source_id}, skipping")
            return

        if self._is_hallucination(result, audio_duration):
            logger.debug(
                f"[Source {self.source_id}] Hallucination rejected: '{result['text']}'"
            )
            return

        if self._stop_event.is_set():
            logger.debug(
                f"LiveTranscriber {self.source_id} stopping, skipping callback"
            )
            return

        duration = audio_duration
        end_timestamp = segment_start_time + duration
        utterance = Utterance(
            source_id=self.source_id,
            start_time=segment_start_time,
            end_time=end_timestamp,
            text=result["text"],
            confidence=result.get("confidence", 1.0),
            is_final=True,
            overlaps_with=[],
        )

        try:
            self.output_callback(utterance)
        except RuntimeError as callback_error:
            logger.error(f"Callback error for source {self.source_id}: {callback_error}")

        self._total_segments += 1
        self._total_inference_time += inference_time

        logger.debug(
            f"Transcribed segment for source {self.source_id}: "
            f"'{result['text'][:50]}...' ({duration:.2f}s, RTF={inference_time / duration:.3f})"
        )

    def _is_hallucination(self, result: Dict[str, Any], audio_duration_sec: float) -> bool:
        """
        3-factor heuristic to catch Whisper's silence-induced hallucinations.

        Checks are only applied when the text matches a known artifact phrase,
        so genuine utterances like "Thank you" in a real conversation are not rejected.
        """
        text = result.get("text", "").strip()
        if not text:
            return True

        segments = result.get("segments", [])
        max_no_speech_prob = max(
            (seg.get("no_speech_prob", 0.0) for seg in segments), default=0.0
        )

        clean_text = text.lower()
        is_known_artifact = any(
            artifact.lower() in clean_text for artifact in settings.KNOWN_HALLUCINATIONS
        )

        if not is_known_artifact:
            return False

        # Factor 1: Whisper's own no_speech signal is high
        if max_no_speech_prob > 0.4:
            return True

        # Factor 2: Audio is much longer than the text warrants
        # (e.g., 5s of shuffling paper → "Thank you.")
        if audio_duration_sec > 2.5 and len(clean_text) < 15:
            return True

        return False

    def _handle_transcription_error(self, error: TranscriptionError) -> None:
        """Log and emit an error utterance."""
        self._error_count += 1
        logger.error(
            f"LiveTranscriber error for source {self.source_id}: {error}",
            exc_info=True,
        )

        if not self._stop_event.is_set():
            try:
                error_utterance = Utterance(
                    source_id=self.source_id,
                    start_time=time.time(),
                    end_time=time.time(),
                    text=f"[TRANSCRIPTION ERROR: {str(error)}]",
                    confidence=0.0,
                    is_final=False,
                )
                self.output_callback(error_utterance)
            except RuntimeError as callback_error:
                logger.error(f"Callback error: {callback_error}")

    def _transcribe(self, audio_np: np.ndarray) -> Dict[str, Any]:
        """
        Run MLX-Whisper on audio segment.

        Args:
            audio_np: Audio array (float32, mono, 16kHz)

        Returns:
            Dictionary with 'text' and 'confidence'
        """
        try:
            import mlx_whisper

            # CRITICAL: Serialize MLX calls across all sources to prevent threading issues
            with _MLX_WHISPER_LOCK:
                # Suppress MLX output by redirecting stdout/stderr to os.devnull
                # Context manager ensures proper cleanup (no leaked file descriptors)
                with open(os.devnull, "w") as devnull:
                    with (
                        contextlib.redirect_stdout(devnull),
                        contextlib.redirect_stderr(devnull),
                    ):
                        result = mlx_whisper.transcribe(
                            audio_np,
                            path_or_hf_repo=self.whisper_model_name,
                            language=self.language,
                            # CRITICAL: No initial_prompt for MLX (causes hallucinations)
                            condition_on_previous_text=False,
                            # Anti-hallucination settings
                            temperature=0.0,  # Greedy decoding only
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6,
                            word_timestamps=False,
                            verbose=False,
                        )

            # Extract text and segments (segments carry no_speech_prob per chunk)
            segments = result.get("segments", [])
            text = " ".join(
                [seg["text"].strip() for seg in segments if seg["text"].strip()]
            )

            return {"text": text, "segments": segments, "confidence": 1.0}

        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.error(
                f"MLX-Whisper transcription failed for source {self.source_id}: {e}",
                exc_info=True,
            )
            raise TranscriptionError(str(e), self.source_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get transcriber statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "source_id": self.source_id,
            "is_running": self._thread is not None and self._thread.is_alive(),
            "total_segments": self._total_segments,
            "total_inference_time": self._total_inference_time,
            "error_count": self._error_count,
            "queue_size": self.input_queue.qsize(),
        }

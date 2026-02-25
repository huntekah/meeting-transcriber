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
from typing import Callable, Dict, Any
import numpy as np

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import TranscriptionError
from ..schemas.transcription import Utterance

# Global lock to serialize MLX Whisper calls across all sources
# MLX might not be fully thread-safe for concurrent transcriptions
_MLX_WHISPER_LOCK = threading.Lock()


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

        # Thread lifecycle
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self._total_segments = 0
        self._total_inference_time = 0.0
        self._error_count = 0

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
        while not self._stop_event.is_set():
            try:
                # Get audio segment from queue (timeout to check stop_event)
                segment = self.input_queue.get(timeout=0.5)

                # Double-check stop event after getting item (might have been set during wait)
                if self._stop_event.is_set():
                    logger.debug(
                        f"LiveTranscriber {self.source_id} stopping, skipping segment"
                    )
                    break

                audio_np: np.ndarray = segment["audio"]
                capture_timestamp: float = segment["timestamp"]

                # Skip if too short (should be filtered by producer, but double-check)
                min_samples = int(settings.MIN_AUDIO_LENGTH * settings.SAMPLE_RATE)
                if len(audio_np) < min_samples:
                    continue

                # Transcribe
                start_time = time.time()
                result = self._transcribe(audio_np)
                inference_time = time.time() - start_time

                # Skip empty transcriptions
                if not result["text"].strip():
                    logger.debug(
                        f"Empty transcription for source {self.source_id}, skipping"
                    )
                    continue

                # Check again before callback (might be shutting down)
                if self._stop_event.is_set():
                    logger.debug(
                        f"LiveTranscriber {self.source_id} stopping, skipping callback"
                    )
                    break

                # Calculate timing
                duration = len(audio_np) / settings.SAMPLE_RATE
                end_timestamp = capture_timestamp + duration

                # Create utterance
                utterance = Utterance(
                    source_id=self.source_id,
                    start_time=capture_timestamp,
                    end_time=end_timestamp,
                    text=result["text"],
                    confidence=result.get("confidence", 1.0),
                    is_final=True,
                    overlaps_with=[],
                )

                # Send to merger via callback (only if not shutting down)
                try:
                    self.output_callback(utterance)
                except Exception as callback_error:
                    # Don't crash the thread if callback fails
                    logger.error(
                        f"Callback error for source {self.source_id}: {callback_error}"
                    )

                # Update stats
                self._total_segments += 1
                self._total_inference_time += inference_time

                logger.debug(
                    f"Transcribed segment for source {self.source_id}: "
                    f"'{result['text'][:50]}...' ({duration:.2f}s, RTF={inference_time / duration:.3f})"
                )

            except queue.Empty:
                # No segments available, continue
                continue

            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"LiveTranscriber error for source {self.source_id}: {e}",
                    exc_info=True,
                )

                # Send error utterance via callback (only if not shutting down)
                if not self._stop_event.is_set():
                    try:
                        error_utterance = Utterance(
                            source_id=self.source_id,
                            start_time=time.time(),
                            end_time=time.time(),
                            text=f"[TRANSCRIPTION ERROR: {str(e)}]",
                            confidence=0.0,
                            is_final=False,
                        )
                        self.output_callback(error_utterance)
                    except Exception as callback_error:
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

            # Extract text from segments
            segments = result.get("segments", [])
            text = " ".join(
                [seg["text"].strip() for seg in segments if seg["text"].strip()]
            )

            return {"text": text, "confidence": 1.0}  # MLX doesn't provide confidence

        except Exception as e:
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

"""
VAD-based audio producer service.

Thread-based audio capture with Voice Activity Detection using Silero VAD.
Adapted from scripts/live_test_v2.py producer_thread pattern.
"""

import time
import threading
import queue
from typing import Dict, Any
import numpy as np
import sounddevice as sd
import torch

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import AudioCaptureError
from .audio_producer import AudioProducerBase

# Global lock to serialize VAD model access across all sources
# Silero VAD is NOT thread-safe - internal RNN state gets corrupted with concurrent calls
_VAD_MODEL_LOCK = threading.Lock()


class VADAudioProducer(AudioProducerBase):
    """
    Thread-based audio capture with VAD-driven segmentation.

    Captures audio from a device, runs Silero VAD on each chunk, and produces
    speech segments to a queue when silence is detected.

    Thread-safe: Uses internal lock for buffer access and queue.Queue for output.
    """

    def __init__(
        self,
        source_id: int,
        device_index: int,
        device_name: str,
        vad_model: torch.nn.Module,
        output_queue: queue.Queue,
        device_channels: int = 1,
        sample_rate: int | None = None,
        chunk_size: int | None = None,
        vad_threshold: float | None = None,
        silence_chunks: int | None = None,
    ):
        """
        Initialize VAD audio producer.

        Args:
            source_id: Unique source identifier
            device_index: Sounddevice device index
            device_name: Human-readable device name
            vad_model: Silero VAD model
            output_queue: Queue to push finalized segments
            device_channels: Number of input channels the device supports (default 1)
            sample_rate: Sample rate in Hz (default from settings)
            chunk_size: Samples per chunk (default from settings)
            vad_threshold: Speech probability threshold (default from settings)
            silence_chunks: Number of silence chunks to trigger finalization (default from settings)
        """
        # Determine effective sample rate for parent initialization
        effective_sample_rate = sample_rate or settings.SAMPLE_RATE

        # Initialize parent class
        super().__init__(
            source_id=source_id,
            device_name=device_name,
            output_queue=output_queue,
            sample_rate=effective_sample_rate,
        )

        self.device_index = device_index
        self.vad_model = vad_model

        # Audio settings
        self.device_channels = device_channels
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.vad_threshold = vad_threshold or settings.VAD_THRESHOLD
        self.silence_chunks = silence_chunks or settings.SILENCE_CHUNKS

        # Thread-safe state
        self._buffer_lock = threading.Lock()
        self._growing_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_counter = 0

        # Audio recording for final save
        self._all_audio_chunks: list[np.ndarray] = []

        # VAD processing queue (callback -> VAD thread)
        self._vad_queue: queue.Queue = queue.Queue(maxsize=100)

        # Thread lifecycle
        self._thread: threading.Thread | None = None
        self._vad_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Session timing
        self._session_start_time: float = 0.0

    def start(self, session_start_time: float | None = None):
        """
        Start audio capture and VAD processing threads.

        Args:
            session_start_time: Unix timestamp of session start (default: current time)
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning(
                f"VADAudioProducer {self.source_id} already running, ignoring start"
            )
            return

        self._session_start_time = session_start_time or time.time()
        self._stop_event.clear()

        # Start VAD processing thread first (processes audio from queue)
        self._vad_thread = threading.Thread(
            target=self._vad_processing_loop, daemon=True
        )
        self._vad_thread.start()

        # Then start audio capture thread (feeds VAD queue)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"VADAudioProducer {self.source_id} started for device '{self.device_name}'"
        )

    def stop(self):
        """
        Gracefully stop audio capture and VAD processing.

        Finalizes any pending audio in the buffer before shutdown.
        """
        if self._thread is None:
            return

        logger.info(f"Stopping VADAudioProducer {self.source_id}...")
        self._stop_event.set()

        # Wait for capture thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Wait for VAD thread to finish
        if self._vad_thread and self._vad_thread.is_alive():
            self._vad_thread.join(timeout=2.0)

        # Finalize any remaining audio
        with self._buffer_lock:
            if len(self._growing_buffer) > 0:
                logger.info(
                    f"Finalizing {len(self._growing_buffer)} pending chunks for source {self.source_id}"
                )
                self._finalize_segment()

        logger.info(f"VADAudioProducer {self.source_id} stopped")

    def _capture_loop(self):
        """
        Main audio capture loop (runs in thread).

        Uses sounddevice.InputStream with callback for audio chunks.
        """

        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            """
            Audio callback invoked by sounddevice for each chunk.

            IMPORTANT: Minimal work here - just capture audio and queue it.
            VAD processing happens in _vad_processing_loop (separate thread).

            Args:
                indata: Audio data (frames, channels)
                frames: Number of frames
                time_info: Timing information
                status: Stream status
            """
            if status:
                logger.warning(f"Audio stream status: {status}")

            # Convert to mono float32
            # If stereo/multi-channel, average across channels (proper downmix)
            if self.device_channels > 1:
                # indata shape: (frames, channels) - average to mono
                audio_chunk = indata.mean(axis=1).astype(np.float32)
            else:
                # Already mono, just flatten
                audio_chunk = indata.flatten().astype(np.float32)

            # Save for final mixed file
            self._all_audio_chunks.append(audio_chunk)

            # Queue for VAD processing (non-blocking)
            try:
                self._vad_queue.put_nowait(audio_chunk)
            except queue.Full:
                # Drop frame if queue is full (shouldn't happen with maxsize=100)
                logger.warning(
                    f"VAD queue full for source {self.source_id}, dropping frame"
                )

        try:
            # Open audio stream with device's actual channel count
            # This prevents audio routing issues on stereo devices like BlackHole
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.device_channels,  # Use device's actual channels
                dtype="float32",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                logger.info(
                    f"Audio stream opened for source {self.source_id} "
                    f"(device {self.device_index}, {self.sample_rate}Hz)"
                )

                # Keep stream alive until stop event
                while not self._stop_event.is_set():
                    time.sleep(0.1)

        except Exception as e:
            logger.error(
                f"Audio capture failed for source {self.source_id}: {e}",
                exc_info=True,
            )
            raise AudioCaptureError(self.device_index, str(e))

    def _vad_processing_loop(self):
        """
        VAD processing loop (runs in separate Python thread).

        Safely processes audio from _vad_queue and runs VAD model.
        This runs in a proper Python thread (not C callback), so PyTorch is safe.
        """
        logger.info(f"VAD processing thread started for source {self.source_id}")

        while not self._stop_event.is_set():
            try:
                # Get audio chunk from queue (blocking with timeout)
                try:
                    audio_chunk = self._vad_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Append to growing buffer
                with self._buffer_lock:
                    self._growing_buffer.append(audio_chunk)

                # Run VAD (safe to call PyTorch model from Python thread)
                # IMPORTANT: Use global lock to serialize VAD calls across all sources
                try:
                    audio_tensor = torch.from_numpy(audio_chunk)

                    # Serialize VAD model access - model is NOT thread-safe
                    with _VAD_MODEL_LOCK:
                        vad_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                    # VAD state machine
                    if vad_prob > self.vad_threshold:
                        # Speech detected
                        if not self._is_speaking:
                            logger.debug(
                                f"[Source {self.source_id}] Speech START (VAD prob: {vad_prob:.3f})"
                            )
                        self._is_speaking = True
                        self._silence_counter = 0
                    else:
                        # Silence detected
                        if self._is_speaking:
                            self._silence_counter += 1
                            if self._silence_counter >= self.silence_chunks:
                                # Enough silence - finalize segment
                                logger.debug(
                                    f"[Source {self.source_id}] Speech END (silence chunks: {self._silence_counter})"
                                )
                                with self._buffer_lock:
                                    self._finalize_segment()
                                self._is_speaking = False
                                self._silence_counter = 0

                except Exception as e:
                    logger.error(
                        f"VAD error for source {self.source_id}: {e}", exc_info=True
                    )

            except Exception as e:
                logger.error(
                    f"VAD processing loop error for source {self.source_id}: {e}",
                    exc_info=True,
                )

        logger.info(f"VAD processing thread stopped for source {self.source_id}")

    def _finalize_segment(self):
        """
        Finalize buffered audio and push to queue for transcription.

        Called when VAD detects end of speech (15 silence chunks).
        Assumes _buffer_lock is already held.
        """
        if len(self._growing_buffer) == 0:
            return

        # Concatenate buffer chunks
        audio_np = np.concatenate(self._growing_buffer)
        self._growing_buffer.clear()

        # Skip if too short
        min_samples = int(settings.MIN_AUDIO_LENGTH * self.sample_rate)
        if len(audio_np) < min_samples:
            logger.debug(
                f"Skipping short segment for source {self.source_id}: "
                f"{len(audio_np)} samples < {min_samples}"
            )
            return

        # Calculate timestamp
        current_time = time.time()

        # Create segment dictionary
        segment: Dict[str, Any] = {
            "audio": audio_np,
            "timestamp": current_time,
            "source_id": self.source_id,
        }

        # Push to queue (non-blocking with timeout)
        try:
            self.output_queue.put(segment, timeout=1.0)
            logger.debug(
                f"Finalized segment for source {self.source_id}: "
                f"{len(audio_np)} samples ({len(audio_np) / self.sample_rate:.2f}s)"
            )
        except queue.Full:
            logger.warning(
                f"Output queue full for source {self.source_id}, dropping segment"
            )

    def get_full_audio(self) -> np.ndarray:
        """
        Get complete audio recording for this source.

        Returns:
            Concatenated numpy array of all captured audio
        """
        if len(self._all_audio_chunks) > 0:
            return np.concatenate(self._all_audio_chunks)
        return np.array([], dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get producer statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "source_id": self.source_id,
            "device_name": self.device_name,
            "is_running": self._thread is not None and self._thread.is_alive(),
            "total_audio_chunks": len(self._all_audio_chunks),
            "buffer_size": len(self._growing_buffer),
            "is_speaking": self._is_speaking,
            "silence_counter": self._silence_counter,
        }

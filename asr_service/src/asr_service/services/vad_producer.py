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


class VADAudioProducer:
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
            sample_rate: Sample rate in Hz (default from settings)
            chunk_size: Samples per chunk (default from settings)
            vad_threshold: Speech probability threshold (default from settings)
            silence_chunks: Number of silence chunks to trigger finalization (default from settings)
        """
        self.source_id = source_id
        self.device_index = device_index
        self.device_name = device_name
        self.vad_model = vad_model
        self.output_queue = output_queue

        # Audio settings
        self.sample_rate = sample_rate or settings.SAMPLE_RATE
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

        # Thread lifecycle
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Session timing
        self._session_start_time: float = 0.0

    def start(self, session_start_time: float | None = None):
        """
        Start audio capture thread.

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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"VADAudioProducer {self.source_id} started for device '{self.device_name}'"
        )

    def stop(self):
        """
        Gracefully stop audio capture.

        Finalizes any pending audio in the buffer before shutdown.
        """
        if self._thread is None:
            return

        logger.info(f"Stopping VADAudioProducer {self.source_id}...")
        self._stop_event.set()

        # Wait for thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

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

            Args:
                indata: Audio data (frames, channels)
                frames: Number of frames
                time_info: Timing information
                status: Stream status
            """
            if status:
                logger.warning(f"Audio stream status: {status}")

            # Convert to mono float32
            audio_chunk = indata[:, 0].astype(np.float32).copy()

            # Save for final mixed file
            self._all_audio_chunks.append(audio_chunk)

            # Append to growing buffer
            with self._buffer_lock:
                self._growing_buffer.append(audio_chunk)

            # Run VAD
            try:
                audio_tensor = torch.from_numpy(audio_chunk)
                vad_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                # VAD state machine
                if vad_prob > self.vad_threshold:
                    # Speech detected
                    self._is_speaking = True
                    self._silence_counter = 0
                else:
                    # Silence detected
                    if self._is_speaking:
                        self._silence_counter += 1
                        if self._silence_counter >= self.silence_chunks:
                            # Enough silence - finalize segment
                            self._finalize_segment()
                            self._is_speaking = False
                            self._silence_counter = 0

            except Exception as e:
                logger.error(
                    f"VAD error for source {self.source_id}: {e}", exc_info=True
                )

        try:
            # Open audio stream
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
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
            'audio': audio_np,
            'timestamp': current_time,
            'source_id': self.source_id,
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
            'source_id': self.source_id,
            'device_name': self.device_name,
            'is_running': self._thread is not None and self._thread.is_alive(),
            'total_audio_chunks': len(self._all_audio_chunks),
            'buffer_size': len(self._growing_buffer),
            'is_speaking': self._is_speaking,
            'silence_counter': self._silence_counter,
        }

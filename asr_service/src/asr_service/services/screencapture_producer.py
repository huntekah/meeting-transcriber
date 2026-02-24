"""
ScreenCaptureKit-based audio producer.

Captures system audio on macOS using ScreenCaptureKit via a compiled Swift binary.
Outputs audio segments to queue for transcription.
"""

import subprocess
import threading
import queue
import time
from typing import Dict, Any
import numpy as np
from pathlib import Path

from ..core.config import settings
from ..core.logging import logger
from ..utils.file_ops import get_project_root
from .audio_producer import AudioProducerBase


class ScreenCaptureAudioProducer(AudioProducerBase):
    """
    ScreenCaptureKit-based audio producer for macOS system audio capture.

    Spawns the screencapture_audio Swift binary as a subprocess and captures
    system audio. Reads PCM data from stdout and segments it for transcription.

    Thread-safe: Uses internal threading for subprocess communication.
    """

    def __init__(
        self,
        source_id: int,
        device_name: str,
        output_queue: queue.Queue,
        sample_rate: int = 16000,
        segment_duration: float = 1.0,
        binary_path: str | Path | None = None,
    ):
        """
        Initialize ScreenCaptureKit audio producer.

        Args:
            source_id: Unique source identifier
            device_name: Human-readable device name (e.g., "System Audio (ScreenCaptureKit)")
            output_queue: Queue to push finalized audio segments
            sample_rate: Sample rate in Hz (default 16000)
            segment_duration: Duration in seconds to buffer before pushing segment (default 1.0s)
            binary_path: Path to compiled screencapture_audio binary
                        (default: scripts/screencapture_audio)
        """
        # Initialize parent class
        super().__init__(
            source_id=source_id,
            device_name=device_name,
            output_queue=output_queue,
            sample_rate=sample_rate,
        )

        # Determine binary path (relative to repo scripts folder)
        if binary_path is None:
            try:
                repo_root = get_project_root()
                binary_path = repo_root / "scripts" / "screencapture_audio"
            except FileNotFoundError as e:
                logger.warning(f"Could not find project root: {e}")
                binary_path = Path("scripts/screencapture_audio")
        self.binary_path = Path(binary_path)

        # Capture parameters
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)

        # Thread management
        self._process: subprocess.Popen | None = None
        self._read_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Audio recording for final save
        self._all_audio_chunks: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # Statistics
        self._total_segments_captured = 0
        self._total_samples_captured = 0

    def start(self, session_start_time: float | None = None):
        """
        Start ScreenCaptureKit audio capture.

        Spawns the screencapture_audio binary and begins reading PCM data.

        Args:
            session_start_time: Unix timestamp when session started (for synchronization)
        """
        if self._process is not None:
            logger.warning(
                f"ScreenCaptureAudioProducer {self.source_id} already started"
            )
            return

        logger.info(
            f"Starting ScreenCaptureAudioProducer {self.source_id} "
            f"(device: {self.device_name}, binary: {self.binary_path})"
        )

        # Check if binary exists
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"ScreenCaptureKit binary not found: {self.binary_path}\n"
                f"Compile with: swiftc scripts/screencapture_audio.swift -o {self.binary_path}"
            )

        # Spawn subprocess
        try:
            self._process = subprocess.Popen(
                [str(self.binary_path), str(self.sample_rate)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
            )
            logger.info(f"ScreenCaptureKit subprocess started (PID: {self._process.pid})")
        except Exception as e:
            logger.error(f"Failed to start ScreenCaptureKit subprocess: {e}")
            raise

        # Start read thread
        self._stop_event.clear()
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name=f"ScreenCaptureRead-{self.source_id}",
            daemon=False,
        )
        self._read_thread.start()
        logger.info(f"ScreenCaptureAudioProducer {self.source_id} started")

    def stop(self) -> None:
        """
        Stop ScreenCaptureKit audio capture.

        Terminates the subprocess and waits for read thread to finish.
        """
        logger.info(f"Stopping ScreenCaptureAudioProducer {self.source_id}...")

        # Signal read thread to stop
        self._stop_event.set()

        # Terminate subprocess
        if self._process is not None:
            try:
                self._process.terminate()
                # Wait for process to exit (with timeout)
                self._process.wait(timeout=2.0)
                logger.info(f"ScreenCaptureKit process terminated (PID: {self._process.pid})")
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"ScreenCaptureKit process didn't terminate, killing it"
                )
                self._process.kill()
                self._process.wait()
            self._process = None

        # Wait for read thread to finish
        if self._read_thread is not None:
            self._read_thread.join(timeout=3.0)
            if self._read_thread.is_alive():
                logger.warning(
                    f"Read thread for source {self.source_id} still alive after timeout"
                )
            self._read_thread = None

        logger.info(
            f"ScreenCaptureAudioProducer {self.source_id} stopped "
            f"({self._total_segments_captured} segments, {self._total_samples_captured} samples)"
        )

    def _read_loop(self):
        """
        Read PCM data from subprocess stdout and segment it.

        Runs in background thread. Accumulates audio and pushes segments
        to output_queue at regular intervals.
        """
        logger.debug(f"Read loop started for source {self.source_id}")

        try:
            if self._process is None or self._process.stdout is None:
                logger.error(f"Invalid subprocess for source {self.source_id}")
                return

            buffer = np.array([], dtype=np.float32)
            bytes_per_sample = 4  # float32 = 4 bytes

            while not self._stop_event.is_set():
                try:
                    # Read chunk from stdout
                    chunk_bytes = self._process.stdout.read(4096)

                    if not chunk_bytes:
                        # EOF - process finished
                        logger.debug(
                            f"EOF reached for source {self.source_id}, "
                            f"finalizing {len(buffer)} remaining samples"
                        )
                        # Push any remaining audio
                        if len(buffer) > 0:
                            self._push_segment(buffer)
                        break

                    # Convert bytes to float32 array
                    chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)
                    buffer = np.concatenate([buffer, chunk_data])

                    # Push segment if we have enough audio
                    while len(buffer) >= self.segment_samples:
                        segment_audio = buffer[:self.segment_samples]
                        buffer = buffer[self.segment_samples:]
                        self._push_segment(segment_audio)

                except Exception as e:
                    logger.error(
                        f"Error reading from subprocess for source {self.source_id}: {e}",
                        exc_info=True,
                    )
                    break

        finally:
            logger.debug(f"Read loop exited for source {self.source_id}")

    def _push_segment(self, audio_data: np.ndarray):
        """
        Push audio segment to output queue.

        Args:
            audio_data: Numpy array of audio samples (float32)
        """
        if audio_data is None or len(audio_data) == 0:
            return

        current_time = time.time()

        # Create segment dictionary (same format as VADAudioProducer)
        segment: Dict[str, Any] = {
            "audio": audio_data,
            "timestamp": current_time,
            "source_id": self.source_id,
        }

        # Push to queue (non-blocking with timeout)
        try:
            self.output_queue.put(segment, timeout=1.0)
            with self._buffer_lock:
                self._all_audio_chunks.append(audio_data)
                self._total_segments_captured += 1
                self._total_samples_captured += len(audio_data)
            logger.debug(
                f"Pushed segment for source {self.source_id}: "
                f"{len(audio_data)} samples ({len(audio_data) / self.sample_rate:.2f}s)"
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
        with self._buffer_lock:
            if len(self._all_audio_chunks) > 0:
                return np.concatenate(self._all_audio_chunks)
        return np.array([], dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get producer statistics.

        Returns:
            Dictionary with statistics
        """
        with self._buffer_lock:
            return {
                "source_id": self.source_id,
                "device_name": self.device_name,
                "device_type": "screencapture",
                "is_running": self._process is not None and self._process.poll() is None,
                "total_segments": self._total_segments_captured,
                "total_samples": self._total_samples_captured,
                "binary_path": str(self.binary_path),
            }

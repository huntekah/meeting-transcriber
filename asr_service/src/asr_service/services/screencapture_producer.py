"""
ScreenCaptureKit-based audio producer.

Captures system audio on macOS using ScreenCaptureKit via a compiled Swift binary.
Outputs audio segments to queue for transcription.
"""

import subprocess  # nosec B404
import threading
import queue
from typing import Dict, Any
import numpy as np
import torch
from pathlib import Path

from ..core.config import settings
from ..core.logging import logger
from ..utils.file_ops import get_project_root
from .audio_producer import AudioProducerBase
from .vad_streaming import VADStreamingBuffer


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
        vad_model: torch.nn.Module,
        sample_rate: int = 16000,
        segment_duration: float = 1.0,
        capture_duration_seconds: int | None = None,
        binary_path: str | Path | None = None,
    ):
        """
        Initialize ScreenCaptureKit audio producer.

        Args:
            source_id: Unique source identifier
            device_name: Human-readable device name (e.g., "System Audio (ScreenCaptureKit)")
            output_queue: Queue to push finalized audio segments
            vad_model: Silero VAD model
            sample_rate: Sample rate in Hz (default 16000)
            segment_duration: Duration in seconds to buffer before pushing segment (default 1.0s)
            capture_duration_seconds: Max duration passed to screencapture binary (default from settings)
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
        self.capture_duration_seconds = (
            capture_duration_seconds
            if capture_duration_seconds is not None
            else settings.SCREENCAPTURE_MAX_DURATION_SECONDS
        )

        # VAD streaming buffer
        self._vad_stream = VADStreamingBuffer(
            source_id=source_id,
            sample_rate=self.sample_rate,
            vad_model=vad_model,
            # Uses BREATH_SILENCE_CHUNKS, SEMANTIC_SILENCE_CHUNKS, MAX_UTTERANCE_SECONDS from settings
        )
        self._vad_chunk_buffer = np.array([], dtype=np.float32)

        # Thread management
        self._process: subprocess.Popen | None = None
        self._read_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
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
            self._process = subprocess.Popen(  # nosec B603
                [
                    str(self.binary_path),
                    str(self.sample_rate),
                    str(self.capture_duration_seconds),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
            )
            logger.info(
                f"ScreenCaptureKit subprocess started (PID: {self._process.pid})"
            )
        except Exception as e:
            logger.error(f"Failed to start ScreenCaptureKit subprocess: {e}")
            self._process = None
            raise

        # Start read thread
        try:
            self._stop_event.clear()
            self._read_thread = threading.Thread(
                target=self._read_loop,
                name=f"ScreenCaptureRead-{self.source_id}",
                daemon=False,
            )
            self._read_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop,
                name=f"ScreenCaptureStderr-{self.source_id}",
                daemon=False,
            )
            self._stderr_thread.start()
            logger.info(f"ScreenCaptureAudioProducer {self.source_id} started")
        except Exception as e:
            # Thread start failed - cleanup the subprocess
            logger.error(f"Failed to start read thread: {e}", exc_info=True)
            if self._process:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                self._process = None
            raise

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
                logger.info(
                    f"ScreenCaptureKit process terminated (PID: {self._process.pid})"
                )
            except subprocess.TimeoutExpired:
                logger.warning("ScreenCaptureKit process didn't terminate, killing it")
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
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=3.0)
            if self._stderr_thread.is_alive():
                logger.warning(
                    f"Stderr thread for source {self.source_id} still alive after timeout"
                )
            self._stderr_thread = None

        segment = self._vad_stream.finalize_pending()
        if segment:
            self._push_segment(segment)

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

            while not self._stop_event.is_set():
                try:
                    # Read chunk from stdout
                    chunk_bytes = self._process.stdout.read(4096)

                    if not chunk_bytes:
                        # EOF - process finished
                        if self._stop_event.is_set():
                            logger.debug(
                                f"EOF reached for source {self.source_id} after stop request, "
                                f"finalizing remaining samples"
                            )
                        else:
                            returncode = self._process.poll()
                            logger.warning(
                                f"ScreenCaptureKit subprocess exited unexpectedly "
                                f"(source {self.source_id}, code={returncode})"
                            )
                        # Push any remaining audio
                        segment = self._vad_stream.finalize_pending()
                        if segment:
                            self._push_segment(segment)
                        break

                    # Convert bytes to float32 array
                    chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)
                    with self._buffer_lock:
                        self._all_audio_chunks.append(chunk_data)

                    self._vad_chunk_buffer = np.concatenate(
                        [self._vad_chunk_buffer, chunk_data]
                    )

                    while len(self._vad_chunk_buffer) >= settings.CHUNK_SIZE:
                        vad_chunk = self._vad_chunk_buffer[: settings.CHUNK_SIZE]
                        self._vad_chunk_buffer = self._vad_chunk_buffer[
                            settings.CHUNK_SIZE :
                        ]
                        segment = self._vad_stream.append_chunk(vad_chunk)
                        if segment:
                            self._push_segment(segment)

                except (OSError, RuntimeError, ValueError) as e:
                    logger.error(
                        f"Error reading from subprocess for source {self.source_id}: {e}",
                        exc_info=True,
                    )
                    break

        finally:
            logger.debug(f"Read loop exited for source {self.source_id}")

    def _stderr_loop(self):
        """
        Read stderr from subprocess and log ScreenCaptureKit output.
        """
        if self._process is None or self._process.stderr is None:
            logger.error(f"Invalid stderr pipe for source {self.source_id}")
            return

        for raw_line in iter(self._process.stderr.readline, b""):
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("ERROR"):
                logger.error(f"ScreenCaptureKit[{self.source_id}] {line}")
            elif line.startswith("INFO"):
                logger.info(f"ScreenCaptureKit[{self.source_id}] {line}")
            else:
                logger.warning(f"ScreenCaptureKit[{self.source_id}] {line}")

        logger.debug(f"Stderr loop exited for source {self.source_id}")

    def _push_segment(self, segment: Dict[str, Any]):
        """
        Push audio segment to output queue.

        Args:
            segment: Segment dictionary with audio and metadata
        """
        audio_data = segment.get("audio")
        if audio_data is None or len(audio_data) == 0:
            return

        # Push to queue (non-blocking with timeout)
        try:
            self.output_queue.put(segment, timeout=1.0)
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
            self._vad_stream.clear_commit_ready()

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
        stats = self._vad_stream.get_stats()
        return {
            "source_id": self.source_id,
            "device_name": self.device_name,
            "device_type": "screencapture",
            "is_running": self._process is not None and self._process.poll() is None,
            "total_segments": self._total_segments_captured,
            "total_samples": self._total_samples_captured,
            "binary_path": str(self.binary_path),
            "buffer_size": stats["buffer_size"],
            "is_speaking": stats["is_speaking"],
            "commit_ready": stats["commit_ready"],
        }

    def get_streaming_snapshot(self) -> Dict[str, Any]:
        """Snapshot of streaming state for provisional transcription."""
        return self._vad_stream.get_streaming_snapshot()

    def clear_commit_ready(self) -> None:
        """Clear commit_ready after finalization is consumed."""
        self._vad_stream.clear_commit_ready()

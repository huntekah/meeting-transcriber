"""
Source pipeline service.

Composes VADAudioProducer + LiveTranscriber for a single audio source.
Represents one audio stream in a multi-source session.
"""

import time
import queue
from typing import Callable, Dict, Any
import numpy as np
import torch

from ..core.logging import logger
from ..schemas.transcription import Utterance
from .vad_producer import VADAudioProducer
from .live_transcriber import LiveTranscriber


class SourcePipeline:
    """
    Logical pairing of VADAudioProducer + LiveTranscriber.

    Represents one audio source in a multi-source session.
    Handles producer-consumer communication via queue.
    """

    def __init__(
        self,
        source_id: int,
        device_index: int,
        device_name: str,
        vad_model: torch.nn.Module,
        whisper_model_name: str,
        utterance_callback: Callable[[Utterance], None],
        device_channels: int = 1,
        language: str = "en",
    ):
        """
        Initialize source pipeline.

        Args:
            source_id: Unique source identifier
            device_index: Audio device index
            device_name: Human-readable device name
            vad_model: Silero VAD model
            whisper_model_name: MLX-Whisper model name
            utterance_callback: Callback to receive transcribed utterances
            device_channels: Number of input channels the device supports
            language: Language code for transcription
        """
        self.source_id = source_id
        self.device_index = device_index
        self.device_name = device_name

        # Create shared queue between producer and consumer
        # Bounded queue to prevent memory issues
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)

        # Create producer
        self.producer = VADAudioProducer(
            source_id=source_id,
            device_index=device_index,
            device_name=device_name,
            vad_model=vad_model,
            output_queue=self._audio_queue,
            device_channels=device_channels,
        )

        # Create consumer
        self.transcriber = LiveTranscriber(
            source_id=source_id,
            input_queue=self._audio_queue,
            output_callback=utterance_callback,
            whisper_model_name=whisper_model_name,
            language=language,
        )

        logger.info(
            f"SourcePipeline {source_id} initialized for device '{device_name}' (index {device_index})"
        )

    def start(self, session_start_time: float | None = None):
        """
        Start both producer and consumer threads.

        Args:
            session_start_time: Unix timestamp of session start (default: current time)
        """
        logger.info(f"Starting SourcePipeline {self.source_id}...")

        # Start producer first
        self.producer.start(session_start_time)

        # Start consumer
        self.transcriber.start()

        logger.info(f"SourcePipeline {self.source_id} started")

    def stop(self):
        """
        Gracefully shutdown in correct order.

        Order is critical:
        1. Stop producer (no more audio input)
        2. Wait for queue to drain
        3. Stop consumer (finish pending transcriptions)
        """
        logger.info(f"Stopping SourcePipeline {self.source_id}...")

        # 1. Stop producer first (no more audio)
        self.producer.stop()

        # 2. Wait for queue to drain (with timeout)
        logger.info(
            f"Waiting for queue to drain for source {self.source_id} "
            f"(current size: {self._audio_queue.qsize()})"
        )
        max_wait = 10.0  # seconds
        wait_start = time.time()
        while not self._audio_queue.empty() and (time.time() - wait_start) < max_wait:
            time.sleep(0.1)

        if not self._audio_queue.empty():
            logger.warning(
                f"Queue for source {self.source_id} not empty after {max_wait}s "
                f"(size: {self._audio_queue.qsize()})"
            )

        # 3. Stop transcriber
        self.transcriber.stop()

        logger.info(f"SourcePipeline {self.source_id} stopped")

    def get_audio(self) -> np.ndarray:
        """
        Get full audio recording from this source.

        Returns:
            Numpy array of all captured audio
        """
        return self.producer.get_full_audio()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with producer and consumer stats
        """
        return {
            "source_id": self.source_id,
            "device_name": self.device_name,
            "producer": self.producer.get_stats(),
            "transcriber": self.transcriber.get_stats(),
            "queue_size": self._audio_queue.qsize(),
        }

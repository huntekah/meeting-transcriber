"""
Source pipeline service.

Composes AudioProducerBase + LiveTranscriber for a single audio source.
Represents one audio stream in a multi-source session.
"""

import time
import queue
from typing import Callable, Dict, Any
import numpy as np

from ..core.logging import logger
from ..schemas.transcription import Utterance
from .audio_producer import AudioProducerBase
from .live_transcriber import LiveTranscriber


class SourcePipeline:
    """
    Logical pairing of AudioProducerBase + LiveTranscriber.

    Represents one audio source in a multi-source session.
    Handles producer-consumer communication via queue.
    """

    def __init__(
        self,
        source_id: int,
        producer: AudioProducerBase,
        whisper_model_name: str,
        utterance_callback: Callable[[Utterance], None],
        language: str = "en",
    ):
        """
        Initialize source pipeline.

        Args:
            source_id: Unique source identifier
            producer: AudioProducerBase instance for capturing audio
            whisper_model_name: MLX-Whisper model name
            utterance_callback: Callback to receive transcribed utterances
            language: Language code for transcription
        """
        self.source_id = source_id
        self.producer = producer
        self.device_name = producer.device_name

        # Create shared queue between producer and consumer
        # Bounded queue to prevent memory issues
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)

        # Update producer's output queue to use our queue
        self.producer.output_queue = self._audio_queue

        # Create consumer
        self.transcriber = LiveTranscriber(
            source_id=source_id,
            input_queue=self._audio_queue,
            output_callback=utterance_callback,
            whisper_model_name=whisper_model_name,
            language=language,
        )

        logger.info(
            f"SourcePipeline {source_id} initialized for device '{self.device_name}'"
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

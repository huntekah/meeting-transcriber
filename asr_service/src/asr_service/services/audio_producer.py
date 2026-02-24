"""
Abstract base class for audio producers.

Defines the interface for different audio capture sources (sounddevice, ScreenCaptureKit, etc).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import queue
import numpy as np


class AudioProducerBase(ABC):
    """
    Abstract base class for audio producers.

    Defines the interface that all audio capture sources must implement.
    Different implementations (VADAudioProducer, ScreenCaptureAudioProducer, etc)
    inherit from this class.
    """

    def __init__(
        self,
        source_id: int,
        device_name: str,
        output_queue: queue.Queue,
        sample_rate: int = 16000,
    ):
        """
        Initialize audio producer base.

        Args:
            source_id: Unique source identifier
            device_name: Human-readable device name
            output_queue: Queue to push finalized audio segments
            sample_rate: Sample rate in Hz (default 16000)
        """
        self.source_id = source_id
        self.device_name = device_name
        self.output_queue = output_queue
        self.sample_rate = sample_rate

    @abstractmethod
    def start(self, session_start_time: float | None = None):
        """
        Start audio capture.

        Args:
            session_start_time: Unix timestamp when session started (for synchronization)
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture and clean up resources."""
        pass

    @abstractmethod
    def get_full_audio(self) -> np.ndarray:
        """
        Get the complete audio recording as a numpy array.

        Returns:
            Numpy array of audio samples (float32)
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get producer statistics.

        Returns:
            Dictionary with producer-specific statistics
        """
        pass

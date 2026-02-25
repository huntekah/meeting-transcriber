"""
Model Manager - Singleton for model lifecycle management.

Manages loading and caching of ML models (VAD, Whisper, Diarization).
Thread-safe singleton with lazy loading.
"""

import sys
import threading
import asyncio
from pathlib import Path
from typing import Optional
import torch

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import ModelLoadingError
from ..utils.file_ops import get_project_root


class ModelManager:
    """
    Thread-safe singleton for managing model lifecycle.

    Models are lazy-loaded on first use to reduce startup time.
    Supports async loading to not block FastAPI event loop.
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False
    _lock = threading.Lock()
    _cold_pipeline_lock = threading.Lock()  # Serialize cold pipeline access (Numba/pyannote not thread-safe)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Model instances
        self.vad_model: Optional[torch.nn.Module] = None
        self.cold_pipeline = None  # ColdPathPipeline_MLX
        self.whisper_model_name: str = settings.MLX_WHISPER_MODEL

        # Lazy loading flags
        self._models_loaded = False
        self._load_lock = asyncio.Lock()

        self._initialized = True
        logger.info("ModelManager singleton initialized")

    async def load_models(self):
        """
        Lazy load models (only once).

        Runs in executor to not block async event loop.
        Thread-safe with lock.
        """
        if self._models_loaded:
            logger.debug("Models already loaded, skipping")
            return

        async with self._load_lock:
            if self._models_loaded:
                return

            logger.info("Loading ASR models...")

            # Load in executor to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_models_sync)

            self._models_loaded = True
            logger.info("Models loaded successfully")

    def _load_models_sync(self):
        """
        Synchronous model loading (runs in executor).

        Loads:
        1. Silero VAD model (shared reference for creating instances)
        2. MLX-Whisper model (pre-loaded with dummy transcription)
        3. Cold path pipeline (lazy - only when cold processing needed)
        """
        try:
            # Load Silero VAD (this creates a shared model we'll use to get fresh instances)
            logger.info("Loading Silero VAD model...")
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
                onnx=False,
            )
            logger.info("Silero VAD model loaded")

            # Pre-load MLX-Whisper by running dummy transcription
            logger.info(f"Pre-loading MLX-Whisper model: {self.whisper_model_name}")
            self._preload_mlx_whisper()
            logger.info("MLX-Whisper model pre-loaded")

            # Cold pipeline is lazy-loaded only when needed
            logger.info("Cold path pipeline will be loaded on demand")

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            raise ModelLoadingError("VAD/Whisper", str(e))

    def _preload_mlx_whisper(self):
        """
        Pre-load MLX Whisper by running a dummy transcription.

        This loads the model into memory so the first real transcription is fast.
        """
        try:
            import mlx_whisper
            import numpy as np

            # Create 1 second of silence
            dummy_audio = np.zeros(16000, dtype=np.float32)

            logger.info("Running dummy transcription to load MLX Whisper...")

            # This will load the model into memory
            mlx_whisper.transcribe(
                dummy_audio,
                path_or_hf_repo=self.whisper_model_name,
                language="en",
                verbose=False,
            )

            logger.info("MLX Whisper model loaded into memory")

        except Exception as e:
            logger.warning(f"MLX Whisper pre-load failed (non-critical): {e}")
            # Don't raise - this is optimization, not required

    def get_cold_pipeline(self):
        """
        Lazy-load cold path pipeline (heavy, only needed for post-processing).

        Returns:
            ColdPathPipeline_MLX instance
        """
        if self.cold_pipeline is not None:
            return self.cold_pipeline

        logger.info("Loading cold path pipeline...")

        try:
            # Add scripts directory to path
            try:
                repo_root = get_project_root()
                scripts_dir = repo_root / "scripts"
            except FileNotFoundError:
                logger.warning("Could not find project root, using relative path")
                scripts_dir = Path("scripts")

            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            # Import cold path pipeline
            from cold_path_pipeline_v2 import ColdPathPipeline_MLX

            # Initialize pipeline
            self.cold_pipeline = ColdPathPipeline_MLX(
                whisper_model=settings.MLX_WHISPER_MODEL,
                diarization_model=settings.DIARIZATION_MODEL,
                hf_token=settings.HF_TOKEN,
                use_diarization=True,
                verbose=False,
            )

            logger.info("Cold path pipeline loaded")
            return self.cold_pipeline

        except Exception as e:
            logger.error(f"Cold pipeline loading failed: {e}", exc_info=True)
            raise ModelLoadingError("ColdPathPipeline_MLX", str(e))

    def is_loaded(self) -> bool:
        """
        Check if base models (VAD, Whisper name) are loaded.

        Returns:
            True if models are ready
        """
        return self._models_loaded

    def unload_models(self):
        """
        Cleanup models for graceful shutdown.

        Frees GPU memory and resets state.
        """
        logger.info("Unloading models...")

        self.vad_model = None
        self.cold_pipeline = None
        self._models_loaded = False

        # Force garbage collection
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded")

    def get_stats(self) -> dict:
        """
        Get model manager statistics.

        Returns:
            Dictionary with model status
        """
        return {
            "models_loaded": self._models_loaded,
            "vad_loaded": self.vad_model is not None,
            "cold_pipeline_loaded": self.cold_pipeline is not None,
            "whisper_model_name": self.whisper_model_name,
        }

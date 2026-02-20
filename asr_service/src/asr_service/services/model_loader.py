import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional
from asr_service.core.config import settings
from asr_service.core.logging import logger


class ASREngine:
    """Singleton class to manage ASR model loading and inference."""

    _instance: Optional["ASREngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = settings.get_device()
        self.torch_dtype = settings.get_torch_dtype()
        self.model_id = settings.MODEL_ID

        self.model = None
        self.processor = None
        self.final_pipeline = None
        self.live_pipeline = None
        self.is_loaded = False

        self._initialized = True

    def load_model(self):
        """Load the Whisper model and create both final and live pipelines."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return

        try:
            logger.info(f"Loading model {self.model_id} on device {self.device}")

            # Load model
            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            }

            # Add flash attention if enabled and on CUDA
            if settings.USE_FLASH_ATTENTION and self.device == "cuda":
                load_kwargs["attn_implementation"] = "flash_attention_2"

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Create FINAL pipeline (high precision, no chunking)
            self.final_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            # Create LIVE pipeline (fast, with chunking)
            self.live_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=settings.LIVE_CHUNK_LENGTH_S,
                batch_size=settings.LIVE_BATCH_SIZE,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            self.is_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe_final(self, audio_path: str) -> dict:
        """
        Transcribe audio with high precision settings.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription text and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        generate_kwargs = {
            "num_beams": settings.FINAL_BEAM_SIZE,
            "condition_on_prev_tokens": settings.FINAL_CONDITION_ON_PREV,
            "task": "transcribe",
        }

        result = self.final_pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        return result

    def transcribe_live(self, audio_path: str) -> dict:
        """
        Transcribe audio with fast settings for live streaming.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        result = self.live_pipeline(
            audio_path,
            return_timestamps=True,
        )

        return result

    def get_final_model(self):
        """Get the final (high precision) pipeline."""
        return self.final_pipeline

    def get_live_model(self):
        """Get the live (fast) pipeline."""
        return self.live_pipeline

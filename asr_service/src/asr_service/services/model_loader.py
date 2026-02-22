"""
ASR Engine - Main inference engine for transcription.

SHOULD:
- Wrap scripts/cold_path_pipeline_v2.ColdPathPipeline_MLX
- Provide singleton pattern for model lifecycle management
- Support both cold path (batch) and live streaming modes
- Handle model loading, caching, and cleanup
- Expose simple API: transcribe_file(), transcribe_stream()

CONTRACTS:
- Class: ASREngine (singleton)
- Methods:
  - load_model() -> None  # Load MLX model + diarization model
  - transcribe_file(audio_path, use_diarization=True, language="en") -> dict
  - is_loaded -> bool  # Property to check if models are ready
  - unload_model() -> None  # Cleanup for graceful shutdown

- Response format (matches cold_path_pipeline_v2.py):
  {
      "segments": [
          {"start": float, "end": float, "text": str, "speaker": str | None}
      ],
      "duration": float,
      "language": str
  }

IMPLEMENTATION NOTES:
- Import from: from scripts.cold_path_pipeline_v2 import ColdPathPipeline_MLX
- Use config from: from asr_service.core.config import settings
- Initialize pipeline in load_model() with settings.MLX_WHISPER_MODEL
- Enable parallel processing by default (parallel=True)
- Handle errors gracefully with logging

REMOVE:
- All old transformers.pipeline code
- FINAL/LIVE pipeline distinction (just use cold path for both)
- Old beam search settings
"""

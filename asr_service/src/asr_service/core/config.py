"""
Configuration settings for ASR service.

SHOULD:
- Load settings from environment variables using pydantic-settings
- Support both faster-whisper and MLX-Whisper model configurations
- Include settings for:
  - Model selection (WHISPER_MODEL, MLX_WHISPER_MODEL)
  - Diarization (DIARIZATION_MODEL, HF_TOKEN)
  - Device selection (auto-detect MPS/CUDA/CPU)
  - Server settings (HOST, PORT)
  - Performance tuning (COLD_PATH_PARALLEL_WORKERS, etc.)

CONTRACTS:
- Expose a global `settings` instance
- Provide device auto-detection method
- Be compatible with scripts/config.py but extended for service needs
- Support environment variable overrides
- Validate HF_TOKEN is present if diarization is enabled

REMOVE:
- Old transformer-specific settings (USE_FLASH_ATTENTION, etc.)
- Old pipeline settings (FINAL_BEAM_SIZE, LIVE_CHUNK_LENGTH_S, etc.)
"""

"""
Audio preprocessing utilities.

SHOULD:
- Provide audio file validation and preprocessing
- Handle format conversions if needed (though MLX accepts most formats)
- Validate audio files before transcription
- Extract audio metadata (duration, sample rate, channels)

CONTRACTS:
- Functions:
  - validate_audio_file(file_path: str) -> bool
  - get_audio_info(file_path: str) -> dict  # Returns duration, sample_rate, channels
  - preprocess_audio(file_path: str) -> str  # Returns path (may be same or converted)

OPTIONAL:
- This module might not be needed if we rely on soundfile/ffmpeg directly
- Consider if cold_path_pipeline_v2.py already handles everything
- May only need validation functions for API endpoints

NOTE:
- Review if this is actually needed or if pipeline handles it all
- Keep minimal - don't duplicate pipeline functionality
"""

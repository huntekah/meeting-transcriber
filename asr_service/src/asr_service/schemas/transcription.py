"""
Pydantic schemas for API request/response models.

SHOULD:
- Define response models for transcription endpoints
- Support both simple transcription and diarized transcription with speakers
- Include models for:
  - TranscriptionResponse (final/cold path results)
  - TranscriptionSegment (with start, end, text, optional speaker)
  - PartialTranscription (live streaming updates)
  - HealthResponse (service health check)
  - ErrorResponse (error handling)

CONTRACTS:
- All responses must be JSON-serializable Pydantic models
- Support speaker diarization fields (speaker: str | None)
- Include timestamps (start, end as floats in seconds)
- Include metadata (duration, language, processing_time)
- Compatible with FastAPI automatic validation

EXAMPLE SEGMENT:
{
    "start": 0.8,
    "end": 5.1,
    "text": "Hello world",
    "speaker": "SPEAKER_00"  # Optional, only if diarization enabled
}

EXAMPLE RESPONSE:
{
    "segments": [...],
    "text": "Full transcript",
    "duration": 60.0,
    "language": "en",
    "processing_time": 12.5
}
"""

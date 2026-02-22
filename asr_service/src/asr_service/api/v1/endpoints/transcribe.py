"""
Transcription endpoints.

SHOULD:
- Provide POST /transcribe endpoint for file uploads
- Support both simple transcription and speaker diarization
- Handle file uploads with proper validation
- Return structured transcription with segments and timestamps

CONTRACTS:
- Endpoint: POST /api/v1/transcribe
- Request:
  - file: UploadFile (audio file)
  - use_diarization: bool = True (query param, optional)
  - language: str = "en" (query param, optional)
- Response: TranscriptionResponse schema

BEHAVIOR:
1. Validate uploaded file (mime type, size)
2. Save to temporary file
3. Call engine.transcribe_file(temp_path, use_diarization, language)
4. Map result to TranscriptionResponse schema
5. Clean up temporary file
6. Return response with processing_time metadata

EXAMPLE REQUEST:
POST /api/v1/transcribe?use_diarization=true&language=en
Content-Type: multipart/form-data
file: audio.mp3

EXAMPLE RESPONSE:
{
    "segments": [
        {
            "start": 0.8,
            "end": 5.1,
            "text": "Hello world",
            "speaker": "SPEAKER_00"
        }
    ],
    "text": "Hello world",
    "duration": 60.0,
    "language": "en",
    "processing_time": 12.5
}

ERROR HANDLING:
- Invalid file type -> 400 Bad Request
- File too large -> 413 Payload Too Large
- Transcription error -> 500 Internal Server Error
- All errors should return ErrorResponse schema

FUTURE (OPTIONAL):
- WebSocket endpoint for live streaming
- POST /api/v1/transcribe_live (websocket)
"""

"""
FastAPI dependency injection.

SHOULD:
- Provide dependency injection for shared resources
- Return singleton ASREngine instance
- Handle lazy initialization of models

CONTRACTS:
- Functions:
  - get_asr_engine() -> ASREngine  # Returns singleton, loads model if needed
  - get_audio_processor() -> AudioProcessor | None  # Optional, may not be needed

BEHAVIOR:
- get_asr_engine() should:
  - Return the singleton ASREngine instance
  - Call engine.load_model() if not already loaded
  - Be used as Depends() in FastAPI endpoints
  - Handle errors gracefully (log and raise HTTPException)

EXAMPLE USAGE:
@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    engine: ASREngine = Depends(get_asr_engine)
):
    result = engine.transcribe_file(file_path)
    return result
"""

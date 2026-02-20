import time
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, WebSocket, WebSocketDisconnect
from asr_service.api.deps import get_asr_engine, get_audio_processor
from asr_service.schemas.transcription import TranscriptionResponse, PartialTranscription, ErrorResponse
from asr_service.services.model_loader import ASREngine
from asr_service.services.audio_processor import AudioProcessor
from asr_service.utils.file_ops import create_temp_file, validate_mime_type
from asr_service.core.logging import logger

router = APIRouter()


@router.post("/transcribe_final", response_model=TranscriptionResponse)
async def transcribe_final(
    file: UploadFile = File(...),
    engine: ASREngine = Depends(get_asr_engine),
    audio_processor: AudioProcessor = Depends(get_audio_processor),
) -> TranscriptionResponse:
    """
    Transcribe uploaded audio/video file with high precision.

    Uses the final (non-chunked) pipeline with beam search for maximum accuracy.
    """
    start_time = time.time()

    # Validate MIME type
    if not validate_mime_type(file.content_type):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Expected audio or video file."
        )

    # Save uploaded file to temporary location
    file_suffix = Path(file.filename).suffix or ".tmp"

    try:
        with create_temp_file(suffix=file_suffix) as temp_file:
            # Write uploaded content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            temp_path = Path(temp_file.name)

            # Validate audio file
            if not audio_processor.validate_audio_file(temp_path):
                raise HTTPException(
                    status_code=422,
                    detail="Invalid audio file"
                )

            # Perform transcription
            logger.info(f"Transcribing file: {file.filename}")
            result = engine.transcribe_final(str(temp_path))

            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s")

            return TranscriptionResponse(
                text=result["text"],
                processing_time=processing_time,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@router.websocket("/ws/live_transcribe")
async def live_transcribe(
    websocket: WebSocket,
) -> None:
    """
    WebSocket endpoint for live audio transcription.

    Accepts raw PCM audio bytes (Float32, 16kHz) and streams back
    partial transcription results.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    engine = ASREngine()
    audio_processor = AudioProcessor()

    accumulated_audio = b""

    try:
        while True:
            # Receive audio data
            data = await websocket.receive()

            if "bytes" in data:
                # Accumulate audio bytes
                accumulated_audio += data["bytes"]

                # Process in chunks when we have enough data
                # (e.g., 1 second = 16000 samples * 4 bytes = 64KB)
                min_chunk_size = 16000 * 4  # 1 second of Float32 PCM at 16kHz

                if len(accumulated_audio) >= min_chunk_size:
                    try:
                        # Save accumulated audio to temp file for processing
                        with create_temp_file(suffix=".wav") as temp_file:
                            # Convert bytes to numpy array
                            audio_array = audio_processor.convert_bytes_to_audio(accumulated_audio)

                            # Save as WAV file (librosa can read this)
                            import soundfile as sf
                            temp_path = Path(temp_file.name)
                            sf.write(str(temp_path), audio_array, 16000)

                            # Transcribe using live pipeline
                            result = engine.transcribe_live(str(temp_path))

                            # Send partial result
                            response = PartialTranscription(
                                partial=result["text"],
                                is_final=False,
                            )
                            await websocket.send_text(response.model_dump_json())

                    except Exception as e:
                        logger.error(f"Live transcription error: {e}")
                        error_response = {"error": str(e), "partial": "", "is_final": False}
                        await websocket.send_text(json.dumps(error_response))

            elif "text" in data:
                # Handle control messages
                message = json.loads(data["text"])

                if message.get("action") == "finalize":
                    # Send final transcription
                    if accumulated_audio:
                        try:
                            with create_temp_file(suffix=".wav") as temp_file:
                                audio_array = audio_processor.convert_bytes_to_audio(accumulated_audio)

                                import soundfile as sf
                                temp_path = Path(temp_file.name)
                                sf.write(str(temp_path), audio_array, 16000)

                                result = engine.transcribe_final(str(temp_path))

                                response = PartialTranscription(
                                    partial=result["text"],
                                    is_final=True,
                                )
                                await websocket.send_text(response.model_dump_json())

                        except Exception as e:
                            logger.error(f"Final transcription error: {e}")
                            error_response = {"error": str(e), "partial": "", "is_final": True}
                            await websocket.send_text(json.dumps(error_response))

                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            error_response = {"error": str(e)}
            await websocket.send_text(json.dumps(error_response))
        except:
            pass
    finally:
        logger.info("WebSocket connection closed")

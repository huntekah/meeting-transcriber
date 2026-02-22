"""
Session management endpoints.

Provides endpoints for creating, managing, and querying transcription sessions.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any

from ....core.logging import logger
from ....core.exceptions import SessionNotFoundError
from ....schemas.transcription import SessionConfig, SessionResponse, TranscriptDocument
from ....services.session_manager import SessionManager
from ....services.model_manager import ModelManager
from ...deps import get_session_manager, get_model_manager

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse, tags=["sessions"], status_code=status.HTTP_201_CREATED)
async def create_session(
    config: SessionConfig,
    session_manager: SessionManager = Depends(get_session_manager),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Create new multi-source recording session.

    Starts audio capture and live transcription from specified devices.

    Request body:
        {
            "sources": [
                {"device_index": 0, "device_name": "Microphone"},
                {"device_index": 2, "device_name": "System Audio"}
            ],
            "output_dir": "/path/to/output"  // Optional
        }

    Returns:
        {
            "session_id": "uuid",
            "state": "recording",
            "websocket_url": "/api/v1/ws/{session_id}"
        }
    """
    try:
        # Create session
        session = await session_manager.create_session(
            sources=config.sources,
            model_manager=model_manager,
            output_dir=config.output_dir,
        )

        # Start recording
        session.start_recording()

        return SessionResponse(
            session_id=session.session_id,
            state=session.state.value,
            websocket_url=f"/api/v1/ws/{session.session_id}",
        )

    except Exception as e:
        logger.error(f"Session creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/sessions/{session_id}/stop", tags=["sessions"])
async def stop_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    Stop recording session and trigger post-processing.

    Gracefully stops all audio pipelines, saves audio, and runs cold path transcription.

    Returns:
        {
            "session_id": "uuid",
            "state": "processing"
        }
    """
    try:
        session = await session_manager.get_session(session_id)
        await session.stop_recording()

        return {"session_id": session.session_id, "state": session.state.value}

    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Session stop failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions/{session_id}", response_model=TranscriptDocument, tags=["sessions"])
async def get_session_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    Get session status and transcript.

    Returns complete session information including live transcript,
    final cold path transcript (if available), and audio file path.

    Returns:
        {
            "session_id": "uuid",
            "started_at": "2024-01-01T12:00:00Z",
            "ended_at": "2024-01-01T12:30:00Z",
            "duration_seconds": 1800.0,
            "state": "completed",
            "utterances": [...],
            "audio_file_path": "/path/to/audio.wav"
        }
    """
    try:
        session = await session_manager.get_session(session_id)
        return session.get_document()

    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/sessions", tags=["sessions"])
async def list_sessions(
    session_manager: SessionManager = Depends(get_session_manager),
) -> Dict[str, List[str]]:
    """
    List all active session IDs.

    Returns:
        {
            "sessions": ["uuid1", "uuid2", ...]
        }
    """
    sessions = await session_manager.list_sessions()
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}", tags=["sessions"], status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    Delete session from registry.

    Note: Does not stop the session if it's still running.
    Use POST /sessions/{session_id}/stop first.
    """
    await session_manager.delete_session(session_id)


@router.get("/sessions/{session_id}/stats", tags=["sessions"])
async def get_session_stats(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Dict[str, Any]:
    """
    Get detailed session statistics.

    Returns statistics for all pipelines, merger, and overall session metrics.
    """
    try:
        session = await session_manager.get_session(session_id)
        return session.get_stats()

    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

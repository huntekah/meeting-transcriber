"""
WebSocket endpoints.

Provides WebSocket endpoint for real-time transcription updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from ....core.logging import logger
from ....services.session_manager import SessionManager
from ...deps import get_session_manager

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    WebSocket endpoint for real-time transcription updates.

    Protocol:
    1. Client connects to /ws/{session_id}
    2. Server sends current state
    3. Server sends backlog (last 20 utterances)
    4. Server streams live updates:
       - "utterance": New transcription from any source
       - "state_change": Session state transition
       - "final_transcript": Cold path results
       - "error": Error notification

    Message format:
        {
            "type": "utterance" | "state_change" | "final_transcript" | "error",
            "data" | "state" | "transcript" | "message": ...
        }
    """
    await websocket.accept()
    logger.info(f"WebSocket connection opened for session {session_id}")

    try:
        # Get session
        session = await session_manager.get_session_or_none(session_id)
        if not session:
            await websocket.send_json({
                "type": "error",
                "message": f"Session {session_id} not found",
                "code": "SESSION_NOT_FOUND",
            })
            await websocket.close(code=1008)  # Policy violation
            return

        # Add client to session (sends backlog automatically)
        await session.add_websocket_client(websocket)

        # Keep connection alive and handle client messages
        while True:
            # We don't expect messages from client, just keep alive
            # Could implement ping/pong here if needed
            data = await websocket.receive_text()

            # Echo back (for debugging)
            logger.debug(f"WebSocket received from client: {data}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
    finally:
        # Remove client from session
        if session:
            await session.remove_websocket_client(websocket)

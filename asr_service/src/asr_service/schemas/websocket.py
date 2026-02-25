"""
WebSocket message schemas.

Pydantic models for WebSocket communication between server and clients.
"""

from typing import Literal
from pydantic import BaseModel, Field
from .transcription import Utterance, SessionState, ColdTranscriptResult


class WSUtteranceMessage(BaseModel):
    """
    WebSocket message for a new utterance.

    Sent when a new utterance is transcribed from any source.
    """

    type: Literal["utterance"] = Field("utterance", description="Message type")
    data: Utterance = Field(..., description="Utterance data")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "utterance",
                "data": {
                    "source_id": 0,
                    "start_time": 1700000000.5,
                    "end_time": 1700000005.2,
                    "text": "Hello world",
                    "confidence": 0.95,
                    "is_final": True,
                    "overlaps_with": [],
                },
            }
        }


class WSStateChangeMessage(BaseModel):
    """
    WebSocket message for session state changes.

    Sent when session transitions between states.
    """

    type: Literal["state_change"] = Field("state_change", description="Message type")
    state: SessionState = Field(..., description="New session state")

    class Config:
        json_schema_extra = {"example": {"type": "state_change", "state": "processing"}}


class WSErrorMessage(BaseModel):
    """
    WebSocket message for errors.

    Sent when an error occurs during the session.
    """

    type: Literal["error"] = Field("error", description="Message type")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "error",
                "message": "Transcription failed",
                "code": "TRANSCRIPTION_ERROR",
            }
        }


class WSFinalTranscriptMessage(BaseModel):
    """
    WebSocket message for final cold path transcript.

    Sent when cold path post-processing is complete.
    """

    type: Literal["final_transcript"] = Field(
        "final_transcript", description="Message type"
    )
    transcript: ColdTranscriptResult = Field(..., description="Final transcript")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "final_transcript",
                "transcript": {
                    "segments": [],
                    "duration": 60.0,
                    "language": "en",
                },
            }
        }


class WSMentionAlertMessage(BaseModel):
    """
    WebSocket message for @-mention alerts.

    Sent when a keyword/name is detected in transcription.
    """

    type: Literal["mention_alert"] = Field("mention_alert", description="Message type")
    text: str = Field(..., description="Utterance text containing mention")
    source_id: int = Field(..., description="Source that mentioned")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "mention_alert",
                "text": "Hey @John, can you help with this?",
                "source_id": 1,
            }
        }


# Union type for all possible WebSocket messages
WSMessage = (
    WSUtteranceMessage
    | WSStateChangeMessage
    | WSErrorMessage
    | WSFinalTranscriptMessage
    | WSMentionAlertMessage
)

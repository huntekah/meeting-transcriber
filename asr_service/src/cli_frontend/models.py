"""
Pydantic models mirroring backend schemas.

These models ensure type safety when communicating with the ASR service API.
"""

from pydantic import BaseModel
from typing import List, Optional, Literal, Union
from datetime import datetime


class AudioDevice(BaseModel):
    """Audio device information."""

    device_index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool


class SourceConfig(BaseModel):
    """Source configuration for session creation."""

    device_index: int
    device_name: str
    device_channels: int = 1  # Default to mono


class Utterance(BaseModel):
    """Live transcription utterance."""

    source_id: int
    start_time: float  # Unix timestamp
    end_time: float  # Unix timestamp
    text: str
    confidence: float = 1.0
    is_final: bool = True
    overlaps_with: List[int] = []  # Source IDs that overlap


class SessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    state: str
    websocket_url: str


# WebSocket message types
class WSUtteranceMessage(BaseModel):
    """WebSocket message for new utterance."""

    type: Literal["utterance"]
    data: Utterance


class WSStateChangeMessage(BaseModel):
    """WebSocket message for state change."""

    type: Literal["state_change"]
    state: str


class WSFinalTranscriptMessage(BaseModel):
    """WebSocket message for final transcript."""

    type: Literal["final_transcript"]
    transcript: dict


class WSErrorMessage(BaseModel):
    """WebSocket message for errors."""

    type: Literal["error"]
    message: str
    code: str


# Union type for all WebSocket messages
WSMessage = Union[
    WSUtteranceMessage,
    WSStateChangeMessage,
    WSFinalTranscriptMessage,
    WSErrorMessage,
]

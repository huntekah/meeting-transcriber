"""
Pydantic models mirroring backend schemas.

These models ensure type safety when communicating with the ASR service API.
"""

from pydantic import BaseModel
from typing import List, Literal, Union, Optional


class AudioDevice(BaseModel):
    """Audio device information."""

    device_index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool
    source_type: str = "sounddevice"  # "sounddevice" or "screencapture"


class SourceConfig(BaseModel):
    """Source configuration for session creation."""

    device_index: int
    device_name: str
    device_channels: int = 1  # Default to mono
    source_type: str = "sounddevice"  # "sounddevice" or "screencapture"


class Utterance(BaseModel):
    """Live transcription utterance."""

    source_id: int
    source_label: str  # Human-readable device name (e.g. "MacBook Pro Microphone")
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


# --- Insight models ---


class SkillInfo(BaseModel):
    """Public metadata for a skill, returned by GET /skills on the LLM Intelligence service."""

    name: str
    description: str
    display: str  # emoji + label, e.g. "🎭 Emotion Translate"


class InsightRequest(BaseModel):
    """Request to the LLM insights service."""

    skill_name: str
    transcript: str
    model: Optional[str] = None  # optional model override sent by the frontend


class InsightResponse(BaseModel):
    """Response from the LLM insights service."""

    markdown: str
    skill_name: str

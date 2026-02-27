"""
Pydantic models mirroring backend schemas.

These models ensure type safety when communicating with the ASR service API.
"""

from enum import Enum
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


class InsightType(str, Enum):
    """Type of LLM insight to generate."""

    EMOTION_TRANSLATE = "emotion_translate"
    CODE = "code"
    KNOWLEDGE = "knowledge"
    PANIC_MODE = "panic_mode"


class InsightRequest(BaseModel):
    """Request to the LLM insights service."""

    transcript: str
    insight_type: InsightType
    window_minutes: Optional[float] = None  # informational; slicing done by frontend


class InsightResponse(BaseModel):
    """Response from the LLM insights service."""

    markdown: str
    insight_type: InsightType
    token_count: Optional[int] = None

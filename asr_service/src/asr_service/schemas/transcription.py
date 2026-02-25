"""
Transcription schemas.

Pydantic models for session management, utterances, and transcription results.
"""

from enum import Enum
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class SessionState(str, Enum):
    """Session lifecycle states."""

    INITIALIZING = "initializing"  # Models loading, pipelines creating
    RECORDING = "recording"  # Live transcription active
    STOPPING = "stopping"  # Graceful shutdown in progress
    PROCESSING = "processing"  # Cold path transcription running
    COMPLETED = "completed"  # Final transcript ready
    FAILED = "failed"  # Error occurred


class Utterance(BaseModel):
    """
    Single utterance from one audio source.

    Attributes:
        source_id: Index of the source that produced this utterance
        start_time: Unix timestamp when utterance started
        end_time: Unix timestamp when utterance ended
        text: Transcribed text
        confidence: Confidence score (0.0-1.0)
        is_final: Whether this is finalized (vs provisional)
        overlaps_with: List of source_ids that overlap with this utterance
    """

    source_id: int = Field(..., description="Source index")
    start_time: float = Field(..., description="Start time (Unix timestamp)")
    end_time: float = Field(..., description="End time (Unix timestamp)")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    is_final: bool = Field(True, description="Is finalized")
    overlaps_with: List[int] = Field(
        default_factory=list, description="Overlapping source IDs"
    )

    # Allow mutation for overlap marking
    class Config:
        frozen = False
        json_schema_extra = {
            "example": {
                "source_id": 0,
                "start_time": 1700000000.5,
                "end_time": 1700000005.2,
                "text": "Hello, how are you?",
                "confidence": 0.95,
                "is_final": True,
                "overlaps_with": [1],
            }
        }


class SourceConfig(BaseModel):
    """
    Configuration for a single audio source.

    Attributes:
        device_index: Device index to capture from
        device_name: Human-readable device name/label
        device_channels: Number of input channels (1=mono, 2=stereo, etc.)
        source_type: Type of audio source ("sounddevice" or "screencapture")
    """

    device_index: int = Field(..., description="Device index")
    device_name: str = Field(..., description="Device name/label")
    device_channels: int = Field(default=1, description="Number of input channels")
    source_type: str = Field(
        default="sounddevice",
        description="Audio source type (sounddevice or screencapture)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "device_index": 0,
                "device_name": "Microphone",
                "device_channels": 1,
                "source_type": "sounddevice",
            }
        }


class SessionConfig(BaseModel):
    """
    Configuration for creating a new session.

    Attributes:
        sources: List of audio sources to capture
        output_dir: Optional output directory for saved files
    """

    sources: List[SourceConfig] = Field(..., description="Audio sources")
    output_dir: Optional[str] = Field(None, description="Output directory path")

    class Config:
        json_schema_extra = {
            "example": {
                "sources": [
                    {"device_index": 0, "device_name": "Microphone"},
                    {"device_index": 2, "device_name": "System Audio"},
                ],
                "output_dir": "/path/to/output",
            }
        }


class SessionResponse(BaseModel):
    """
    Response after creating a session.

    Attributes:
        session_id: Unique session identifier
        state: str = Current session state
        websocket_url: WebSocket URL for real-time updates
    """

    session_id: str = Field(..., description="Session ID")
    state: str = Field(..., description="Current state")
    websocket_url: str = Field(..., description="WebSocket URL")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "state": "recording",
                "websocket_url": "/api/v1/ws/abc123",
            }
        }


class TranscriptDocument(BaseModel):
    """
    Complete transcript document for a session.

    Attributes:
        session_id: Session identifier
        started_at: When session started
        ended_at: When session ended
        duration_seconds: Total duration in seconds
        state: Final session state
        utterances: All utterances in chronological order
        audio_file_path: Path to saved audio file
    """

    session_id: str = Field(..., description="Session ID")
    started_at: datetime = Field(..., description="Start time")
    ended_at: Optional[datetime] = Field(None, description="End time")
    duration_seconds: float = Field(..., description="Duration in seconds")
    state: SessionState = Field(..., description="Session state")
    utterances: List[Utterance] = Field(..., description="All utterances")
    audio_file_path: Optional[str] = Field(None, description="Audio file path")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "started_at": "2024-01-01T12:00:00Z",
                "ended_at": "2024-01-01T12:30:00Z",
                "duration_seconds": 1800.0,
                "state": "completed",
                "utterances": [],
                "audio_file_path": "/path/to/audio.wav",
            }
        }


class ColdTranscriptSegment(BaseModel):
    """
    Segment from cold path transcription (with speaker diarization).

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        text: Transcribed text
        speaker: Speaker label (e.g., "SPEAKER_00")
        words: Optional word-level timestamps
    """

    start: float = Field(..., description="Start time (seconds)")
    end: float = Field(..., description="End time (seconds)")
    text: str = Field(..., description="Transcribed text")
    speaker: Optional[str] = Field(None, description="Speaker label")
    words: Optional[List[Dict[str, Any]]] = Field(
        None, description="Word-level timestamps"
    )


class ColdTranscriptResult(BaseModel):
    """
    Result from cold path post-processing.

    Attributes:
        segments: Transcribed segments with speaker labels
        duration: Total audio duration
        language: Detected language
    """

    segments: List[ColdTranscriptSegment] = Field(..., description="Segments")
    duration: float = Field(..., description="Duration in seconds")
    language: str = Field(..., description="Language code")

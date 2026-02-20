from pydantic import BaseModel
from typing import Optional


class TranscriptionResponse(BaseModel):
    """Response schema for final transcription."""
    text: str
    processing_time: float


class PartialTranscription(BaseModel):
    """Schema for partial/live transcription updates."""
    partial: str
    is_final: bool


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    device: str
    model_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None

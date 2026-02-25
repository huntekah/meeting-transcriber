"""
Custom exceptions for ASR service.

Provides domain-specific exception classes for error handling.
"""


class ASRServiceError(Exception):
    """Base exception for all ASR service errors."""

    pass


class SessionNotFoundError(ASRServiceError):
    """Raised when a requested session does not exist."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class DeviceNotFoundError(ASRServiceError):
    """Raised when a requested audio device is not available."""

    def __init__(self, device_id: int | str):
        self.device_id = device_id
        super().__init__(f"Audio device not found: {device_id}")


class TranscriptionError(ASRServiceError):
    """Raised when transcription fails."""

    def __init__(self, message: str, source_id: int | None = None):
        self.source_id = source_id
        if source_id is not None:
            super().__init__(f"Transcription error (source {source_id}): {message}")
        else:
            super().__init__(f"Transcription error: {message}")


class ModelLoadingError(ASRServiceError):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class SessionStateError(ASRServiceError):
    """Raised when an operation is invalid for the current session state."""

    def __init__(self, session_id: str, current_state: str, required_state: str):
        self.session_id = session_id
        self.current_state = current_state
        self.required_state = required_state
        super().__init__(
            f"Session {session_id} is in state '{current_state}', "
            f"but operation requires '{required_state}'"
        )


class AudioCaptureError(ASRServiceError):
    """Raised when audio capture fails."""

    def __init__(self, device_id: int | str, reason: str):
        self.device_id = device_id
        self.reason = reason
        super().__init__(f"Audio capture failed for device {device_id}: {reason}")

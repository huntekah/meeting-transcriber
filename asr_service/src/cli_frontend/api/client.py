"""
HTTP client for ASR service REST API.

Provides async methods for interacting with session management endpoints.
"""

import httpx
from typing import List, Optional, Dict, Any
from cli_frontend.models import AudioDevice, SourceConfig
from cli_frontend.config import CLISettings


class ASRClient:
    """Async HTTP client for ASR service API."""

    def __init__(self, settings: CLISettings):
        self.base_url = settings.api_base_url
        # 60s timeout to accommodate cold path processing (can take 25-30s)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def get_devices(self) -> List[AudioDevice]:
        """
        Get list of available audio input devices.

        GET /api/v1/devices

        Returns:
            List of AudioDevice objects
        """
        response = await self.client.get("/api/v1/devices")
        response.raise_for_status()
        return [AudioDevice(**d) for d in response.json()]

    async def create_session(
        self, sources: List[SourceConfig], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create new recording session and start recording.

        POST /api/v1/sessions

        Args:
            sources: List of audio sources to record from
            output_dir: Optional output directory for transcripts

        Returns:
            Session response with session_id, state, websocket_url
        """
        payload: Dict[str, Any] = {
            "sources": [s.model_dump() for s in sources],
        }
        if output_dir:
            payload["output_dir"] = output_dir

        response = await self.client.post("/api/v1/sessions", json=payload)
        response.raise_for_status()
        return response.json()

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """
        Stop recording session and trigger post-processing.

        POST /api/v1/sessions/{session_id}/stop

        Args:
            session_id: Session ID to stop

        Returns:
            Session response with updated state
        """
        response = await self.client.post(f"/api/v1/sessions/{session_id}/stop")
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get session status and transcript.

        GET /api/v1/sessions/{session_id}

        Args:
            session_id: Session ID to query

        Returns:
            Complete session document with transcript
        """
        response = await self.client.get(f"/api/v1/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

    async def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        GET /api/v1/sessions

        Returns:
            List of session IDs
        """
        response = await self.client.get("/api/v1/sessions")
        response.raise_for_status()
        return response.json().get("sessions", [])

    async def close(self):
        """Close HTTP client connection."""
        await self.client.aclose()

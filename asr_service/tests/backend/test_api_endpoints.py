"""
Tests for API endpoints.

Tests the FastAPI REST endpoints without requiring actual hardware or models.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


class TestRootEndpoints:
    """Test root and health endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test GET / returns service info."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Multi-Source ASR Service"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: AsyncClient):
        """Test GET /health returns health status."""
        with patch("asr_service.services.model_manager.ModelManager") as mock_manager:
            # Mock ModelManager
            mock_instance = MagicMock()
            mock_instance.is_loaded.return_value = False
            mock_manager.return_value = mock_instance

            response = await async_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "models_loaded" in data
            assert "device" in data


class TestDevicesEndpoint:
    """Test device discovery endpoint."""

    @pytest.mark.asyncio
    async def test_get_devices(self, async_client: AsyncClient):
        """Test GET /api/v1/devices returns device list."""
        with patch("asr_service.services.audio_devices.sd.query_devices") as mock_query:
            # Mock sounddevice response
            mock_query.return_value = [
                {
                    "name": "Test Microphone",
                    "index": 0,
                    "max_input_channels": 1,
                    "default_samplerate": 48000,
                },
                {
                    "name": "Test System Audio",
                    "index": 1,
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                },
            ]

            with patch("asr_service.services.audio_devices.sd.default.device", (0, -1)):
                # Disable ScreenCaptureKit for this test (different machines may have it or not)
                with patch("asr_service.services.audio_devices._is_screencapture_binary_available", return_value=False):
                    response = await async_client.get("/api/v1/devices")

                    assert response.status_code == 200
                    devices = response.json()

                    # Should have at least the 2 mocked devices
                    assert len(devices) >= 2

                    # Verify the mocked devices are present (by name and properties)
                    device_names = [d["name"] for d in devices]
                    assert "Test Microphone" in device_names
                    assert "Test System Audio" in device_names

                    # Verify default device marking
                    mic = next((d for d in devices if d["name"] == "Test Microphone"), None)
                    assert mic is not None
                    assert mic["device_index"] == 0
                    assert mic["channels"] == 1
                    assert mic["is_default"] is True


class TestSessionEndpoints:
    """Test session management endpoints."""

    @pytest.mark.asyncio
    async def test_create_session(self, async_client: AsyncClient):
        """Test POST /api/v1/sessions creates a session."""
        with patch("asr_service.services.session_manager.SessionManager.create_session") as mock_create:
            # Mock session creation
            mock_session = MagicMock()
            mock_session.session_id = "test-session-123"
            mock_session.state.value = "recording"
            mock_create.return_value = mock_session

            payload = {
                "sources": [
                    {"device_index": 0, "device_name": "Test Mic"},
                ],
                "output_dir": "/tmp/test",
            }

            response = await async_client.post("/api/v1/sessions", json=payload)

            assert response.status_code == 201
            data = response.json()
            assert data["session_id"] == "test-session-123"
            assert data["state"] == "recording"
            assert "websocket_url" in data

    @pytest.mark.asyncio
    async def test_list_sessions(self, async_client: AsyncClient):
        """Test GET /api/v1/sessions lists active sessions."""
        with patch("asr_service.services.session_manager.SessionManager.list_sessions") as mock_list:
            mock_list.return_value = ["session-1", "session-2"]

            response = await async_client.get("/api/v1/sessions")

            assert response.status_code == 200
            data = response.json()
            assert data["sessions"] == ["session-1", "session-2"]

    @pytest.mark.asyncio
    async def test_get_session(self, async_client: AsyncClient):
        """Test GET /api/v1/sessions/{id} returns session details."""
        # Create a real session first
        from asr_service.services.session_manager import SessionManager
        from asr_service.schemas.transcription import SourceConfig

        # Clear singleton and create manager
        if hasattr(SessionManager, '_instance'):
            SessionManager._instance = None

        manager = SessionManager()

        # Mock model manager
        mock_model_manager = MagicMock()
        mock_model_manager.load_models = AsyncMock()
        mock_model_manager.vad_model = MagicMock()
        mock_model_manager.whisper_model_name = "test"

        # Create a test session
        sources = [SourceConfig(device_index=0, device_name="Test")]
        session = await manager.create_session(sources, mock_model_manager)
        session_id = session.session_id

        # Get the session
        response = await async_client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_stop_session(self, async_client: AsyncClient):
        """Test POST /api/v1/sessions/{id}/stop stops recording."""
        with patch("asr_service.services.session_manager.SessionManager.get_session") as mock_get:
            # Mock session
            mock_session = AsyncMock()
            mock_session.session_id = "test-123"
            mock_session.state.value = "stopping"
            mock_get.return_value = mock_session

            response = await async_client.post("/api/v1/sessions/test-123/stop")

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-123"
            mock_session.stop_recording.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_session(self, async_client: AsyncClient):
        """Test DELETE /api/v1/sessions/{id} removes session."""
        with patch("asr_service.services.session_manager.SessionManager.delete_session") as mock_delete:
            mock_delete.return_value = None

            response = await async_client.delete("/api/v1/sessions/test-123")

            assert response.status_code == 204
            mock_delete.assert_called_once_with("test-123")

    @pytest.mark.asyncio
    async def test_get_session_stats(self, async_client: AsyncClient):
        """Test GET /api/v1/sessions/{id}/stats returns statistics."""
        with patch("asr_service.services.session_manager.SessionManager.get_session") as mock_get:
            # Mock session with stats
            mock_session = MagicMock()
            mock_session.get_stats.return_value = {
                "session_id": "test-123",
                "state": "recording",
                "source_count": 2,
            }
            mock_get.return_value = mock_session

            response = await async_client.get("/api/v1/sessions/test-123/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-123"
            assert data["source_count"] == 2


class TestErrorHandling:
    """Test error handling in API."""

    @pytest.mark.asyncio
    async def test_session_not_found(self, async_client: AsyncClient):
        """Test 404 when session doesn't exist."""
        from asr_service.core.exceptions import SessionNotFoundError

        with patch("asr_service.services.session_manager.SessionManager.get_session") as mock_get:
            mock_get.side_effect = SessionNotFoundError("Session not found")

            response = await async_client.get("/api/v1/sessions/nonexistent")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_session_payload(self, async_client: AsyncClient):
        """Test 422 validation error for invalid payload."""
        payload = {
            # Missing required 'sources' field
        }

        response = await async_client.post("/api/v1/sessions", json=payload)

        # Should fail validation (422 Unprocessable Entity)
        assert response.status_code == 422

"""
Tests for CLI frontend API client.

Tests HTTP client and WebSocket client for ASR service communication.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from cli_frontend.api.client import ASRClient
from cli_frontend.api.websocket import WSClient
from cli_frontend.config import CLISettings
from cli_frontend.models import SourceConfig, AudioDevice


class TestASRClient:
    """Test ASRClient HTTP client."""

    @pytest.mark.asyncio
    async def test_get_devices(self):
        """Test getting device list from API."""
        settings = CLISettings()
        client = ASRClient(settings)

        with patch.object(client.client, "get") as mock_get:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "device_index": 0,
                    "name": "Test Mic",
                    "channels": 1,
                    "sample_rate": 48000,
                    "is_default": True,
                }
            ]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            devices = await client.get_devices()

            assert len(devices) == 1
            assert devices[0].name == "Test Mic"
            assert isinstance(devices[0], AudioDevice)

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session via API."""
        settings = CLISettings()
        client = ASRClient(settings)

        with patch.object(client.client, "post") as mock_post:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "session_id": "test-123",
                "state": "recording",
                "websocket_url": "/api/v1/ws/test-123",
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            sources = [SourceConfig(device_index=0, device_name="Test")]
            result = await client.create_session(sources, output_dir="/tmp/test")

            assert result["session_id"] == "test-123"
            assert result["state"] == "recording"
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_session(self):
        """Test stopping a session via API."""
        settings = CLISettings()
        client = ASRClient(settings)

        with patch.object(client.client, "post") as mock_post:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "session_id": "test-123",
                "state": "stopping",
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = await client.stop_session("test-123")

            assert result["state"] == "stopping"
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test getting session details via API."""
        settings = CLISettings()
        client = ASRClient(settings)

        with patch.object(client.client, "get") as mock_get:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "session_id": "test-123",
                "state": "completed",
                "utterances": [],
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await client.get_session("test-123")

            assert result["session_id"] == "test-123"
            assert result["state"] == "completed"

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing sessions via API."""
        settings = CLISettings()
        client = ASRClient(settings)

        with patch.object(client.client, "get") as mock_get:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "sessions": ["session-1", "session-2"]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await client.list_sessions()

            assert result == ["session-1", "session-2"]


class TestWSClient:
    """Test WebSocket client."""

    def test_initial_state(self):
        """Test WebSocket client initial state."""
        client = WSClient("ws://localhost:8000/ws/test")

        assert client.ws_url == "ws://localhost:8000/ws/test"
        assert client.websocket is None
        assert client._running is False

    @pytest.mark.asyncio
    async def test_connect_and_receive_messages(self):
        """Test connecting and receiving messages."""
        client = WSClient("ws://test/ws")
        messages_received = []

        async def on_message(data):
            messages_received.append(data)

        with patch("cli_frontend.api.websocket.websockets.connect") as mock_connect:
            # Mock WebSocket
            from websockets.exceptions import ConnectionClosed

            mock_ws = AsyncMock()
            # After 2 messages, raise ConnectionClosed to exit loop
            mock_ws.recv.side_effect = [
                json.dumps({"type": "utterance", "data": {"text": "Hello"}}),
                json.dumps({"type": "state_change", "state": "recording"}),
                ConnectionClosed(None, None),
            ]

            mock_connect.return_value.__aenter__.return_value = mock_ws

            await client.connect(on_message)

            # Should have received 2 messages
            assert len(messages_received) == 2
            assert messages_received[0]["type"] == "utterance"
            assert messages_received[1]["type"] == "state_change"

    def test_disconnect(self):
        """Test disconnecting WebSocket."""
        client = WSClient("ws://test/ws")
        client._running = True

        client.disconnect()

        assert client._running is False

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message via WebSocket."""
        client = WSClient("ws://test/ws")

        # Mock websocket
        mock_ws = AsyncMock()
        client.websocket = mock_ws
        client._running = True

        await client.send({"type": "ping"})

        mock_ws.send.assert_called_once()
        call_args = mock_ws.send.call_args[0][0]
        assert json.loads(call_args)["type"] == "ping"

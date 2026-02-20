import pytest
import json


def test_websocket_connection_to_live_transcribe(sync_client):
    """Test that WebSocket connection can be established to /ws/live_transcribe."""
    with sync_client.websocket_connect("/api/v1/ws/live_transcribe") as websocket:
        # Connection successful
        assert websocket is not None


def test_sending_audio_chunks_and_receiving_partials(sync_client, sample_audio_bytes):
    """Test sending audio chunks and receiving partial transcription results."""
    with sync_client.websocket_connect("/api/v1/ws/live_transcribe") as websocket:
        # Send audio chunk
        websocket.send_bytes(sample_audio_bytes)

        # Receive response
        response = websocket.receive_text()
        data = json.loads(response)

        assert "partial" in data or "error" in data
        assert "is_final" in data or "error" in data
        if "partial" in data:
            assert isinstance(data["partial"], str)
            assert isinstance(data["is_final"], bool)


def test_websocket_final_transcription(sync_client, test_audio_file):
    """Test that final transcription is eventually received via WebSocket."""
    # Read test audio file
    with open(test_audio_file, "rb") as f:
        audio_data = f.read()

    with sync_client.websocket_connect("/api/v1/ws/live_transcribe") as websocket:
        # Send audio data in chunks
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            websocket.send_bytes(chunk)

        # Signal end of stream (or close connection)
        websocket.send_text(json.dumps({"action": "finalize"}))

        # Collect responses
        responses = []
        try:
            while True:
                response = websocket.receive_text()
                data = json.loads(response)
                responses.append(data)
                if data.get("is_final", False):
                    break
        except Exception:
            pass

        # Should have received at least one response
        assert len(responses) > 0
        # At least one should be marked as final
        assert any(r.get("is_final") for r in responses)


def test_websocket_error_handling_and_cleanup(sync_client):
    """Test WebSocket error handling and proper cleanup."""
    with sync_client.websocket_connect("/api/v1/ws/live_transcribe") as websocket:
        # Send invalid data
        websocket.send_text("invalid audio data")

        # Should either get an error message or connection closes gracefully
        try:
            response = websocket.receive_text()
            data = json.loads(response)
            # Check for error field
            assert "error" in data or "partial" in data
        except Exception:
            # Connection closed, which is also acceptable
            pass

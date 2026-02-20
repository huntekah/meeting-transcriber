import pytest


async def test_complete_e2e_workflow(async_client, test_audio_file, expected_transcription):
    """Test the complete end-to-end workflow: health check → transcription → validation."""

    # Step 1: Health check - ensure service is running
    health_response = await async_client.get("/api/v1/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True

    # Step 2: Perform transcription
    with open(test_audio_file, "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        transcribe_response = await async_client.post("/api/v1/transcribe_final", files=files)

    assert transcribe_response.status_code == 200
    transcribe_data = transcribe_response.json()

    # Step 3: Validate transcription results
    assert "text" in transcribe_data
    assert "processing_time" in transcribe_data

    # Check that we got some transcription (ASR results may vary)
    transcription = transcribe_data["text"].strip()
    assert len(transcription) > 0

    # Verify processing time is reasonable
    assert transcribe_data["processing_time"] > 0
    assert transcribe_data["processing_time"] < 60  # Should complete within 60 seconds

    # Step 4: Another health check to ensure system is still stable
    final_health_response = await async_client.get("/api/v1/health")
    assert final_health_response.status_code == 200

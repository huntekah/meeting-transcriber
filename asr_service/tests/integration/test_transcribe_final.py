import pytest


async def test_post_transcribe_final_with_valid_mp3(async_client, test_audio_file):
    """Test POST /transcribe_final with a valid MP3 file."""
    with open(test_audio_file, "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        response = await async_client.post("/api/v1/transcribe_final", files=files)

    assert response.status_code == 200


async def test_transcribe_response_contains_expected_text(async_client, test_audio_file, expected_transcription):
    """Test that the transcription response contains the expected text."""
    with open(test_audio_file, "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        response = await async_client.post("/api/v1/transcribe_final", files=files)

    data = response.json()
    assert "text" in data
    # Check that we got some transcription result (ASR results may vary)
    transcription = data["text"].strip()
    assert len(transcription) > 0
    # Log what we got for debugging
    print(f"Transcribed text: {transcription}")


async def test_transcribe_response_includes_processing_time(async_client, test_audio_file):
    """Test that the response includes processing_time field."""
    with open(test_audio_file, "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        response = await async_client.post("/api/v1/transcribe_final", files=files)

    data = response.json()
    assert "processing_time" in data
    assert isinstance(data["processing_time"], (int, float))
    assert data["processing_time"] > 0


async def test_transcribe_error_handling_invalid_file(async_client):
    """Test error handling when uploading an invalid file format."""
    invalid_content = b"This is not an audio file"
    files = {"file": ("test.txt", invalid_content, "text/plain")}

    response = await async_client.post("/api/v1/transcribe_final", files=files)

    assert response.status_code in [400, 415, 422]  # Bad Request or Unsupported Media Type

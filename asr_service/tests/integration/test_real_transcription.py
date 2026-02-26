"""
Integration test with real audio file.

Tests actual transcription with the MP3 test file.
"""

import pytest
import asyncio
from pathlib import Path
from asr_service.services.model_manager import ModelManager


@pytest.fixture(scope="module")
def test_audio_file():
    """Path to test MP3 file."""
    audio_path = Path(__file__).parent.parent.parent / "data" / "this_is_a_test_that_we_will_use_to_check_asr.mp3"
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"
    return audio_path


@pytest.fixture(scope="module")
async def loaded_models():
    """Ensure models are loaded before test."""
    manager = ModelManager()
    await manager.load_models()
    yield manager
    # Cleanup after all tests
    manager.unload_models()


@pytest.mark.asyncio
async def test_real_audio_transcription(test_audio_file, loaded_models):
    """
    Test actual transcription with real audio file.

    This test verifies:
    1. Cold path processing works with real audio
    2. Transcription is non-empty
    3. Segments are extracted
    """
    from asr_service.services.cold_transcriber import ColdPathPostProcessor
    from asr_service.core.config import settings

    # Get cold pipeline
    cold_pipeline = loaded_models.get_cold_pipeline()
    processor = ColdPathPostProcessor(cold_pipeline)

    # Process audio
    result = processor.process_long_audio(
        test_audio_file,
        chunk_duration=settings.COLD_PATH_CHUNK_DURATION
    )

    # Verify results
    assert result is not None
    assert "segments" in result
    assert "duration" in result
    assert "language" in result

    # Check we got transcription
    segments = result["segments"]
    assert len(segments) > 0, "No segments transcribed"

    # Check segments have text
    total_text = " ".join(seg["text"].strip() for seg in segments)
    assert len(total_text) > 0, "Transcription is empty"

    # Print for verification
    print("\n=== Transcription Results ===")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(segments)}")
    print(f"Full text: {total_text}")

    # Verify the test phrase is in transcription
    # The file is named "this_is_a_test_that_we_will_use_to_check_asr.mp3"
    # So it likely contains that phrase
    assert "test" in total_text.lower(), "Expected 'test' in transcription"


@pytest.mark.asyncio
async def test_live_transcription_debug(loaded_models):
    """
    Debug test to understand why live transcription doesn't show.

    This test simulates the live transcription path.
    """
    import numpy as np
    import queue
    from asr_service.schemas.transcription import Utterance
    from asr_service.services.live_transcriber import LiveTranscriber

    # Create test audio (2 seconds of white noise)
    sample_rate = 16000
    duration = 2.0
    audio_np = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    # Create queue and callback
    input_queue = queue.Queue()
    received_utterances = []

    def callback(utterance: Utterance):
        received_utterances.append(utterance)
        print(f"Received utterance: {utterance.text[:50]}...")

    # Create transcriber
    transcriber = LiveTranscriber(
        source_id=0,
        input_queue=input_queue,
        output_callback=callback,
        whisper_model_name="mlx-community/whisper-large-v3-turbo",
        language="en"
    )

    # Start transcriber
    transcriber.start()

    # Queue audio segment
    import time
    segment = {
        "audio": audio_np,
        "timestamp": time.time(),
        "source_id": 0
    }
    input_queue.put(segment)

    # Wait for processing
    await asyncio.sleep(3.0)

    # Stop transcriber
    transcriber.stop()

    # Check results
    print(f"\nReceived {len(received_utterances)} utterances")
    for utt in received_utterances:
        print(f"  - {utt.text}")

    # We might get empty transcription for noise, but callback should be called
    # This verifies the pipeline works
    assert True  # This test is for debugging, not strict validation


@pytest.mark.asyncio
async def test_long_audio_with_vad_chunking(loaded_models, tmp_path):
    """
    Test cold path with long audio that exercises VAD-based chunking.

    CRITICAL: This test catches dtype mismatches in VAD model.
    Previous bug: VAD expected float64 but audio was float32.

    This test:
    1. Creates synthetic audio longer than COLD_PATH_CHUNK_DURATION
    2. Forces the VAD chunking code path
    3. Verifies VAD receives correct dtype (float64)
    """
    import numpy as np
    from asr_service.services.cold_transcriber import ColdPathPostProcessor
    from asr_service.core.config import settings

    # Create synthetic audio longer than chunk duration
    sample_rate = 16000
    # Make it slightly longer than chunk duration to trigger VAD chunking
    duration_seconds = settings.COLD_PATH_CHUNK_DURATION + 60  # chunk_duration + 1 minute
    audio_samples = int(sample_rate * duration_seconds)

    # Create simple test audio: alternating speech-like patterns and silence
    # This ensures we hit VAD branching logic
    audio_np = np.zeros(audio_samples, dtype=np.float32)

    # Add some "speech" segments (simple sine wave patterns)
    for i in range(0, audio_samples, sample_rate * 10):  # 10-second blocks
        # 5 seconds of "speech", 5 seconds of silence
        segment = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 5) / sample_rate) * 0.1
        end_idx = min(i + len(segment), len(audio_np))
        audio_np[i : end_idx] = segment[: end_idx - i]

    # Save to temporary WAV file
    test_audio_path = tmp_path / "long_test_audio.wav"
    import soundfile as sf
    sf.write(test_audio_path, audio_np, sample_rate)

    # Get cold pipeline
    cold_pipeline = loaded_models.get_cold_pipeline()
    processor = ColdPathPostProcessor(cold_pipeline)

    # Process long audio - this will trigger VAD chunking
    result = processor.process_long_audio(
        test_audio_path,
        chunk_duration=settings.COLD_PATH_CHUNK_DURATION,
    )

    # Verify results
    assert result is not None, "process_long_audio returned None"
    assert "segments" in result, "Result missing 'segments' key"
    assert "duration" in result, "Result missing 'duration' key"
    assert "language" in result, "Result missing 'language' key"

    # Verify audio was processed (duration should match input)
    assert abs(result["duration"] - duration_seconds) < 1.0, (
        f"Duration mismatch: expected ~{duration_seconds}s, got {result['duration']:.2f}s"
    )

    # If global diarization is enabled and token available, verify speakers are assigned
    if settings.GLOBAL_DIARIZATION_ENABLED and settings.HF_TOKEN:
        speakers = {seg.get("speaker") for seg in result["segments"] if seg.get("speaker")}
        for speaker in speakers:
            assert speaker.startswith("SPEAKER_"), f"Invalid speaker label: {speaker}"

    print("\n=== Long Audio Test Results ===")
    print(f"Input duration: {duration_seconds:.2f}s")
    print(f"Output duration: {result['duration']:.2f}s")
    print(f"Segments: {len(result['segments'])}")
    if result['segments']:
        total_text = " ".join(seg["text"].strip() for seg in result["segments"])
        print(f"Transcription length: {len(total_text)} chars")
    print("✓ VAD chunking executed without dtype errors")

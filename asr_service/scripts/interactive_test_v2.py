#!/usr/bin/env python3
"""
Interactive testing script for ASR pipeline (MLX Version).

Uses MLX-Whisper for native M4 Metal acceleration (2x faster than CPU).

Usage:
    python interactive_test_v2.py --minutes 1     # Test first 1 minute of video
    python interactive_test_v2.py --minutes 5     # Test first 5 minutes of video
    python interactive_test_v2.py --mic           # Record from microphone and transcribe
"""
import argparse
import sys
import time
import threading
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import ffmpeg
from dotenv import load_dotenv

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import unified config
from config import config  # noqa: E402

FULL_VIDEO_PATH = Path(__file__).parent.parent / "data" / "full_meeting_audio.mp3"
TEMP_AUDIO_PATH = Path(__file__).parent.parent / "data" / "temp_segment.mp3"


class Timer:
    """Simple timer for tracking operations."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        print(f"⏱️  {self.name}...", end="", flush=True)
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f" ✓ ({elapsed:.2f}s)")


def extract_first_n_minutes(video_path: Path, minutes: int, output_path: Path):
    """Extract first n minutes from video using ffmpeg."""
    duration = minutes * 60

    try:
        (
            ffmpeg
            .input(str(video_path), t=duration)
            .output(str(output_path), acodec='mp3', ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        sys.exit(1)


def record_from_microphone(output_path: Path, sample_rate: int = 16000):
    """Record audio from microphone until user presses Enter."""
    print("🎤 Recording from microphone...")
    print("   Press ENTER to stop recording")

    recording = []
    recording_active = True
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        if recording_active:
            recording.append(indata.copy())

    def timer_display():
        """Display recording timer."""
        while recording_active:
            elapsed = time.time() - start_time
            print(f"\r   Recording: {elapsed:.1f}s", end="", flush=True)
            time.sleep(0.1)

    # Start timer thread
    timer_thread = threading.Thread(target=timer_display, daemon=True)
    timer_thread.start()

    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
        input()  # Wait for Enter key

    recording_active = False
    timer_thread.join()

    elapsed = time.time() - start_time
    print(f"\n   ✓ Recorded {elapsed:.1f}s")

    # Save recording
    audio_data = np.concatenate(recording, axis=0)
    sf.write(str(output_path), audio_data, sample_rate)

    return elapsed


def transcribe_with_cold_path(audio_path: Path, use_diarization: bool = False):
    """
    Transcribe audio using Cold Path approach (MLX Version):
    1. Diarization (Pyannote) - optional
    2. ASR (MLX-Whisper with Metal acceleration)
    3. Alignment (WhisperX) - optional
    """
    import os
    from cold_path_pipeline_v2 import ColdPathPipeline_MLX

    # Get HF token from environment (needed for diarization)
    hf_token = os.getenv("HF_TOKEN")

    # Create pipeline using settings from config
    pipeline = ColdPathPipeline_MLX(
        whisper_model=config.MLX_WHISPER_MODEL,
        diarization_model=config.DIARIZATION_MODEL,
        use_diarization=use_diarization,
        hf_token=hf_token,
        verbose=True
    )

    # Process audio
    result = pipeline.process(
        audio_path,
        language="en",
        use_alignment=use_diarization  # Only use alignment if diarization is enabled
    )

    # Print formatted result
    print("\n" + pipeline.format_transcript(result))

    # Return full text
    full_text = " ".join([seg['text'].strip() for seg in result['segments']])
    return full_text


def test_minutes(minutes: int, use_diarization: bool = True):
    """Test transcription on first n minutes of video."""
    print(f"\n{'='*60}")
    print(f"Testing first {minutes} minute(s) of video (MLX-Whisper)")
    if use_diarization:
        print("With speaker diarization enabled")
    print(f"{'='*60}\n")

    if not FULL_VIDEO_PATH.exists():
        print(f"❌ Video not found: {FULL_VIDEO_PATH}")
        sys.exit(1)

    with Timer(f"Extracting first {minutes} minute(s)"):
        extract_first_n_minutes(FULL_VIDEO_PATH, minutes, TEMP_AUDIO_PATH)

    transcribe_with_cold_path(TEMP_AUDIO_PATH, use_diarization=use_diarization)

    # Clean up
    TEMP_AUDIO_PATH.unlink()


def test_microphone():
    """Test transcription on microphone input."""
    print(f"\n{'='*60}")
    print("Microphone Recording Test (MLX-Whisper)")
    print(f"{'='*60}\n")

    output_path = Path(__file__).parent.parent / "data" / "mic_recording.wav"

    # Start model loading in background
    import os
    from cold_path_pipeline_v2 import ColdPathPipeline_MLX

    model_loaded = threading.Event()
    model_container = {}

    def load_model():
        hf_token = os.getenv("HF_TOKEN")
        with Timer("Loading Cold Path pipeline (background)"):
            model_container['pipeline'] = ColdPathPipeline_MLX(
                whisper_model=config.MLX_WHISPER_MODEL,
                use_diarization=False,  # Disable for mic test
                hf_token=hf_token,
                verbose=False
            )
            model_loaded.set()

    loader_thread = threading.Thread(target=load_model, daemon=True)
    loader_thread.start()

    # Record audio
    _ = record_from_microphone(output_path)

    # Wait for model to load
    if not model_loaded.is_set():
        print("\n⏳ Waiting for model to finish loading...")
        model_loaded.wait()

    # Transcribe
    transcribe_with_cold_path(output_path)


def main():
    parser = argparse.ArgumentParser(description="Interactive ASR testing script (MLX)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--minutes", type=int, help="Test first N minutes of video")
    group.add_argument("--mic", action="store_true", help="Record from microphone")

    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")

    args = parser.parse_args()

    if args.minutes:
        test_minutes(args.minutes, use_diarization=not args.no_diarization)
    elif args.mic:
        test_microphone()


if __name__ == "__main__":
    main()

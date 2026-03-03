#!/usr/bin/env python3
"""
Live Streaming Transcription Test (Qwen3-ASR-1.7B Version)

Real-time microphone transcription with VAD-based sentence boundary detection.
Uses Qwen3-ASR-1.7B for transcription.

Architecture:
- Producer Thread: Captures audio + runs Silero VAD for speech detection
- Consumer Thread: Runs Qwen3-ASR inference (provisional + finalized)
- Context Chaining: Maintains last 50 words for better accuracy

Usage:
    python live_test_v3.1.py
    # Speak into microphone
    # Press ENTER to stop
"""
import sys
import time
import threading
from pathlib import Path
from collections import deque
from typing import Optional
import warnings

# Suppress known harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

try:
    from qwen_asr import Qwen3ASRModel  # noqa: E402
except ImportError:
    print("❌ Error: qwen-asr package not found!")
    print("   Install with: pip install qwen-asr")
    sys.exit(1)

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import unified config
from config import config  # noqa: E402


# ==============================================================================
# Configuration
# ==============================================================================

SAMPLE_RATE = 16000  # 16kHz for ASR
CHUNK_SIZE = 512     # 32ms chunks (512 samples at 16kHz - required by Silero VAD)
VAD_THRESHOLD = 0.5  # Speech probability threshold
SILENCE_CHUNKS = 15  # ~480ms of silence to trigger finalization (15 * 32ms)
MIN_AUDIO_LENGTH = 0.5  # Minimum 0.5s of audio before transcription
PROVISIONAL_INTERVAL = 0.3  # 300ms between provisional updates
MAX_CONTEXT_WORDS = 50  # Keep last 50 words for context chaining


# ==============================================================================
# Utilities
# ==============================================================================

class Timer:
    """Simple timer for tracking operations."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        print(f"⏱️  {self.name}...", end="", flush=True)
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f" ✓ ({elapsed:.2f}s)")


def append_and_trim_context(context: str, new_text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    """
    Append new text to context and keep only last N words.

    Args:
        context: Existing context string
        new_text: New text to append
        max_words: Maximum number of words to keep

    Returns:
        Trimmed context string
    """
    combined = f"{context} {new_text}".strip()
    words = combined.split()
    if len(words) > max_words:
        return " ".join(words[-max_words:])
    return combined


# ==============================================================================
# Model Loading
# ==============================================================================

def load_qwen_model(model_name: str = "Qwen/Qwen3-ASR-1.7B"):
    """
    Load Qwen3-ASR model using transformers backend.

    Args:
        model_name: Model name (Qwen/Qwen3-ASR-1.7B recommended)

    Returns:
        Loaded Qwen3ASRModel instance
    """
    with Timer(f"Loading Qwen3-ASR ({model_name})"):
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",  # Auto-select best device (cuda/mps/cpu)
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
    return model


def load_vad_model():
    """
    Load Silero VAD model for speech detection.

    Returns:
        Loaded VAD model
    """
    with Timer("Loading Silero VAD"):
        # torch.hub.load returns (model, utils) tuple for silero-vad
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False
        )
    return model


# ==============================================================================
# Producer Thread: Audio Capture + VAD
# ==============================================================================

def producer_thread(vad_model, state: dict, stop_event: threading.Event):
    """
    Captures audio from microphone and runs VAD to detect speech boundaries.

    Args:
        vad_model: Silero VAD model
        state: Shared state dictionary with buffer and flags
        stop_event: Event to signal thread shutdown
    """
    silence_counter = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal silence_counter

        if status:
            print(f"\n⚠️  Audio status: {status}", file=sys.stderr)

        if stop_event.is_set():
            return

        # Convert to float32 for VAD (Silero expects float32 in [-1, 1])
        audio_chunk = indata[:, 0].astype(np.float32)

        # Append to buffer (thread-safe)
        with state['buffer_lock']:
            state['growing_audio_buffer'].append(audio_chunk)

        # Run VAD on this chunk
        audio_tensor = torch.from_numpy(audio_chunk)
        vad_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        # State machine for speech detection
        if vad_prob > VAD_THRESHOLD:
            # Speech detected - reset finalization if it was pending
            if state['commit_ready']:
                # New speech started before consumer finalized - don't interrupt
                pass
            else:
                state['is_speaking'] = True
                silence_counter = 0
        else:
            # Silence detected
            if state['is_speaking'] and not state['commit_ready']:
                silence_counter += 1

                # Trigger finalization after sustained silence
                if silence_counter >= SILENCE_CHUNKS:
                    state['is_speaking'] = False
                    state['commit_ready'] = True
                    silence_counter = 0

    # Start audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            # Keep thread alive until stop signal
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"\n❌ Producer thread error: {e}", file=sys.stderr)
        stop_event.set()


# ==============================================================================
# Consumer Thread: Inference Engine
# ==============================================================================

def consumer_thread(qwen_model: Qwen3ASRModel, state: dict, stop_event: threading.Event):
    """
    Runs Qwen3-ASR inference on the growing audio buffer.

    Performs:
    - Provisional passes while speaking (every 300ms)
    - Finalization pass when speech ends

    Args:
        qwen_model: Loaded Qwen3ASRModel instance
        state: Shared state dictionary
        stop_event: Event to signal thread shutdown
    """

    def copy_buffer_as_numpy() -> Optional[np.ndarray]:
        """Copy current buffer to numpy array (thread-safe)."""
        with state['buffer_lock']:
            if len(state['growing_audio_buffer']) == 0:
                return None
            # Concatenate all chunks
            return np.concatenate(list(state['growing_audio_buffer']))

    def transcribe(audio_np: np.ndarray, provisional: bool = False) -> str:
        """
        Transcribe audio using Qwen3-ASR.

        Args:
            audio_np: Audio data as numpy array
            provisional: If True, use faster settings for real-time updates

        Returns:
            Transcribed text
        """
        start_time = time.time()

        # Qwen3-ASR API: Pass audio as (np.ndarray, sample_rate) tuple
        # language=None enables automatic language detection
        results = qwen_model.transcribe(
            audio=(audio_np, SAMPLE_RATE),
            language=None,  # Auto-detect language
        )

        # Track stats
        elapsed = time.time() - start_time
        state['stats']['total_transcription_time'] += elapsed
        if provisional:
            state['stats']['provisional_count'] += 1
        else:
            state['stats']['finalized_count'] += 1

        # Qwen3-ASR returns list of results with .text attribute
        if results and len(results) > 0:
            text = results[0].text.strip()
            return text
        return ""

    # Main inference loop
    try:
        while not stop_event.is_set():
            # Check if we have enough audio
            audio_np = copy_buffer_as_numpy()

            if audio_np is None or len(audio_np) < int(MIN_AUDIO_LENGTH * SAMPLE_RATE):
                time.sleep(0.1)
                continue

            # Provisional pass (while speaking)
            if not state['commit_ready']:
                text = transcribe(audio_np, provisional=True)
                if text.strip():
                    # Overwrite current line with provisional text
                    print(f"\r💬 {text}", end="", flush=True)

                # Wait before next provisional update
                time.sleep(PROVISIONAL_INTERVAL)

            # Finalization pass (when speech ended)
            else:
                text = transcribe(audio_np, provisional=False)

                if text.strip():
                    # Print finalized text on new line
                    print(f"\r✅ {text}")

                    # Update context for next segment
                    state['previous_context'] = append_and_trim_context(
                        state['previous_context'],
                        text,
                        MAX_CONTEXT_WORDS
                    )

                # Reset buffer and state
                with state['buffer_lock']:
                    state['growing_audio_buffer'].clear()
                state['commit_ready'] = False

    except Exception as e:
        print(f"\n❌ Consumer thread error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        stop_event.set()


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main entry point for live transcription test."""
    print("\n" + "="*60)
    print("🎤 Live Streaming Transcription Test (Qwen3-ASR-1.7B)")
    print("="*60 + "\n")

    # Show audio device info
    print("Audio Device Info:")
    default_input = sd.query_devices(kind='input')
    print(f"  Input: {default_input['name']}")
    print(f"  Channels: {default_input['max_input_channels']}")
    print(f"  Sample Rate: {default_input['default_samplerate']} Hz\n")

    # Load models
    print("Loading models...\n")
    vad_model = load_vad_model()

    # Get model name from config or use default
    qwen_model_name = getattr(config, 'QWEN3_ASR_1_7B_MODEL', 'Qwen/Qwen3-ASR-1.7B')
    qwen_model = load_qwen_model(qwen_model_name)

    print("\n✓ Models loaded successfully!\n")
    print("Press ENTER to stop recording\n")
    print("-" * 60)

    # Initialize shared state
    state = {
        'growing_audio_buffer': deque(),
        'previous_context': "",
        'is_speaking': False,
        'commit_ready': False,
        'buffer_lock': threading.Lock(),
        'stats': {
            'total_transcription_time': 0.0,
            'provisional_count': 0,
            'finalized_count': 0
        }
    }
    stop_event = threading.Event()

    # Track start time
    session_start = time.time()

    # Start producer thread (audio capture + VAD)
    producer = threading.Thread(
        target=producer_thread,
        args=(vad_model, state, stop_event),
        daemon=True
    )

    # Start consumer thread (inference)
    consumer = threading.Thread(
        target=consumer_thread,
        args=(qwen_model, state, stop_event),
        daemon=True
    )

    producer.start()
    consumer.start()

    # Wait for user to press Enter
    try:
        input()
    except KeyboardInterrupt:
        pass

    # Clean shutdown
    print("\n\n🛑 Stopping...")
    stop_event.set()

    # Wait for threads to finish (with timeout)
    producer.join(timeout=2.0)
    consumer.join(timeout=2.0)

    # Calculate session duration
    session_duration = time.time() - session_start

    # Print statistics
    print("\n" + "="*60)
    print("📊 Session Statistics (Qwen3-ASR-1.7B)")
    print("="*60)
    print(f"Total session time:       {session_duration:.2f}s")
    print(f"Total transcription time: {state['stats']['total_transcription_time']:.2f}s")
    print(f"Provisional updates:      {state['stats']['provisional_count']}")
    print(f"Finalized segments:       {state['stats']['finalized_count']}")

    if state['stats']['total_transcription_time'] > 0:
        # Real-time factor: how much faster/slower than real-time
        # RTF < 1.0 means faster than real-time (good for streaming)
        rtf = state['stats']['total_transcription_time'] / session_duration
        print(f"Real-time factor (RTF):   {rtf:.2f}x")
        print("  (RTF < 1.0 = faster than real-time)")

    print("="*60 + "\n")
    print("✓ Done\n")


if __name__ == "__main__":
    main()

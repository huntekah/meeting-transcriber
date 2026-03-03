#!/usr/bin/env python3
"""
Benchmark script for Qwen3-ASR models using pre-recorded audio.

Compares:
- Qwen3-ASR-0.6B
- Qwen3-ASR-1.7B

Metrics:
- Model loading time
- Transcription time
- Real-time factor (RTF)
- Audio duration vs processing time

Memory Requirements:
- Qwen3-ASR-0.6B: ~2-3GB RAM
- Qwen3-ASR-1.7B: ~5-7GB RAM
- Recommend 16GB+ RAM for running both models

Usage:
    python benchmark_qwen.py [audio_file]
    python benchmark_qwen.py --models Qwen/Qwen3-ASR-0.6B  # Test single model
"""
import sys
import time
from pathlib import Path
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import librosa
import gc

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    print("❌ Error: qwen-asr package not found!")
    print("   Install with: uv pip install qwen-asr")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.verbose:
            print(f"⏱️  {self.name}...", end="", flush=True)
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f" ✓ ({self.elapsed:.2f}s)")


def load_audio(audio_path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
    """
    Load audio file and return numpy array.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz for ASR)

    Returns:
        Tuple of (audio_array, duration_in_seconds)
    """
    with Timer(f"Loading audio from {audio_path.name}"):
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        duration = len(audio) / sr

    print(f"   Audio duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Samples: {len(audio):,}")
    return audio, duration


def benchmark_model(model_name: str, audio: np.ndarray, sample_rate: int, audio_duration: float):
    """
    Benchmark a Qwen3-ASR model.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-ASR-0.6B")
        audio: Audio data as numpy array
        sample_rate: Sample rate of audio
        audio_duration: Duration of audio in seconds

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*70)
    print(f"Benchmarking: {model_name}")
    print("="*70)

    # Load model with conservative memory settings
    with Timer(f"Loading {model_name}") as load_timer:
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            max_inference_batch_size=8 if "0.6B" in model_name else 4,  # Reduced to prevent OOM
            max_new_tokens=512,
        )

    # Run transcription
    print("\nRunning transcription...")
    with Timer("Transcription") as transcribe_timer:
        results = model.transcribe(
            audio=(audio, sample_rate),
            language=None,  # Auto-detect
        )

    # Extract text
    text = results[0].text.strip() if results and len(results) > 0 else ""
    detected_language = results[0].language if results and len(results) > 0 else "unknown"

    # Calculate metrics
    rtf = transcribe_timer.elapsed / audio_duration
    throughput = audio_duration / transcribe_timer.elapsed

    # Print results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"Model:                  {model_name}")
    print(f"Model load time:        {load_timer.elapsed:.2f}s")
    print(f"Transcription time:     {transcribe_timer.elapsed:.2f}s")
    print(f"Audio duration:         {audio_duration:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}x")
    print(f"Throughput:             {throughput:.2f}x real-time")
    print(f"Detected language:      {detected_language}")
    print(f"Text length:            {len(text)} chars, {len(text.split())} words")
    print("-"*70)

    # Preview transcription
    preview_length = 200
    print(f"\nTranscription preview (first {preview_length} chars):")
    print(f"  {text[:preview_length]}...")

    result = {
        "model": model_name,
        "load_time": load_timer.elapsed,
        "transcription_time": transcribe_timer.elapsed,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "throughput": throughput,
        "language": detected_language,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split())
    }

    # Clean up model to free memory before loading next model
    print("\nCleaning up model memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def print_comparison(results: list[dict]):
    """Print comparison table of all results."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print()
    print(f"{'Model':<30} {'Load Time':>12} {'Trans Time':>12} {'RTF':>8} {'Throughput':>12}")
    print("-"*70)

    for r in results:
        model_short = r['model'].split('/')[-1]
        print(f"{model_short:<30} {r['load_time']:>10.2f}s {r['transcription_time']:>10.2f}s "
              f"{r['rtf']:>7.3f}x {r['throughput']:>10.2f}x")

    print("="*70)
    print()
    print("RTF = Real-Time Factor (lower is better)")
    print("  RTF < 1.0 = Faster than real-time (good for streaming)")
    print("  RTF > 1.0 = Slower than real-time (not suitable for live)")
    print()
    print("Throughput = How many minutes of audio per minute of processing")
    print("  Higher is better")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR models")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="~/.meeting_scribe/meetings/2026-02-25-20-26/8edfeb0f-7c07-4d82-b459-3b26762177ed_mixed.wav",
        help="Path to audio file (default: 11-minute meeting recording)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"],
        help="Models to benchmark"
    )
    args = parser.parse_args()

    # Expand path
    audio_path = Path(args.audio_file).expanduser()

    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("QWEN3-ASR BENCHMARK")
    print("="*70)
    print(f"Audio file: {audio_path}")
    print(f"Models to test: {', '.join([m.split('/')[-1] for m in args.models])}")

    # Memory warning
    if len(args.models) > 1:
        print("\n⚠️  Testing multiple models - each model will be loaded sequentially")
        print("   Memory will be cleared between models to prevent OOM")
    print()

    # Load audio once
    audio, audio_duration = load_audio(audio_path)

    # Benchmark each model
    results = []
    for model_name in args.models:
        try:
            result = benchmark_model(model_name, audio, 16000, audio_duration)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison if multiple models
    if len(results) > 1:
        print_comparison(results)

    print("\n✓ Benchmark complete!\n")


if __name__ == "__main__":
    main()

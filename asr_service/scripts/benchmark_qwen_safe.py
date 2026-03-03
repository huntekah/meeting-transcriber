#!/usr/bin/env python3
"""
Safe benchmark script for Qwen3-ASR models with better error handling.

Includes:
- CPU-only fallback mode
- Memory monitoring
- Smaller chunk processing
- Device diagnostics

Usage:
    python benchmark_qwen_safe.py [audio_file]
    python benchmark_qwen_safe.py --cpu-only  # Force CPU mode (safer but slower)
    python benchmark_qwen_safe.py --model Qwen/Qwen3-ASR-0.6B  # Test single model
"""
import sys
import time
from pathlib import Path
import argparse
import warnings
import psutil
import os

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


def get_memory_info():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)

    system_mem = psutil.virtual_memory()
    system_gb = system_mem.available / (1024 ** 3)

    return {
        'process_gb': mem_gb,
        'system_available_gb': system_gb,
        'system_percent': system_mem.percent
    }


def print_device_info(device):
    """Print device information."""
    print("\n" + "="*70)
    print("DEVICE INFORMATION")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == 'mps':
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device.type == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mem = get_memory_info()
    print(f"\nSystem memory available: {mem['system_available_gb']:.2f} GB")
    print(f"Process memory usage: {mem['process_gb']:.2f} GB")
    print("="*70)


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


def load_audio(audio_path: Path, sample_rate: int = 16000, max_duration: float = None) -> tuple[np.ndarray, float]:
    """
    Load audio file and return numpy array.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz for ASR)
        max_duration: Maximum duration in seconds (for testing)

    Returns:
        Tuple of (audio_array, duration_in_seconds)
    """
    with Timer(f"Loading audio from {audio_path.name}"):
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=max_duration)
        duration = len(audio) / sr

    print(f"   Audio duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Samples: {len(audio):,}")

    mem = get_memory_info()
    print(f"   Memory after load: {mem['process_gb']:.2f} GB")

    return audio, duration


def benchmark_model(model_name: str, audio: np.ndarray, sample_rate: int,
                   audio_duration: float, device, batch_size: int):
    """
    Benchmark a Qwen3-ASR model.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-ASR-0.6B")
        audio: Audio data as numpy array
        sample_rate: Sample rate of audio
        audio_duration: Duration of audio in seconds
        device: torch device to use
        batch_size: Batch size for inference

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*70)
    print(f"Benchmarking: {model_name}")
    print("="*70)

    # Check memory before loading
    mem_before = get_memory_info()
    print(f"Memory before model load: {mem_before['process_gb']:.2f} GB")

    # Load model
    with Timer(f"Loading {model_name}") as load_timer:
        try:
            # Use explicit device settings to avoid MPS issues
            if device.type == 'cpu':
                model = Qwen3ASRModel.from_pretrained(
                    model_name,
                    dtype=torch.float32,  # CPU requires float32
                    device_map="cpu",
                    max_inference_batch_size=batch_size,
                    max_new_tokens=512,
                )
            else:
                model = Qwen3ASRModel.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map=device.type,
                    max_inference_batch_size=batch_size,
                    max_new_tokens=512,
                )
        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            raise

    # Check memory after loading
    mem_after_load = get_memory_info()
    mem_used_by_model = mem_after_load['process_gb'] - mem_before['process_gb']
    print(f"Memory after model load: {mem_after_load['process_gb']:.2f} GB")
    print(f"Model memory footprint: {mem_used_by_model:.2f} GB")

    # Run transcription
    print("\nRunning transcription...")
    with Timer("Transcription") as transcribe_timer:
        try:
            results = model.transcribe(
                audio=(audio, sample_rate),
                language=None,  # Auto-detect
            )
        except Exception as e:
            print(f"\n❌ Transcription failed: {e}")
            # Clean up before re-raising
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise

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
    print(f"Device:                 {device}")
    print(f"Batch size:             {batch_size}")
    print(f"Model load time:        {load_timer.elapsed:.2f}s")
    print(f"Model memory:           {mem_used_by_model:.2f} GB")
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
        "device": str(device),
        "batch_size": batch_size,
        "load_time": load_timer.elapsed,
        "model_memory_gb": mem_used_by_model,
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
    print("\n🧹 Cleaning up model memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Wait a bit for memory to clear
    time.sleep(2)

    mem_after_cleanup = get_memory_info()
    print(f"   Memory after cleanup: {mem_after_cleanup['process_gb']:.2f} GB")

    return result


def print_comparison(results: list[dict]):
    """Print comparison table of all results."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print()
    print(f"{'Model':<25} {'Device':>8} {'Load':>8} {'Trans':>8} {'RTF':>8} {'Memory':>10}")
    print("-"*70)

    for r in results:
        model_short = r['model'].split('/')[-1]
        device_short = r['device'].split(':')[0]
        print(f"{model_short:<25} {device_short:>8} {r['load_time']:>7.1f}s "
              f"{r['transcription_time']:>7.1f}s {r['rtf']:>7.3f}x {r['model_memory_gb']:>8.2f}GB")

    print("="*70)
    print()
    print("RTF = Real-Time Factor (lower is better)")
    print("  RTF < 1.0 = Faster than real-time (good for streaming)")
    print("  RTF > 1.0 = Slower than real-time (not suitable for live)")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR models (safe mode)")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="~/.meeting_scribe/meetings/2026-02-25-20-26/8edfeb0f-7c07-4d82-b459-3b26762177ed_mixed.wav",
        help="Path to audio file (default: 11-minute meeting recording)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-ASR-0.6B"],  # Default to smaller model only
        help="Models to benchmark (default: 0.6B only)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only mode (slower but safer)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum audio duration in seconds (for testing with shorter clips)"
    )
    args = parser.parse_args()

    # Determine device
    if args.cpu_only:
        device = torch.device('cpu')
        print("⚠️  Running in CPU-only mode (forced)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✓ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU available, using CPU")

    # Print device diagnostics
    print_device_info(device)

    # Expand path
    audio_path = Path(args.audio_file).expanduser()

    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("QWEN3-ASR SAFE BENCHMARK")
    print("="*70)
    print(f"Audio file: {audio_path}")
    print(f"Models to test: {', '.join([m.split('/')[-1] for m in args.models])}")
    print(f"Batch size: {args.batch_size}")

    # Memory warning
    if len(args.models) > 1:
        print("\n⚠️  Testing multiple models - each will be loaded sequentially")
        print("   Memory will be cleared between models")

    if not args.cpu_only and device.type == 'mps':
        print("\n⚠️  Using MPS (Apple GPU) - if you experience crashes, try --cpu-only")

    print()

    # Load audio once
    audio, audio_duration = load_audio(audio_path, max_duration=args.max_duration)

    # Benchmark each model
    results = []
    for i, model_name in enumerate(args.models):
        try:
            print(f"\n{'─'*70}")
            print(f"Model {i+1}/{len(args.models)}")
            print(f"{'─'*70}")

            result = benchmark_model(
                model_name,
                audio,
                16000,
                audio_duration,
                device,
                args.batch_size
            )
            results.append(result)

        except Exception as e:
            print(f"\n❌ Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

            print("\n💡 Suggestions:")
            print("   1. Try --cpu-only flag (slower but more stable)")
            print("   2. Test with shorter audio using --max-duration 30")
            print("   3. Reduce batch size with --batch-size 2")
            print("   4. Test single model only")

            continue

    # Print comparison if multiple models
    if len(results) > 1:
        print_comparison(results)
    elif len(results) == 1:
        print("\n✓ Benchmark complete!")
    else:
        print("\n❌ All benchmarks failed!")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()

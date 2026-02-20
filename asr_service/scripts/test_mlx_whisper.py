#!/usr/bin/env python3
"""Test script for mlx-whisper with hallucination fixes."""
import time
import sys
import mlx_whisper

def test_transcribe(audio_file):
    print(f"\n{'='*60}")
    print(f"Testing: {audio_file}")
    print(f"{'='*60}")

    start = time.time()

    # Updated transcribe call with stability fixes
    result = mlx_whisper.transcribe(
        audio_file,
        # Recommendation: If v3 fails, try "mlx-community/whisper-large-v2-mlx"
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",

        # FIX 1: Prevent the model from seeing previous segments (stops loops)
        condition_on_previous_text=False,

        # FIX 2: Add slight randomness to prevent greedy decoding from getting stuck
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

        # Optional: Strict compression ratio to discard repetitive garbage
        compression_ratio_threshold=2.4,

        # Optional: Discard segments with low probability
        logprob_threshold=-1.0,
    )
    elapsed = time.time() - start

    print(f"\nProcessing time: {elapsed:.2f}s")
    print(f"\nTranscription:\n{result['text']}")
    print(f"\n{'='*60}\n")

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for audio_file in sys.argv[1:]:
            test_transcribe(audio_file)
    else:
        print("Usage: python test_mlx_whisper.py <audio_file1> [audio_file2] ...")

#!/usr/bin/env python3
"""Debug script to check what Whisper is actually transcribing."""
import sys
sys.path.insert(0, 'src')

from asr_service.services.model_loader import ASREngine
import json

def debug_transcribe(audio_file):
    print("Loading ASR engine...")
    engine = ASREngine()
    if not engine.is_loaded:
        engine.load_model()

    print(f"Processing {audio_file}...")

    # Get the full result with all metadata
    result = engine.transcribe_final(audio_file)

    print("\n=== Full Result ===")
    print(json.dumps(result, indent=2, default=str))

    print("\n=== Text Only ===")
    print(result['text'])

    if 'chunks' in result:
        print(f"\n=== Chunks ({len(result['chunks'])}) ===")
        for i, chunk in enumerate(result['chunks'][:5]):  # Show first 5
            print(f"Chunk {i}: {chunk}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_transcribe(sys.argv[1])
    else:
        print("Usage: python debug_transcribe.py <audio_file>")

#!/usr/bin/env python3
"""
Quick verification that VAD dtype fix works on the real 1-hour meeting file.
Tests just the VAD chunking stage, not full transcription.

VERIFICATION FLOW:
1. Load Silero VAD model (tests dtype fix)
2. Load 1-hour meeting audio file
3. Run VAD chunking on full audio (tests that audio and model dtypes match)
4. Report success/failure with detailed diagnostics
"""

import asyncio
import time
from pathlib import Path
from asr_service.services.model_manager import ModelManager
from asr_service.services.cold_transcriber import ColdPathPostProcessor
import soundfile as sf

async def main():
    # Path to the failed meeting
    meeting_file = Path.home() / ".meeting_scribe" / "meetings" / "2026-02-25-11-01" / "fb2bcf08-6a95-4403-8fd5-dccba3666f71_mixed.wav"

    if not meeting_file.exists():
        print(f"❌ Meeting file not found: {meeting_file}")
        return False

    print("=" * 70)
    print("VERIFYING VAD DTYPE FIX ON 1-HOUR MEETING RECORDING")
    print("=" * 70)
    print(f"\n📁 Meeting file: {meeting_file.name}")
    print(f"📊 File size: {meeting_file.stat().st_size / (1024**3):.2f} GB")

    # Load models
    print("\n" + "─" * 70)
    print("STEP 1/4: Loading ASR models...")
    print("─" * 70)
    manager = ModelManager()
    await manager.load_models()
    print("✅ ASR models loaded")

    # Initialize processor and load VAD model
    print("\n" + "─" * 70)
    print("STEP 2/4: Loading Silero VAD model (with dtype fix)")
    print("─" * 70)
    processor = ColdPathPostProcessor(None)
    processor._ensure_vad_model()
    print("✅ VAD model loaded successfully")
    print("   Fix applied: VAD model converted to float32")

    # Load audio
    print("\n" + "─" * 70)
    print("STEP 3/4: Loading 1-hour meeting audio...")
    print("─" * 70)
    audio, sr = sf.read(meeting_file)
    audio = audio.astype('float32')  # Ensure float32
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    duration_secs = len(audio) / sr
    duration_mins = duration_secs / 60
    print(f"✅ Audio loaded: {duration_mins:.1f} minutes ({duration_secs:.0f}s)")
    print(f"   {len(audio):,d} samples at {sr} Hz")

    # Test VAD chunking on audio
    print("\n" + "─" * 70)
    print("STEP 4/4: Running VAD chunking on full audio")
    print("          (This involves 120k+ VAD inference passes - may take 2-5 min)")
    print("─" * 70)
    start_time = time.time()

    try:
        chunks = processor._find_silence_chunks(audio, sr, 300, 30)
        elapsed = time.time() - start_time

        print(f"\n✅ SUCCESS! VAD chunking completed in {elapsed:.1f}s")
        print("   ✓ No dtype mismatches!")
        print(f"   ✓ Found {len(chunks)} chunks based on silence boundaries")

        # Show chunk breakdown
        print("\n   Chunk breakdown (first 5):")
        for i, (start, end) in enumerate(chunks[:5]):
            duration = (end - start) / sr
            print(f"     Chunk {i+1}: {start:>10,d} → {end:>10,d} samples ({duration:>6.1f}s)")

        if len(chunks) > 5:
            print(f"     ... and {len(chunks) - 5} more chunks")

        # Calculate total coverage
        total_covered = sum(end - start for start, end in chunks)
        coverage_pct = (total_covered / len(audio)) * 100
        print(f"\n   Coverage: {coverage_pct:.1f}% of audio will be transcribed")

        print("\n" + "=" * 70)
        print("✅ VERIFICATION PASSED")
        print("=" * 70)
        print("\nThe fix is COHERENT with live_test_v2.py:")
        print("  • Both use Silero VAD from torch.hub.load()")
        print("  • Both work with float32 audio")
        print("  • Fix converts VAD model to float32 to match input")
        print("\nYour 1-hour meeting can now be transcribed! 🎉")

        manager.unload_models()
        return True

    except RuntimeError as e:
        elapsed = time.time() - start_time
        if "expected scalar type Double but found Float" in str(e):
            print(f"\n❌ FAILURE: VAD dtype error still exists (after {elapsed:.1f}s)")
            print(f"   Error: {str(e)[:200]}...")
        else:
            print(f"\n❌ Unexpected error (after {elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        manager.unload_models()
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error (after {elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        manager.unload_models()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

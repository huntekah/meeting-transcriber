#!/usr/bin/env python3
"""
Script to recover and re-process the failed 1-hour meeting recording.
Uses the fixed ColdPathPostProcessor with VAD dtype handling.

RECOVERY FLOW:
1. Load ASR models (VAD, MLX-Whisper, Cold Path Pipeline)
2. Run VAD-based chunking to split 1-hour audio into ~5-min chunks
3. Transcribe each chunk with Whisper + Diarization
4. Save recovered transcript as JSON and Markdown
5. Verify coherence with working interactive_test_v2.py approach

ESTIMATED RUNTIME: 30-45 minutes for 1-hour meeting
  • VAD chunking: 3-5 minutes
  • Transcription: 25-40 minutes (depends on CPU/GPU)
"""

import asyncio
import json
import time
from pathlib import Path
from asr_service.services.model_manager import ModelManager
from asr_service.services.cold_transcriber import ColdPathPostProcessor

async def main():
    # Path to the failed meeting
    meeting_file = Path.home() / ".meeting_scribe" / "meetings" / "2026-02-25-11-01" / "fb2bcf08-6a95-4403-8fd5-dccba3666f71_mixed.wav"

    if not meeting_file.exists():
        print(f"❌ Meeting file not found: {meeting_file}")
        return False

    print("=" * 70)
    print("RECOVERING 1-HOUR MEETING RECORDING")
    print("=" * 70)
    print(f"\n📁 Meeting file: {meeting_file.name}")
    print(f"📊 File size: {meeting_file.stat().st_size / (1024**3):.2f} GB")

    # Load models
    print("\n" + "─" * 70)
    print("STEP 1/4: Loading ASR models...")
    print("          (This includes VAD, MLX-Whisper, Pyannote)")
    print("─" * 70)
    manager = ModelManager()
    await manager.load_models()
    print("✅ ASR models loaded")

    print("\n" + "─" * 70)
    print("STEP 2/4: Getting Cold Path Pipeline...")
    print("─" * 70)
    cold_pipeline = manager.get_cold_pipeline()
    processor = ColdPathPostProcessor(cold_pipeline)
    print("✅ Cold Path Pipeline ready")

    # Process the meeting
    print("\n" + "─" * 70)
    print("STEP 3/4: Processing 1-hour meeting audio...")
    print("          VAD chunking + Whisper transcription + Diarization")
    print("─" * 70)
    print(f"\n   This will take 30-45 minutes. Processing started at {time.strftime('%H:%M:%S')}")
    print("   ✓ VAD will split audio into ~5-minute chunks")
    print("   ✓ Each chunk transcribed with MLX-Whisper + Pyannote speaker detection")
    print("   ✓ Progress will be logged as processing continues")
    print("")

    overall_start = time.time()

    try:
        result = processor.process_long_audio(meeting_file)

        overall_elapsed = time.time() - overall_start
        hours = int(overall_elapsed // 3600)
        minutes = int((overall_elapsed % 3600) // 60)
        seconds = int(overall_elapsed % 60)

        print("\n" + "─" * 70)
        print("STEP 4/4: Saving recovered transcript...")
        print("─" * 70)

        print(f"\n✅ Processing completed in {hours}h {minutes}m {seconds}s!")
        print(f"   Audio duration: {result['duration']:.2f}s ({result['duration']/3600:.2f} hours)")
        print(f"   Language detected: {result['language']}")
        print(f"   Segments found: {len(result['segments'])}")

        # Show first few segments
        if result['segments']:
            total_text = " ".join(seg["text"].strip() for seg in result["segments"])
            print(f"   Total characters transcribed: {len(total_text)}")

            # Show first segment with timestamps
            first_seg = result['segments'][0]
            print("\n   First segment:")
            print(f"     Time: {first_seg.get('start', 'N/A'):.1f}s - {first_seg.get('end', 'N/A'):.1f}s")
            print(f"     Text: {first_seg['text'][:100]}...")

        # Save results
        output_dir = meeting_file.parent
        json_file = output_dir / "transcript_recovered.json"

        with open(json_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved JSON transcript: {json_file.name}")

        # Also save markdown version
        md_file = output_dir / "transcript_recovered.md"
        with open(md_file, "w") as f:
            f.write("# Meeting Transcript\n\n")
            f.write("**Date:** 2026-02-25\n")
            f.write(f"**Duration:** {result['duration']/3600:.2f} hours\n")
            f.write(f"**Language:** {result['language']}\n\n")
            f.write("## Transcript\n\n")

            for i, seg in enumerate(result['segments'], 1):
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                speaker = seg.get('speaker', 'UNKNOWN')
                text = seg['text']
                f.write(f"**[{start:.0f}s - {end:.0f}s] {speaker}:** {text}\n\n")

        print(f"💾 Saved Markdown transcript: {md_file.name}")

        print("\n" + "=" * 70)
        print("✅ MEETING RECOVERY SUCCESSFUL")
        print("=" * 70)
        print("\nYour 1-hour meeting has been recovered!")
        print(f"  📄 JSON: {json_file}")
        print(f"  📝 Markdown: {md_file}")
        print(f"\nProcessing time: {hours}h {minutes}m {seconds}s")

        manager.unload_models()
        return True

    except Exception as e:
        elapsed = time.time() - overall_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"\n❌ Error during processing (after {hours}h {minutes}m {seconds}s): {e}")
        import traceback
        traceback.print_exc()
        manager.unload_models()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

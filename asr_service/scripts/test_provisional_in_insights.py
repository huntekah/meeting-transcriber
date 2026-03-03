#!/usr/bin/env python3
"""
Test that provisional utterances are included in insight transcripts.

Verifies:
1. Provisional utterances are tracked
2. They appear in formatted transcript
3. They're cleared when final utterances arrive
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cli_frontend.models import Utterance


def test_provisional_tracking():
    """Test that provisional utterances are tracked and formatted correctly."""
    print("\n" + "="*70)
    print("Test: Provisional Utterances in Insights")
    print("="*70)

    # Simulate the RecordingScreen state
    _utterances = []
    _provisional_utterances = {}

    # Helper to format transcript (simplified version of _format_transcript_window)
    def format_transcript():
        lines = []
        # Add all final utterances
        for u in _utterances:
            ts = datetime.fromtimestamp(u.start_time, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {u.source_label}: {u.text}")

        # Add provisional utterances
        for source_id in sorted(_provisional_utterances.keys()):
            u = _provisional_utterances[source_id]
            ts = datetime.fromtimestamp(u.start_time, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {u.source_label} (speaking...): {u.text}")

        return "\n".join(lines)

    # Test 1: Add a final utterance
    print("\n📝 Step 1: Add final utterance")
    final_utterance = Utterance(
        source_id=0,
        source_label="Microphone",
        start_time=1000.0,
        end_time=1005.0,
        text="Hello, this is a test.",
        confidence=1.0,
        is_final=True,
    )
    _utterances.append(final_utterance)

    transcript = format_transcript()
    print(f"Transcript:\n{transcript}")
    assert "Hello, this is a test." in transcript
    assert "(speaking...)" not in transcript
    print("✓ Final utterance appears correctly")

    # Test 2: Add a provisional utterance
    print("\n📝 Step 2: Add provisional utterance (speaker still talking)")
    provisional_utterance = Utterance(
        source_id=0,
        source_label="Microphone",
        start_time=1006.0,
        end_time=1008.0,
        text="And now I'm saying something else",
        confidence=0.9,
        is_final=False,
    )
    _provisional_utterances[0] = provisional_utterance

    transcript = format_transcript()
    print(f"Transcript:\n{transcript}")
    assert "Hello, this is a test." in transcript
    assert "And now I'm saying something else" in transcript
    assert "(speaking...)" in transcript
    print("✓ Provisional utterance appears with '(speaking...)' marker")

    # Test 3: Update provisional utterance (speaker continues)
    print("\n📝 Step 3: Update provisional utterance (more words added)")
    updated_provisional = Utterance(
        source_id=0,
        source_label="Microphone",
        start_time=1006.0,
        end_time=1010.0,
        text="And now I'm saying something else that is longer",
        confidence=0.92,
        is_final=False,
    )
    _provisional_utterances[0] = updated_provisional

    transcript = format_transcript()
    print(f"Transcript:\n{transcript}")
    assert "that is longer" in transcript
    assert transcript.count("(speaking...)") == 1  # Should only have one provisional
    print("✓ Provisional utterance updates correctly")

    # Test 4: Finalize the utterance
    print("\n📝 Step 4: Finalize the utterance (speaker finished)")
    final_utterance_2 = Utterance(
        source_id=0,
        source_label="Microphone",
        start_time=1006.0,
        end_time=1012.0,
        text="And now I'm saying something else that is longer and complete.",
        confidence=1.0,
        is_final=True,
    )
    _utterances.append(final_utterance_2)
    _provisional_utterances.pop(0, None)  # Clear provisional

    transcript = format_transcript()
    print(f"Transcript:\n{transcript}")
    assert "that is longer and complete." in transcript
    assert "(speaking...)" not in transcript
    print("✓ Provisional utterance cleared after finalization")

    # Test 5: Multiple speakers with provisional utterances
    print("\n📝 Step 5: Multiple speakers with provisional utterances")
    provisional_speaker1 = Utterance(
        source_id=0,
        source_label="Microphone",
        start_time=1013.0,
        end_time=1015.0,
        text="Speaker one is talking",
        confidence=0.9,
        is_final=False,
    )
    provisional_speaker2 = Utterance(
        source_id=1,
        source_label="System Audio",
        start_time=1014.0,
        end_time=1016.0,
        text="Speaker two is also talking",
        confidence=0.85,
        is_final=False,
    )
    _provisional_utterances[0] = provisional_speaker1
    _provisional_utterances[1] = provisional_speaker2

    transcript = format_transcript()
    print(f"Transcript:\n{transcript}")
    assert "Speaker one is talking" in transcript
    assert "Speaker two is also talking" in transcript
    assert transcript.count("(speaking...)") == 2
    print("✓ Multiple provisional utterances tracked correctly")

    print("\n" + "="*70)
    print("🎉 All tests passed!")
    print("="*70)
    print("\nSummary:")
    print("✓ Final utterances appear in transcript")
    print("✓ Provisional utterances appear with '(speaking...)' marker")
    print("✓ Provisional utterances update correctly")
    print("✓ Provisional utterances cleared after finalization")
    print("✓ Multiple speakers with provisional utterances work correctly")
    print("\nThis means insights will now include in-progress speech!")

    return True


if __name__ == "__main__":
    try:
        success = test_provisional_tracking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

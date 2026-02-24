#!/usr/bin/env python3
"""
Test ScreenCaptureKit audio capture.

Captures system audio using macOS ScreenCaptureKit (requires macOS 13+).
"""

import subprocess
import numpy as np
from pathlib import Path


def test_screencapture_audio(duration: int = 10, sample_rate: int = 16000):
    """
    Test ScreenCaptureKit audio capture.

    Args:
        duration: How long to capture (seconds)
        sample_rate: Sample rate in Hz
    """
    script_dir = Path(__file__).parent
    swift_script = script_dir / "screencapture_audio.swift"

    if not swift_script.exists():
        print(f"❌ Swift script not found: {swift_script}")
        return

    print("=" * 80)
    print("ScreenCaptureKit Audio Capture Test")
    print("=" * 80)
    print(f"\nSample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print("\n⚠️  IMPORTANT: You need to grant Screen Recording permission!")
    print("   Go to: System Settings → Privacy & Security → Screen Recording")
    print("   Enable permission for Terminal/Python")
    print()
    print("🔊 PLAY some audio (YouTube, music, etc.) NOW...")
    print()

    # Wait for user
    input("Press Enter when ready to capture...")

    # Run Swift script and capture stdout
    print("\n📹 Starting ScreenCaptureKit capture...")

    try:
        process = subprocess.Popen(
            ["swift", str(swift_script), str(sample_rate), str(duration)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read raw PCM from stdout
        audio_bytes = bytearray()

        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            audio_bytes.extend(chunk)

        # Wait for process to finish
        process.wait()

        # Print stderr (info messages)
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
        print("\nSwift output:")
        print(stderr_output)

        if process.returncode != 0:
            print(f"❌ Swift script failed with exit code {process.returncode}")
            return

        # Convert bytes to numpy array (float32)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        print(f"\n✅ Captured {len(audio_array)} samples ({len(audio_array) / sample_rate:.2f} seconds)")

        # Analyze audio
        rms = np.sqrt(np.mean(audio_array ** 2))
        max_amplitude = np.max(np.abs(audio_array))

        print("\n📊 Audio Analysis:")
        print(f"   RMS level: {rms:.6f}")
        print(f"   Max amplitude: {max_amplitude:.6f}")

        if rms < 0.0001:
            print("\n❌ NO AUDIO DETECTED!")
            print("   Possible causes:")
            print("   - No audio playing")
            print("   - Permission not granted")
            print("   - System audio muted")
        else:
            print("\n✅ AUDIO DETECTED!")

            # Save to file
            output_file = script_dir / "screencapture_test.wav"
            print(f"\n💾 Saving to: {output_file}")

            import wave
            with wave.open(str(output_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            print(f"   Play with: ffplay {output_file}")

    except FileNotFoundError:
        print("❌ Swift compiler not found. Install Xcode Command Line Tools:")
        print("   xcode-select --install")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n⚠️  NOTE: This requires macOS 13.0 or later")
    print()

    test_screencapture_audio(duration=10, sample_rate=16000)

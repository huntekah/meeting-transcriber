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
    swift_binary = script_dir / "screencapture_audio"

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

    # Compile Swift script if needed (fixes macOS permission chain issues)
    print("\n🔨 Compiling Swift script into binary for better macOS permissions...")
    if not swift_binary.exists() or swift_binary.stat().st_mtime < swift_script.stat().st_mtime:
        try:
            subprocess.run(
                ["swiftc", str(swift_script), "-o", str(swift_binary)],
                check=True,
                capture_output=True,
            )
            print("✅ Compilation successful")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to compile Swift script: {e}")
            print(e.stderr.decode('utf-8', errors='ignore'))
            return
    else:
        print("✅ Using cached binary")

    # Run compiled binary and capture stdout
    print("\n📹 Starting ScreenCaptureKit capture...")

    try:
        process = subprocess.Popen(
            [str(swift_binary), str(sample_rate), str(duration)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for process to complete with timeout (allow extra time for startup/shutdown)
        timeout_seconds = duration + 5
        try:
            stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
            audio_bytes = bytearray(stdout_data)
            stderr_output = stderr_data.decode('utf-8', errors='ignore')
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  Timeout after {timeout_seconds}s - process didn't complete, terminating...")
            process.kill()
            stdout_data, stderr_data = process.communicate(timeout=2)
            audio_bytes = bytearray(stdout_data) if stdout_data else bytearray()
            stderr_output = stderr_data.decode('utf-8', errors='ignore')

        # Print stderr (info messages)
        print("\nSwift output:")
        print(stderr_output)

        if process.returncode != 0:
            print(f"\n❌ Swift script failed with exit code {process.returncode}")
            if "Stream is nil" in stderr_output or "Start stream failed" in stderr_output:
                print("\n⚠️  PERMISSION DENIED: Screen Recording permission not granted!")
                print("\nTo enable Screen Recording permission:")
                print("1. Open System Settings")
                print("2. Go to Privacy & Security → Screen Recording")
                print("3. Add Terminal (or Python) to the allowed apps")
                print("4. Try again")
                print("\nIf you don't see Terminal/Python in the list, try:")
                print("   - python scripts/test_screencapture.py (if running with 'python')")
                print("   - Or use Terminal app instead of IDE terminal")
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

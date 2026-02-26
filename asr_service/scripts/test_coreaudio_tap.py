#!/usr/bin/env python3
"""
Test CoreAudio process taps (macOS 14.2+).

Captures system audio using CoreAudio taps and outputs raw PCM to stdout.
"""

import re
import subprocess
from pathlib import Path

import numpy as np


def _parse_tap_format(stderr_output: str) -> dict:
    match = re.search(
        r"Tap format: sampleRate=(?P<sr>[\d\.]+), channels=(?P<ch>\d+), "
        r"formatID=(?P<fmt>\w+), bitsPerChannel=(?P<bits>\d+), flags=0x(?P<flags>[0-9a-fA-F]+)",
        stderr_output,
    )
    if not match:
        return {}
    return {
        "sample_rate": float(match.group("sr")),
        "channels": int(match.group("ch")),
        "format_id": match.group("fmt"),
        "bits_per_channel": int(match.group("bits")),
        "flags": int(match.group("flags"), 16),
    }


def test_coreaudio_tap(duration: int = 10, sample_rate: int = 16000):
    script_dir = Path(__file__).parent
    swift_script = script_dir / "coreaudio_tap_audio.swift"
    swift_binary = script_dir / "coreaudio_tap_audio"

    if not swift_script.exists():
        print(f"❌ Swift script not found: {swift_script}")
        return

    print("=" * 80)
    print("CoreAudio Tap Audio Capture Test")
    print("=" * 80)
    print(f"\nSample rate request: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print("\n⚠️  Requires macOS 14.2+ and system audio permission.")
    print("   You may need to allow the Terminal/Python binary in System Settings.")
    print()
    print("🔊 PLAY some audio (YouTube, music, etc.) NOW...")
    print()
    input("Press Enter when ready to capture...")

    print("\n🔨 Compiling Swift script into binary...")
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
            print(e.stderr.decode("utf-8", errors="ignore"))
            return
    else:
        print("✅ Using cached binary")

    print("\n🎧 Starting CoreAudio tap capture...")
    try:
        process = subprocess.Popen(
            [str(swift_binary), str(sample_rate), str(duration)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        timeout_seconds = duration + 5
        try:
            stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  Timeout after {timeout_seconds}s - terminating...")
            process.kill()
            stdout_data, stderr_data = process.communicate(timeout=2)

        stderr_output = stderr_data.decode("utf-8", errors="ignore")
        print("\nSwift output:")
        print(stderr_output)

        if process.returncode != 0:
            print(f"\n❌ Swift binary failed with exit code {process.returncode}")
            return

        audio_bytes = bytearray(stdout_data)
        format_info = _parse_tap_format(stderr_output)
        if format_info:
            sample_rate = format_info["sample_rate"]
            channels = format_info["channels"]
            format_id = format_info["format_id"]
            bits = format_info["bits_per_channel"]
            flags = format_info["flags"]
            print(
                f"\n✅ Format: {format_id}, {channels}ch, {sample_rate} Hz, "
                f"{bits} bits, flags=0x{flags:x}"
            )
        else:
            print("\n⚠️  Could not parse tap format; treating as raw bytes")
            channels = 1
            bits = 0
            format_id = "unknown"

        if format_id == "lpcm" and bits == 32:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            duration_seconds = len(audio_array) / sample_rate / max(channels, 1)
            rms = np.sqrt(np.mean(audio_array**2)) if len(audio_array) else 0.0
            max_amplitude = np.max(np.abs(audio_array)) if len(audio_array) else 0.0

            print(f"\n✅ Captured {len(audio_array)} samples (~{duration_seconds:.2f}s)")
            print("\n📊 Audio Analysis:")
            print(f"   RMS level: {rms:.6f}")
            print(f"   Max amplitude: {max_amplitude:.6f}")
        else:
            print(f"\n✅ Captured {len(audio_bytes)} bytes (format {format_id})")

    except FileNotFoundError:
        print("❌ Swift compiler not found. Install Xcode Command Line Tools:")
        print("   xcode-select --install")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_coreaudio_tap(duration=10, sample_rate=16000)

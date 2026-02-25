#!/usr/bin/env python3
"""
Test BlackHole by playing a tone to it while simultaneously capturing.

This verifies the entire pipeline works end-to-end.
"""

import numpy as np
import sounddevice as sd
import time
import threading


def find_device(name_pattern: str) -> int:
    """Find device index by name."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_pattern.lower() in device['name'].lower():
            return i
    return -1


def play_tone_to_blackhole():
    """Play a 440Hz tone (A note) to BlackHole output."""
    blackhole_idx = find_device("BlackHole 2ch")
    if blackhole_idx < 0:
        print("❌ BlackHole 2ch not found")
        return

    print(f"🎵 Playing 440Hz tone to BlackHole 2ch (device {blackhole_idx})...")

    duration = 15  # seconds
    sample_rate = 48000
    frequency = 440  # A note

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% volume
    tone_stereo = np.column_stack([tone, tone])  # Stereo

    # Play to BlackHole
    sd.play(tone_stereo, samplerate=sample_rate, device=blackhole_idx)
    print(f"   Tone playing for {duration} seconds...")
    sd.wait()
    print("   Tone playback finished")


def capture_from_blackhole():
    """Capture from BlackHole input and show levels."""
    blackhole_idx = find_device("BlackHole 2ch")
    if blackhole_idx < 0:
        print("❌ BlackHole 2ch not found")
        return

    print(f"🎙️  Capturing from BlackHole 2ch (device {blackhole_idx})...")

    duration = 15
    sample_rate = 48000
    audio_chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"⚠️  {status}")

        # Calculate level
        level = np.sqrt(np.mean(indata ** 2))
        bar_length = int(level * 100)
        bar = "█" * min(bar_length, 50)

        if bar_length > 0:
            print(f"🔊 Level: {bar:50s} ({level:.6f})", end='\r')
        else:
            print(f"🔇 Level: {bar:50s} ({level:.6f}) [NO AUDIO]", end='\r')

        audio_chunks.append(indata.copy())

    with sd.InputStream(
        device=blackhole_idx,
        channels=2,
        samplerate=sample_rate,
        callback=callback,
    ):
        time.sleep(duration)

    print("\n\n✅ Capture complete")

    # Analyze
    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0)
        audio_mono = audio.mean(axis=1)
        rms = np.sqrt(np.mean(audio_mono ** 2))
        print(f"\nOverall RMS: {rms:.6f}")

        if rms > 0.01:
            print("✅ SUCCESS! Audio captured from BlackHole!")
        else:
            print("❌ FAILED! No audio captured.")


if __name__ == "__main__":
    print("=" * 80)
    print("BlackHole Loopback Test")
    print("=" * 80)
    print("\nThis will play a 440Hz tone TO BlackHole output")
    print("while simultaneously capturing FROM BlackHole input.")
    print("\nIf BlackHole is working, you should see audio levels.\n")

    # Start capture in background thread
    capture_thread = threading.Thread(target=capture_from_blackhole)
    capture_thread.start()

    # Wait a moment for capture to start
    time.sleep(1)

    # Play tone
    play_tone_to_blackhole()

    # Wait for capture to finish
    capture_thread.join()

    print("\n" + "=" * 80)

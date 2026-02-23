#!/usr/bin/env python3
"""
Simple audio capture test to verify device is receiving audio.

Shows real-time audio levels and saves a 10-second recording.
"""

import sys
import time
import numpy as np
import sounddevice as sd
from pathlib import Path


def list_devices():
    """List all available audio devices."""
    print("\n" + "=" * 80)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 80)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append(f"IN:{device['max_input_channels']}ch")
        if device['max_output_channels'] > 0:
            device_type.append(f"OUT:{device['max_output_channels']}ch")

        marker = " [DEFAULT]" if i == sd.default.device[0] else ""
        print(f"{i:2d}: {device['name']:40s} ({', '.join(device_type)}) {marker}")
    print("=" * 80 + "\n")


def test_device(device_index: int, duration: int = 10):
    """
    Test audio capture from a specific device.

    Args:
        device_index: Device index to test
        duration: How long to record (seconds)
    """
    device_info = sd.query_devices(device_index)
    print(f"\n{'=' * 80}")
    print(f"TESTING DEVICE: {device_info['name']}")
    print(f"Channels: {device_info['max_input_channels']}")
    print(f"Sample Rate: {device_info['default_samplerate']} Hz")
    print(f"Recording for {duration} seconds...")
    print(f"{'=' * 80}\n")

    sample_rate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']

    if channels == 0:
        print(f"❌ ERROR: Device {device_index} has no input channels!")
        return

    # Storage for audio
    recorded_audio = []

    def callback(indata, frames, time_info, status):
        """Audio callback - shows live levels and stores audio."""
        if status:
            print(f"⚠️  Stream status: {status}")

        # Calculate RMS level for each channel
        if channels > 1:
            # For stereo, average across channels for level display
            level = np.sqrt(np.mean(indata ** 2))
        else:
            level = np.sqrt(np.mean(indata ** 2))

        # Store audio
        recorded_audio.append(indata.copy())

        # Visual level meter (0-50 chars)
        bar_length = int(level * 1000)  # Scale for visibility
        bar = "█" * min(bar_length, 50)

        # Color based on level
        if bar_length > 0:
            print(f"🔊 Level: {bar:50s} ({level:.6f})", end='\r')
        else:
            print(f"🔇 Level: {bar:50s} ({level:.6f}) [NO AUDIO]", end='\r')

    try:
        # Open stream
        with sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=sample_rate,
            callback=callback,
            blocksize=1024,
        ):
            print("🎙️  Recording... (play some audio now!)\n")
            time.sleep(duration)
            print("\n\n✅ Recording complete!")

        # Analyze results
        if recorded_audio:
            audio_array = np.concatenate(recorded_audio, axis=0)

            # Downmix to mono if stereo
            if channels > 1:
                audio_mono = audio_array.mean(axis=1)
            else:
                audio_mono = audio_array.flatten()

            rms_overall = np.sqrt(np.mean(audio_mono ** 2))
            max_amplitude = np.max(np.abs(audio_mono))

            print(f"\n{'=' * 80}")
            print(f"ANALYSIS")
            print(f"{'=' * 80}")
            print(f"Total samples: {len(audio_mono):,}")
            print(f"Duration: {len(audio_mono) / sample_rate:.2f} seconds")
            print(f"RMS level: {rms_overall:.6f}")
            print(f"Max amplitude: {max_amplitude:.6f}")

            if rms_overall < 0.0001:
                print(f"\n❌ NO AUDIO DETECTED!")
                print(f"   Possible causes:")
                print(f"   - No audio playing through this device")
                print(f"   - Wrong device selected")
                print(f"   - Audio routing not configured (for BlackHole, need to route audio to it)")
            else:
                print(f"\n✅ AUDIO DETECTED! (RMS: {rms_overall:.6f})")

            # Save to file
            output_file = Path(__file__).parent / f"test_device_{device_index}.wav"
            print(f"\n💾 Saving to: {output_file}")

            import wave
            with wave.open(str(output_file), 'wb') as wf:
                wf.setnchannels(1)  # Save as mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_mono * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            print(f"   You can play it with: ffplay {output_file}")
            print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def find_device_by_name(name_pattern: str) -> int:
    """Find device index by name pattern."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_pattern.lower() in device['name'].lower():
            return i
    return -1


if __name__ == "__main__":
    list_devices()

    # Auto-detect BlackHole 2ch
    blackhole_idx = find_device_by_name("BlackHole 2ch")

    if len(sys.argv) > 1:
        # User specified device index
        device_idx = int(sys.argv[1])
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        test_device(device_idx, duration)
    elif blackhole_idx >= 0:
        # Auto-test BlackHole
        print(f"🎯 Found BlackHole 2ch at index {blackhole_idx}")
        print(f"   Testing automatically (or specify device: python {sys.argv[0]} <device_index>)")
        test_device(blackhole_idx, duration=10)
    else:
        print(f"❓ BlackHole 2ch not found.")
        print(f"\nUsage: python {sys.argv[0]} <device_index> [duration_seconds]")
        print(f"Example: python {sys.argv[0]} 3 10")

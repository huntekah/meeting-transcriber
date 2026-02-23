#!/usr/bin/env python3
"""
Test creating a session with 2 sources to verify both pipelines start.
"""

import asyncio
import httpx
import sounddevice as sd


def find_device(name_pattern: str) -> dict:
    """Find input device by name."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Only consider INPUT devices (max_input_channels > 0)
        if device['max_input_channels'] > 0 and name_pattern.lower() in device['name'].lower():
            return {
                "device_index": i,
                "device_name": device['name'],
                "device_channels": device['max_input_channels'],
            }
    return None


async def test_two_sources():
    """Create session with microphone + BlackHole."""
    api_base = "http://localhost:8000"

    # Find devices
    mic = find_device("WH-1000XM5") or find_device("Microphone")
    blackhole = find_device("BlackHole 2ch")

    if not mic:
        print("❌ No microphone found")
        return

    if not blackhole:
        print("❌ BlackHole 2ch not found")
        return

    print(f"🎙️  Found microphone: {mic['device_name']} (device {mic['device_index']})")
    print(f"🔊 Found BlackHole: {blackhole['device_name']} (device {blackhole['device_index']})")

    # Create session with BOTH sources
    payload = {
        "sources": [
            mic,
            blackhole,
        ]
    }

    print(f"\n📤 Creating session with 2 sources...")
    print(f"   Payload: {payload}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{api_base}/api/v1/sessions", json=payload)
            response.raise_for_status()
            result = response.json()

            print(f"\n✅ Session created!")
            print(f"   Session ID: {result['session_id']}")
            print(f"   State: {result['state']}")
            print(f"   WebSocket: {result['websocket_url']}")

            # Wait a moment for pipelines to start
            await asyncio.sleep(2)

            # Get session details
            print(f"\n📊 Fetching session details...")
            response = await client.get(f"{api_base}/api/v1/sessions/{result['session_id']}")
            session = response.json()

            print(f"   State: {session['state']}")
            print(f"   Utterances so far: {len(session['utterances'])}")

            # Check backend logs
            print(f"\n💡 Now check backend logs for:")
            print(f"   - 'Creating pipeline for source 0'")
            print(f"   - 'Creating pipeline for source 1'")
            print(f"   - 'VADAudioProducer 0 started'")
            print(f"   - 'VADAudioProducer 1 started'")

            print(f"\n🎤 Speak into your mic and play audio through BlackHole...")
            print(f"   Watch CLI log: tail -f ~/.asr_cli_debug.log | grep 'source='")
            print(f"   You should see BOTH source=0 and source=1")

            # Keep session alive
            print(f"\n⏸  Session running. Press Ctrl+C to stop...")
            try:
                await asyncio.sleep(60)
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping session...")

            # Stop session
            response = await client.post(
                f"{api_base}/api/v1/sessions/{result['session_id']}/stop",
                timeout=60.0
            )
            print(f"   Session stopped: {response.json()}")

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"   {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_two_sources())

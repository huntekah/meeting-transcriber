#!/usr/bin/env python3
"""
Test two sources with live WebSocket output.

Shows utterances from both sources in real-time.
"""

import asyncio
import httpx
import websockets
import json
import sounddevice as sd


def find_device(name_pattern: str) -> dict:
    """Find input device by name."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0 and name_pattern.lower() in device['name'].lower():
            return {
                "device_index": i,
                "device_name": device['name'],
                "device_channels": device['max_input_channels'],
            }
    return None


async def listen_websocket(ws_url: str, session_id: str):
    """Listen to WebSocket and display utterances."""
    full_url = f"http://localhost:8000{ws_url}".replace("http://", "ws://")

    print(f"\n🔌 Connecting to WebSocket: {full_url}")

    utterance_count = {0: 0, 1: 0}

    try:
        async with websockets.connect(full_url) as websocket:
            print("✅ WebSocket connected!\n")
            print("=" * 80)
            print("LIVE TRANSCRIPTION (Ctrl+C to stop)")
            print("=" * 80)
            print()

            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    if data.get("type") == "utterance":
                        utterance = data["data"]
                        source_id = utterance["source_id"]
                        text = utterance["text"]

                        utterance_count[source_id] = utterance_count.get(source_id, 0) + 1

                        # Color codes
                        colors = {
                            0: "\033[96m",  # Cyan
                            1: "\033[92m",  # Green
                        }
                        reset = "\033[0m"

                        color = colors.get(source_id, "")
                        label = f"[Source {source_id}]"

                        print(f"{color}{label:12s}{reset} {text}")
                        print(f"             {color}(Total: {utterance_count[source_id]}){reset}\n")

                    elif data.get("type") == "state_change":
                        state = data["state"]
                        print(f"\n⚙️  State changed: {state}\n")

                        if state in ["completed", "failed"]:
                            break

                except websockets.ConnectionClosed:
                    print("\n❌ WebSocket connection closed")
                    break

    except Exception as e:
        print(f"\n❌ WebSocket error: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Source 0 utterances: {utterance_count.get(0, 0)}")
    print(f"Source 1 utterances: {utterance_count.get(1, 0)}")

    if utterance_count.get(1, 0) == 0:
        print("\n⚠️  NO SOURCE 1 UTTERANCES!")
        print("   Possible causes:")
        print("   - No audio routed through BlackHole")
        print("   - YouTube playing music (not speech)")
        print("   - VAD not detecting speech (volume too low)")


async def main():
    """Create session and monitor both sources."""
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

    print("=" * 80)
    print("TWO-SOURCE TEST")
    print("=" * 80)
    print(f"\nSource 0 (Microphone): {mic['device_name']}")
    print(f"Source 1 (System Audio): {blackhole['device_name']}")

    # Create session
    payload = {"sources": [mic, blackhole]}

    print("\n📤 Creating session...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{api_base}/api/v1/sessions", json=payload)
            response.raise_for_status()
            result = response.json()

            session_id = result['session_id']
            ws_url = result['websocket_url']

            print(f"✅ Session created: {session_id}")
            print("\n🎙️  NOW SPEAK into your microphone!")
            print("🔊 AND PLAY a YouTube video with talking (not music)")
            print("\nYou should see:")
            print("  - \033[96m[Source 0]\033[0m utterances (cyan) from your mic")
            print("  - \033[92m[Source 1]\033[0m utterances (green) from BlackHole")

            # Listen to WebSocket
            await listen_websocket(ws_url, session_id)

            # Stop session
            print("\n🛑 Stopping session...")
            response = await client.post(
                f"{api_base}/api/v1/sessions/{session_id}/stop",
                timeout=60.0
            )
            print("✅ Session stopped")

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"   {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")

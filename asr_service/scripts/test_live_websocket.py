#!/usr/bin/env python3
"""
Test script to verify live transcription WebSocket broadcasts.

Run this while the backend is running (`make run`) to see if utterances are being broadcast.
"""

import asyncio
import websockets
import json
import httpx


async def test_websocket_broadcast():
    """Test WebSocket connection and message reception."""
    api_base = "http://localhost:8000"

    print("=== Testing Live WebSocket Broadcast ===\n")

    # 1. Get available devices
    print("1. Fetching devices...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base}/api/v1/devices")
        devices = response.json()
        print(f"   Found {len(devices)} devices")

        # Find default device
        default_device = next((d for d in devices if d["is_default"]), devices[0])
        print(f"   Using device: {default_device['name']}\n")

        # 2. Create session
        print("2. Creating session...")
        session_data = {
            "sources": [{
                "device_index": default_device["device_index"],
                "device_name": default_device["name"],
                "device_channels": default_device["channels"]
            }]
        }
        response = await client.post(f"{api_base}/api/v1/sessions", json=session_data)
        session_info = response.json()
        session_id = session_info["session_id"]
        ws_url = session_info["websocket_url"]
        print(f"   Session ID: {session_id}")
        print(f"   WebSocket URL: {ws_url}\n")

    # 3. Connect to WebSocket
    print("3. Connecting to WebSocket...")
    full_ws_url = f"ws://localhost:8000{ws_url}"

    received_messages = []
    utterance_count = 0

    try:
        async with websockets.connect(full_ws_url) as websocket:
            print(f"   Connected to {full_ws_url}")
            print("   Listening for messages (say something into your microphone)...\n")

            # Listen for 30 seconds
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < 30:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    received_messages.append(data)

                    msg_type = data.get("type")

                    if msg_type == "state_change":
                        print(f"   [STATE] {data['state']}")

                    elif msg_type == "utterance":
                        utterance_count += 1
                        utt = data["data"]
                        print(f"   [UTTERANCE #{utterance_count}] Source {utt['source_id']}: {utt['text'][:80]}")
                        print(f"      Time: {utt['start_time']:.2f}s - {utt['end_time']:.2f}s")
                        print(f"      Overlaps: {utt.get('overlaps_with', [])}")

                    elif msg_type == "final_transcript":
                        print(f"   [FINAL] {len(data['transcript']['segments'])} segments")

                    elif msg_type == "error":
                        print(f"   [ERROR] {data}")

                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    print("   WebSocket connection closed")
                    break

            print(f"\n4. Test complete!")
            print(f"   Total messages received: {len(received_messages)}")
            print(f"   Utterances received: {utterance_count}")

            # Stop session
            print("\n5. Stopping session...")
            print("   (This may take 20-30s due to cold path processing...)")
            async with httpx.AsyncClient(timeout=60.0) as client:  # 60s timeout for cold path
                try:
                    response = await client.post(f"{api_base}/api/v1/sessions/{session_id}/stop")
                    print(f"   Session stopped (status: {response.status_code})")
                except httpx.ReadTimeout:
                    print("   Session stop timed out (cold path still processing)")
                    print("   This is normal - session will complete in background")

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_websocket_broadcast())

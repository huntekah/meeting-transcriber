#!/usr/bin/env python3
"""
Debug script to check active session sources and their status.

Run this while a recording session is active to see what's happening.
"""

import asyncio
import httpx
import sys


async def debug_session(session_id: str):
    """Check session status and source activity."""
    api_base = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Get session info
            response = await client.get(f"{api_base}/api/v1/sessions/{session_id}")
            session = response.json()

            print("=" * 80)
            print("SESSION DEBUG INFO")
            print("=" * 80)
            print(f"\nSession ID: {session['session_id']}")
            print(f"State: {session['state']}")
            if 'created_at' in session:
                print(f"Created: {session['created_at']}")

            print(f"\n{'=' * 80}")
            print(f"SOURCES ({len(session['sources'])} configured)")
            print(f"{'=' * 80}")

            for i, source in enumerate(session['sources']):
                print(f"\n[Source {i}]")
                print(f"  Device Index: {source['device_index']}")
                print(f"  Device Name: {source['device_name']}")
                print(f"  Device Channels: {source.get('device_channels', 'N/A')}")

            # Get active sessions (includes stats)
            response = await client.get(f"{api_base}/api/v1/sessions")
            sessions = response.json()

            active_session = None
            for s in sessions:
                if s['session_id'] == session_id:
                    active_session = s
                    break

            if active_session and 'stats' in active_session:
                print(f"\n{'=' * 80}")
                print("RUNTIME STATS")
                print(f"{'=' * 80}")
                stats = active_session['stats']

                print(f"\nPipelines running: {stats.get('pipelines_running', 'N/A')}")

                if 'pipelines' in stats:
                    print(f"\nPer-source stats:")
                    for pipeline_stat in stats['pipelines']:
                        source_id = pipeline_stat.get('source_id', '?')
                        print(f"\n  [Source {source_id}]")
                        print(f"    Running: {pipeline_stat.get('is_running', False)}")
                        print(f"    Segments processed: {pipeline_stat.get('segments_processed', 0)}")
                        print(f"    Utterances: {pipeline_stat.get('utterances_count', 0)}")

                        if 'producer_stats' in pipeline_stat:
                            prod = pipeline_stat['producer_stats']
                            print(f"    Producer:")
                            print(f"      Audio chunks captured: {prod.get('total_audio_chunks', 0)}")
                            print(f"      Currently speaking: {prod.get('is_speaking', False)}")
                            print(f"      Buffer size: {prod.get('buffer_size', 0)}")

                        if 'transcriber_stats' in pipeline_stat:
                            trans = pipeline_stat['transcriber_stats']
                            print(f"    Transcriber:")
                            print(f"      Queue size: {trans.get('queue_size', 0)}")
                            print(f"      Segments processed: {trans.get('segments_processed', 0)}")

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"   {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def list_active_sessions():
    """List all active sessions."""
    api_base = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{api_base}/api/v1/sessions")
            sessions = response.json()

            # Check if response is a list
            if not isinstance(sessions, list):
                print(f"❌ Unexpected response format: {type(sessions)}")
                print(f"   Response: {sessions}")
                return None

            if not sessions:
                print("No active sessions found.")
                return None

            print("=" * 80)
            print("ACTIVE SESSIONS")
            print("=" * 80)

            for session in sessions:
                print(f"\nSession ID: {session['session_id']}")
                print(f"  State: {session['state']}")
                print(f"  Sources: {len(session.get('sources', []))}")
                for i, src in enumerate(session.get('sources', [])):
                    print(f"    [{i}] {src.get('device_name', 'Unknown')}")

            if len(sessions) == 1:
                return sessions[0]['session_id']

            return None

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


async def main():
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
        await debug_session(session_id)
    else:
        # Auto-detect active session
        session_id = await list_active_sessions()
        if session_id:
            print(f"\n{'=' * 80}")
            print("Auto-detected session, fetching details...")
            print(f"{'=' * 80}")
            await debug_session(session_id)
        else:
            print("\nUsage: python debug_session.py [session_id]")
            print("Or run while a session is active to auto-detect")


if __name__ == "__main__":
    asyncio.run(main())

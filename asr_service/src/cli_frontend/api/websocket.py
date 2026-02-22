"""
WebSocket client for live transcription updates.

Handles WebSocket connection to ASR service and message dispatching.
"""

import asyncio
import websockets
import json
from typing import Callable, Optional
from websockets.exceptions import ConnectionClosed


class WSClient:
    """WebSocket client for live updates from ASR service."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def connect(self, on_message: Callable):
        """
        Connect to WebSocket and listen for messages.

        Args:
            on_message: Async callback function called for each message.
                        Receives dict with parsed JSON data.
        """
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.websocket = websocket
                self._running = True

                while self._running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await on_message(data)
                    except ConnectionClosed:
                        break
                    except json.JSONDecodeError as e:
                        # Log but don't crash on malformed JSON
                        print(f"JSON decode error: {e}")
                        continue
                    except Exception as e:
                        # Log but don't crash on callback errors
                        print(f"Message handler error: {e}")
                        continue

        except ConnectionClosed:
            pass
        except Exception as e:
            print(f"WebSocket connection error: {e}")
        finally:
            self._running = False
            self.websocket = None

    def start(self, on_message: Callable) -> asyncio.Task:
        """
        Start WebSocket connection in background task.

        Args:
            on_message: Async callback for messages

        Returns:
            asyncio.Task that can be cancelled
        """
        self._task = asyncio.create_task(self.connect(on_message))
        return self._task

    def disconnect(self):
        """Disconnect WebSocket gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def send(self, data: dict):
        """
        Send message to server.

        Args:
            data: Dictionary to send as JSON
        """
        if self.websocket and self._running:
            await self.websocket.send(json.dumps(data))

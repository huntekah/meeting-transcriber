"""
WebSocket client for live transcription updates.

Handles WebSocket connection to ASR service and message dispatching.
"""

import asyncio
import websockets
import json
from typing import Callable, Optional
from websockets.exceptions import ConnectionClosed
from ..logging import logger


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
        logger.info(f"Connecting to WebSocket: {self.ws_url}")
        message_count = 0
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.websocket = websocket
                self._running = True
                logger.info(f"WebSocket connected successfully")
                while self._running:
                    try:
                        message = await websocket.recv()
                        message_count += 1
                        logger.debug(f"Received message #{message_count}: {message[:100]}...")
                        data = json.loads(message)
                        logger.debug(f"Parsed message type: {data.get('type')}")
                        await on_message(data)
                        logger.debug(f"Message handler completed")
                    except ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                    except json.JSONDecodeError as e:
                        # Log but don't crash on malformed JSON
                        logger.error(f"JSON decode error: {e}")
                        continue
                    except Exception as e:
                        # Log but don't crash on callback errors
                        logger.error(f"Message handler error: {e}", exc_info=True)
                        continue

        except ConnectionClosed:
            logger.info("WebSocket connection closed gracefully")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}", exc_info=True)
        finally:
            self._running = False
            self.websocket = None
            logger.info(f"WebSocket disconnected (received {message_count} messages total)")

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

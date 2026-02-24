"""
Recording screen with live transcript view.

Shows real-time transcription updates via WebSocket.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, Button
from cli_frontend.widgets.transcript_view import TranscriptView
from cli_frontend.widgets.status_bar import StatusBar
from cli_frontend.api.client import ASRClient
from cli_frontend.api.websocket import WSClient
from cli_frontend.config import CLISettings
from cli_frontend.models import WSUtteranceMessage, WSStateChangeMessage
from cli_frontend.logging import logger


class RecordingScreen(Screen):
    """Live recording screen with transcript view."""

    BINDINGS = [
        ("ctrl+r", "stop_recording", "Stop"),
        ("ctrl+q", "quit", "Quit"),
        ("escape", "stop_recording", "Stop"),
    ]

    def __init__(
        self, session_id: str, ws_url: str, client: ASRClient, settings: CLISettings
    ):
        super().__init__()
        self.session_id = session_id
        # Ensure WebSocket URL is properly formatted
        if not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
            # Extract host from API base URL
            base_url = settings.api_base_url.replace("http://", "").replace(
                "https://", ""
            )
            self.ws_url = f"ws://{base_url}{ws_url}"
        else:
            self.ws_url = ws_url

        self.client = client
        self.settings = settings
        self.ws_client = None
        self.state = "initializing"
        self._stopping = False

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        with Vertical(id="recording_container"):
            yield StatusBar(id="status_bar")

            yield TranscriptView(id="transcript")

            with Horizontal(id="controls"):
                yield Button("⏹ Stop Recording", variant="error", id="stop_btn")

        yield Footer()

    async def on_mount(self):
        """Start WebSocket connection when mounted."""
        logger.info(f"RecordingScreen mounted for session {self.session_id}")
        logger.info(f"WebSocket URL: {self.ws_url}")

        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.start_recording()

        # Start WebSocket connection
        logger.info("Starting WebSocket client...")
        self.ws_client = WSClient(self.ws_url)
        self.ws_client.start(self.on_ws_message)
        logger.info("WebSocket client started")

    async def on_ws_message(self, data: dict):
        """
        Handle WebSocket messages.

        Args:
            data: Parsed JSON message from WebSocket
        """
        msg_type = data.get("type")
        logger.info(f"Received WebSocket message type: {msg_type}")

        if msg_type == "utterance":
            # New transcription utterance
            try:
                logger.info(f"Processing utterance message: {data}")
                msg = WSUtteranceMessage(**data)
                logger.info(f"Parsed utterance: source={msg.data.source_id}, text='{msg.data.text[:50]}...'")

                transcript = self.query_one("#transcript", TranscriptView)
                logger.debug(f"Got TranscriptView widget: {transcript}")

                transcript.add_utterance(msg.data)
                logger.info("Utterance added to transcript view")

            except Exception as e:
                logger.error(f"Error processing utterance: {e}", exc_info=True)

        elif msg_type == "state_change":
            # Session state transition
            try:
                logger.info(f"Processing state change: {data}")
                msg = WSStateChangeMessage(**data)
                self.state = msg.state
                status_bar = self.query_one("#status_bar", StatusBar)

                if msg.state == "stopping":
                    status_bar.set_status("⏸ Stopping...")
                elif msg.state == "processing":
                    status_bar.set_status("⚙ Processing transcript...")
                    status_bar.stop_recording()
                elif msg.state == "completed":
                    status_bar.set_status("✓ Completed")
                    # Auto-return to setup after brief delay
                    self.set_timer(2.0, self.return_to_setup)
                elif msg.state == "failed":
                    status_bar.set_status("❌ Failed")

                logger.info(f"State changed to: {msg.state}")

            except Exception as e:
                logger.error(f"Error processing state change: {e}", exc_info=True)

        elif msg_type == "final_transcript":
            # Final cold path transcript ready
            logger.info("Received final transcript")
            status_bar = self.query_one("#status_bar", StatusBar)
            status_bar.set_status("✓ Final transcript ready")

        elif msg_type == "error":
            # Error notification
            message = data.get("message", "Unknown error")
            logger.error(f"Received error message: {message}")
            status_bar = self.query_one("#status_bar", StatusBar)
            status_bar.set_status(f"❌ Error: {message}")

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def action_stop_recording(self):
        """Handle stop recording action."""
        await self.stop_recording()

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button clicks."""
        if event.button.id == "stop_btn":
            self.app.call_later(self.stop_recording)

    async def stop_recording(self):
        """Stop recording and disconnect."""
        if self._stopping:
            return

        self._stopping = True
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.set_status("⏹ Stopping recording...")

        try:
            # Stop session via API
            await self.client.stop_session(self.session_id)

        except Exception as e:
            status_bar.set_status(f"❌ Error stopping: {e}")
            self._stopping = False
            return

    def return_to_setup(self):
        """Return to setup screen."""
        # Disconnect WebSocket
        if self.ws_client:
            self.ws_client.disconnect()

        # Pop back to setup screen
        self.app.pop_screen()

    async def on_unmount(self):
        """Cleanup when screen is unmounted."""
        if self.ws_client:
            self.ws_client.disconnect()

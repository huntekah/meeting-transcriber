"""
Recording screen with live transcript view and BYT insights pane.

Shows real-time transcription updates via WebSocket, plus side-by-side
LLM-powered insights (BYT pane). Hotkeys toggle each panel for focus.
"""

from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer
from textual import work
from cli_frontend.widgets.transcript_view import TranscriptView, LiveTranscriptView
from cli_frontend.widgets.byt_pane import BytPane
from cli_frontend.widgets.status_bar import StatusBar
from cli_frontend.api.client import ASRClient
from cli_frontend.api.insights_client import InsightsClient
from cli_frontend.api.websocket import WSClient
from cli_frontend.config import CLISettings
from cli_frontend.models import (
    Utterance,
    WSUtteranceMessage,
    WSStateChangeMessage,
)
from cli_frontend.logging import logger


class RecordingScreen(Screen):
    """Live recording screen with transcript view and BYT insights pane."""

    BINDINGS = [
        ("escape", "stop_recording", "Stop"),
        ("ctrl+x", "confirm_discard", "Discard"),
        ("ctrl+comma", "open_settings", "Settings"),
        # Keep but hide from Footer — available via keyboard / code
        ("ctrl+t", "toggle_transcript", ""),
        ("ctrl+b", "toggle_byt", ""),
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
        self._discarding = False

        # Accumulated final utterances for insight context window
        self._utterances: list[Utterance] = []

        # Insights HTTP client
        self._insights_client = InsightsClient(settings.insights_service_url)

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        with Vertical(id="recording_container"):
            yield StatusBar(id="status_bar")

            # Split-screen: TranscriptView (left) + BytPane (right)
            with Horizontal(id="main_split"):
                yield TranscriptView(id="transcript")
                yield BytPane(
                    skills=[],  # populated async in on_mount via GET /skills
                    auto_refresh_seconds=self.settings.insight_auto_refresh_seconds,
                    default_context_minutes=self.settings.insight_context_minutes,
                    id="byt_pane",
                )

            yield LiveTranscriptView(id="live_transcript")

        yield Footer()

    async def on_mount(self):
        """Start WebSocket connection and load BYT skills when mounted."""
        logger.info(f"RecordingScreen mounted for session {self.session_id}")
        logger.info(f"WebSocket URL: {self.ws_url}")

        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.start_recording()

        # Start WebSocket connection
        logger.info("Starting WebSocket client...")
        self.ws_client = WSClient(self.ws_url)
        self.ws_client.start(self.on_ws_message)
        logger.info("WebSocket client started")

        # Load skills from LLM Intelligence service (non-blocking; pane shows guidance if empty)
        skills = await self._insights_client.get_skills()
        byt = self.query_one("#byt_pane", BytPane)
        if skills:
            await byt.reload_skills(skills)
            logger.info(f"BYT pane loaded {len(skills)} skill(s): {[s.name for s in skills]}")
        else:
            logger.warning("No skills loaded — BYT pane will show setup guidance")

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
                logger.info(
                    f"Parsed utterance: source={msg.data.source_id}, text='{msg.data.text[:50]}...'"
                )

                transcript = self.query_one("#transcript", TranscriptView)
                live_transcript = self.query_one("#live_transcript", LiveTranscriptView)
                logger.debug(f"Got TranscriptView widget: {transcript}")

                if not msg.data.is_final and msg.data.text.startswith(
                    "[TRANSCRIPTION ERROR"
                ):
                    transcript.add_utterance(msg.data)
                    live_transcript.clear_partial(msg.data.source_id)
                    logger.info("Error utterance added to transcript view")
                elif msg.data.is_final:
                    transcript.add_utterance(msg.data)
                    live_transcript.clear_partial(msg.data.source_id)
                    self._utterances.append(msg.data)
                    logger.info("Utterance added to transcript view")
                elif msg.data.text.strip():
                    await live_transcript.update_partial(msg.data)
                    logger.info("Live utterance updated")

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
                elif msg.state == "cancelled":
                    status_bar.set_status("✗ Discarded")
                    self.set_timer(1.0, self.return_to_setup)

                if msg.state in {"stopping", "processing", "completed", "failed", "cancelled"}:
                    live_transcript = self.query_one(
                        "#live_transcript", LiveTranscriptView
                    )
                    live_transcript.clear_partial()

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

    # ------------------------------------------------------------------
    # BYT insight fetching
    # ------------------------------------------------------------------

    def on_byt_pane_refresh_requested(self, event: BytPane.RefreshRequested) -> None:
        """Triggered by BytPane when a refresh is needed. Fires a background worker."""
        self._fetch_insight(event.skill_name, event.window_minutes)

    @work(exclusive=False)
    async def _fetch_insight(
        self, skill_name: str, window_minutes: float | None
    ) -> None:
        """Format transcript window, call insights service, update BYT pane.

        Runs as a Textual background worker so the UI stays responsive during
        the LLM HTTP call (which can take 10-30 seconds).
        """
        byt = self.query_one("#byt_pane", BytPane)
        byt.set_loading(skill_name, True)

        transcript_text = self._format_transcript_window(window_minutes)
        if not transcript_text.strip():
            byt.update_insight(
                skill_name, "*No transcript yet — start speaking to generate insights.*"
            )
            return

        try:
            response = await self._insights_client.get_insight(
                transcript=transcript_text,
                skill_name=skill_name,
            )
            byt.update_insight(skill_name, response.markdown)
        except Exception as e:
            logger.error(f"Insight fetch failed for {skill_name}: {e}", exc_info=True)
            byt.update_insight(
                skill_name,
                f"*⚠ Could not reach insights service: {e}*\n\n"
                "Make sure the LLM Intelligence service is running (`make run-insights`).",
            )

    def _format_transcript_window(self, window_minutes: float | None) -> str:
        """
        Format accumulated utterances into timestamped plain text.

        Args:
            window_minutes: If given, only include utterances from the last N minutes.
                            None means the full session.
        """
        utterances = self._utterances
        if window_minutes is not None and utterances:
            cutoff = utterances[-1].start_time - window_minutes * 60
            utterances = [u for u in utterances if u.start_time >= cutoff]

        lines = []
        for u in utterances:
            ts = datetime.fromtimestamp(u.start_time, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {u.source_label}: {u.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Toggle hotkeys
    # ------------------------------------------------------------------

    def action_toggle_transcript(self) -> None:
        """Ctrl+T — hide/show TranscriptView (BYT expands to fill)."""
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.toggle_class("hidden")

    def action_toggle_byt(self) -> None:
        """Ctrl+B — hide/show BytPane (TranscriptView expands to fill)."""
        byt = self.query_one("#byt_pane", BytPane)
        byt.toggle_class("hidden")

    # ------------------------------------------------------------------
    # Existing controls
    # ------------------------------------------------------------------

    async def action_stop_recording(self):
        """Handle stop recording action."""
        await self.stop_recording()

    def action_confirm_discard(self) -> None:
        """Ctrl+X — open discard confirmation modal."""
        if self._stopping or self._discarding:
            return
        from .discard_modal import DiscardModal
        self.app.push_screen(DiscardModal(), callback=self._on_discard_result)

    def _on_discard_result(self, confirmed: bool) -> None:
        if confirmed:
            self.app.call_later(self.discard_recording)

    def action_open_settings(self) -> None:
        """Ctrl+, — open settings modal."""
        from cli_frontend.screens.settings import SettingsScreen
        self.app.push_screen(SettingsScreen(self.settings))

    async def stop_recording(self):
        """Stop recording and disconnect."""
        if self._stopping or self._discarding:
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

    async def discard_recording(self):
        """Cancel recording and discard outputs."""
        if self._stopping or self._discarding:
            return

        self._discarding = True
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.set_status("✗ Discarding recording...")

        try:
            await self.client.cancel_session(self.session_id)
            self.return_to_setup()

        except Exception as e:
            status_bar.set_status(f"❌ Error discarding: {e}")
            self._discarding = False
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
        await self._insights_client.close()


"""
Setup screen — zero-friction launch dashboard.

Shows auto-detected device names as status labels.
"⚙ Configure Devices" reveals device selectors for those who need to change them.
"""

from typing import List
from textual.timer import Timer
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Label,
    Static,
)
from cli_frontend.widgets.device_selector import DeviceSelector
from cli_frontend.api.client import ASRClient
from cli_frontend.config import CLISettings
from cli_frontend.models import SourceConfig, AudioDevice
from cli_frontend.settings_persistence import persistence
from cli_frontend.logging import logger


class SetupScreen(Screen):
    """Zero-friction launch dashboard."""

    RETRY_SECONDS = 2

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+comma", "open_settings", "Settings"),
    ]

    def __init__(self, client: ASRClient, settings: CLISettings):
        super().__init__()
        self.client = client
        self.settings = settings
        self.devices: List[AudioDevice] = []
        self.loading = False
        self._retry_timer: Timer | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="setup_container"):
            with Horizontal(id="setup_header"):
                yield Label("🎙 MeetingScribe", id="title")
                yield Static("⏳ Connecting…", id="ready_badge")

            # Auto-selected source names (updated after devices load)
            yield Static("", id="source_labels")

            with Horizontal(id="button_container"):
                yield Button(
                    "▶  Start Recording",
                    variant="primary",
                    id="start_btn",
                    disabled=True,
                )

            with Collapsible(title="⚙ Configure", collapsed=True, id="configure_collapsible"):
                yield Label("Microphone:", classes="field_label")
                yield DeviceSelector([], id="mic_selector", label="Select microphone…")
                yield Label("System audio (optional):", classes="field_label")
                yield DeviceSelector(
                    [], id="system_selector", label="Select system audio…"
                )

            yield Static("", id="status_message")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self):
        await self.load_devices()

    def on_screen_resume(self) -> None:
        """Clear transient status message when returning from recording."""
        self.query_one("#status_message", Static).update("")

    # ------------------------------------------------------------------
    # Device loading
    # ------------------------------------------------------------------

    async def load_devices(self):
        """Fetch devices from backend and populate selectors + status labels."""
        if self.loading:
            return
        self.loading = True
        self._set_badge("⏳ Connecting…", "badge_loading")

        try:
            self.devices = await self.client.get_devices()
            logger.info(f"Received {len(self.devices)} devices from backend")

            saved_mic_name, saved_system_name = persistence.load_device_selection()

            mic_selector = self.query_one("#mic_selector", DeviceSelector)
            mic_selector.set_devices(
                self.devices,
                select_default=(saved_mic_name is None),
                pre_select_name=saved_mic_name,
            )

            system_selector = self.query_one("#system_selector", DeviceSelector)
            system_selector.set_devices(
                self.devices,
                select_default=False,
                pre_select_name=saved_system_name,
            )

            self.query_one("#status_message", Static).update("")
            self._refresh_status_labels()
            if self._retry_timer:
                self._retry_timer.stop()
                self._retry_timer = None

        except Exception as e:
            logger.error(f"Error loading devices: {e}")
            self._set_badge("🔴 Backend unreachable", "badge_error")
            self.query_one("#status_message", Static).update(
                f"Cannot reach ASR service at {self.settings.api_base_url}"
            )
            if not self._retry_timer:
                self._retry_timer = self.set_interval(self.RETRY_SECONDS, self._retry_load_devices)
        finally:
            self.loading = False

    def _retry_load_devices(self) -> None:
        """Retry device loading until the backend becomes available."""
        if self.loading:
            return
        self.app.call_later(self.load_devices)

    def _refresh_status_labels(self):
        """Update the source label display and ready badge based on current selections."""
        mic_idx = self.query_one("#mic_selector", DeviceSelector).value
        sys_idx = self.query_one("#system_selector", DeviceSelector).value

        lines = []
        if mic_idx is not None:
            dev = next((d for d in self.devices if d.device_index == mic_idx), None)
            if dev:
                lines.append(f"🎤  {dev.name}")
        if sys_idx is not None:
            dev = next((d for d in self.devices if d.device_index == sys_idx), None)
            if dev:
                lines.append(f"🔊  {dev.name}")

        label_widget = self.query_one("#source_labels", Static)
        if lines:
            label_widget.update("\n".join(lines))
            self._set_badge("🟢 Ready", "badge_ready")
            self.query_one("#start_btn", Button).disabled = False
        else:
            label_widget.update("No audio source selected — open ⚙ Configure")
            self._set_badge("🟡 Select a device", "badge_warn")
            self.query_one("#start_btn", Button).disabled = True

    def _set_badge(self, text: str, css_class: str):
        badge = self.query_one("#ready_badge", Static)
        badge.update(text)
        for cls in ("badge_loading", "badge_ready", "badge_warn", "badge_error"):
            badge.remove_class(cls)
        badge.add_class(css_class)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "start_btn":
            self.app.call_later(self.start_recording)

    def action_open_settings(self) -> None:
        from cli_frontend.screens.settings import SettingsScreen
        self.app.push_screen(SettingsScreen(self.settings))

    # ------------------------------------------------------------------
    # Recording start
    # ------------------------------------------------------------------

    async def start_recording(self):
        """Validate selections and create session."""
        if self.loading:
            return

        status = self.query_one("#status_message", Static)
        mic_idx = self.query_one("#mic_selector", DeviceSelector).value
        system_idx = self.query_one("#system_selector", DeviceSelector).value
        output_dir = self.settings.output_dir  # comes from Settings / persistence

        if mic_idx is None and system_idx is None:
            status.update("❌ Please select at least one audio source")
            return

        sources = []
        mic_device_name = None
        system_device_name = None

        if mic_idx is not None:
            device = next((d for d in self.devices if d.device_index == mic_idx), None)
            if device:
                mic_device_name = device.name
                sources.append(
                    SourceConfig(
                        device_index=mic_idx,
                        device_name=device.name,
                        device_channels=device.channels,
                        source_type=device.source_type,
                    )
                )

        if system_idx is not None:
            device = next((d for d in self.devices if d.device_index == system_idx), None)
            if device:
                system_device_name = device.name
                sources.append(
                    SourceConfig(
                        device_index=system_idx,
                        device_name=device.name,
                        device_channels=device.channels,
                        source_type=device.source_type,
                    )
                )

        if not sources:
            status.update("❌ Invalid device selection")
            return

        persistence.save_device_selection(mic_device_name, system_device_name)

        status.update("Starting…")
        try:
            session_data = await self.client.create_session(
                sources=sources,
                output_dir=output_dir or None,
                language=self.settings.asr_language,
            )
            from .recording import RecordingScreen
            self.app.push_screen(
                RecordingScreen(
                    session_id=session_data["session_id"],
                    ws_url=session_data["websocket_url"],
                    client=self.client,
                    settings=self.settings,
                )
            )
        except Exception as e:
            status.update(f"❌ Error creating session: {e}")

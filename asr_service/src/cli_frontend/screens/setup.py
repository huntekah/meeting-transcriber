"""
Setup screen for device selection and configuration.

First screen shown to user when starting the app.
"""

from typing import List
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Input, Label, Static
from cli_frontend.widgets.device_selector import DeviceSelector
from cli_frontend.api.client import ASRClient
from cli_frontend.config import CLISettings
from cli_frontend.models import SourceConfig, AudioDevice
from cli_frontend.settings_persistence import persistence
from cli_frontend.logging import logger


class SetupScreen(Screen):
    """Device selection and configuration screen."""

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, client: ASRClient, settings: CLISettings):
        super().__init__()
        self.client = client
        self.settings = settings
        self.devices: List[AudioDevice] = []
        self.loading = False

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        with Vertical(id="setup_container"):
            yield Label("🎙 MeetingScribe Setup", id="title")
            yield Static()  # Spacer

            yield Label("Select your audio sources:")
            yield Label("Primary microphone:", classes="field_label")
            yield DeviceSelector([], id="mic_selector", label="Select microphone...")

            yield Label("System audio (optional):", classes="field_label")
            yield DeviceSelector(
                [], id="system_selector", label="Select system audio..."
            )

            yield Static()  # Spacer

            yield Label("Save transcripts to:", classes="field_label")
            yield Input(
                value=self.settings.output_dir,
                placeholder="~/Documents/MeetingScribe/",
                id="output_dir",
            )

            yield Static()  # Spacer

            with Horizontal(id="button_container"):
                yield Button("Start Recording", variant="primary", id="start_btn")
                yield Button("Quit", variant="default", id="quit_btn")

            yield Static(id="status_message")

        yield Footer()

    async def on_mount(self):
        """Load devices when screen is mounted."""
        await self.load_devices()

    async def load_devices(self):
        """Fetch devices from backend and populate selectors."""
        self.loading = True
        status = self.query_one("#status_message", Static)
        status.update("Loading devices...")

        try:
            self.devices = await self.client.get_devices()

            # Log all devices received
            logger.info(f"Received {len(self.devices)} devices from backend:")
            for d in self.devices:
                logger.info(
                    f"  [{d.device_index}] {d.name} ({d.channels}ch, {d.sample_rate}Hz) - source_type={d.source_type}"
                )

            # Load saved device selections (by name — robust against index shifts)
            saved_mic_name, saved_system_name = persistence.load_device_selection()
            logger.debug(
                f"Loaded saved selections: mic={saved_mic_name}, system={saved_system_name}"
            )

            # Populate microphone selector
            mic_selector = self.query_one("#mic_selector", DeviceSelector)
            mic_selector.set_devices(
                self.devices,
                select_default=(saved_mic_name is None),
                pre_select_name=saved_mic_name,
            )

            # Populate system audio selector
            system_selector = self.query_one("#system_selector", DeviceSelector)
            system_selector.set_devices(
                self.devices,
                select_default=False,
                pre_select_name=saved_system_name,
            )

            status.update(f"✓ Found {len(self.devices)} audio device(s)")

        except Exception as e:
            logger.error(f"Error loading devices: {e}")
            status.update(f"❌ Error loading devices: {e}")
        finally:
            self.loading = False

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button clicks."""
        if event.button.id == "start_btn":
            self.app.call_later(self.start_recording)
        elif event.button.id == "quit_btn":
            self.app.exit()

    async def start_recording(self):
        """Validate selections and create session."""
        if self.loading:
            return

        status = self.query_one("#status_message", Static)

        # Get selected devices
        mic_idx = self.query_one("#mic_selector").value
        system_idx = self.query_one("#system_selector").value
        output_dir = self.query_one("#output_dir").value

        # Validate at least one source selected
        if mic_idx is None and system_idx is None:
            status.update("❌ Please select at least one audio source")
            return

        # Build sources list
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
            device = next(
                (d for d in self.devices if d.device_index == system_idx), None
            )
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

        # Save device selections by name for next time (robust against index shifts)
        persistence.save_device_selection(mic_device_name, system_device_name)
        persistence.save_output_dir(output_dir)
        logger.debug(
            f"Saved device selection: mic={mic_device_name}, system={system_device_name}, output={output_dir}"
        )

        # Create session
        status.update("Creating session...")
        try:
            session_data = await self.client.create_session(
                sources=sources, output_dir=output_dir or None
            )

            # Import here to avoid circular import
            from .recording import RecordingScreen

            # Switch to recording screen
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

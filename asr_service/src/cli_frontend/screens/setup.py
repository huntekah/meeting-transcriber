"""
Setup screen for device selection and configuration.

First screen shown to user when starting the app.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Input, Label, Static
from ..widgets.device_selector import DeviceSelector
from ..api.client import ASRClient
from ..config import CLISettings
from ..models import SourceConfig


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
        self.devices = []
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
            yield DeviceSelector([], id="system_selector", label="Select system audio...")

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

            # Populate device selectors
            mic_selector = self.query_one("#mic_selector", DeviceSelector)
            mic_selector.set_devices(self.devices, select_default=True)

            system_selector = self.query_one("#system_selector", DeviceSelector)
            system_selector.set_devices(self.devices, select_default=False)

            status.update(f"✓ Found {len(self.devices)} audio device(s)")

        except Exception as e:
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
        if mic_idx is not None:
            device = next((d for d in self.devices if d.device_index == mic_idx), None)
            if device:
                sources.append(SourceConfig(device_index=mic_idx, device_name=device.name))

        if system_idx is not None:
            device = next((d for d in self.devices if d.device_index == system_idx), None)
            if device:
                sources.append(SourceConfig(device_index=system_idx, device_name=device.name))

        if not sources:
            status.update("❌ Invalid device selection")
            return

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

"""
Settings screen for configuring CLI frontend preferences.

Modal screen for adjusting appearance and output settings.
"""

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Label, Switch, Button, Input, Static
from cli_frontend.config import CLISettings


class SettingsScreen(ModalScreen):
    """Settings modal screen."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, settings: CLISettings):
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="settings_modal"):
            with Vertical():
                yield Label("⚙ Settings", id="settings_title")
                yield Static()  # Spacer

                # Appearance section
                yield Label("Appearance", classes="section_header")

                with Horizontal(classes="setting_row"):
                    yield Switch(
                        value=self.settings.show_timestamps, id="show_timestamps"
                    )
                    yield Label("Show timestamps")

                with Horizontal(classes="setting_row"):
                    yield Switch(
                        value=self.settings.show_backchannels, id="show_backchannels"
                    )
                    yield Label('Show backchannels ("uh-huh", "yeah")')

                with Horizontal(classes="setting_row"):
                    yield Switch(value=self.settings.auto_scroll, id="auto_scroll")
                    yield Label("Auto-scroll transcript")

                yield Static()  # Spacer

                # Output section
                yield Label("Output", classes="section_header")

                yield Label("Save directory:", classes="field_label")
                yield Input(
                    value=self.settings.output_dir,
                    id="output_dir",
                    placeholder="~/Documents/MeetingScribe/",
                )

                yield Label("API base URL:", classes="field_label")
                yield Input(
                    value=self.settings.api_base_url,
                    id="api_base_url",
                    placeholder="http://localhost:8000",
                )

                yield Static()  # Spacer

                # Buttons
                with Horizontal(id="settings_buttons"):
                    yield Button("Save", variant="primary", id="save_btn")
                    yield Button("Cancel", variant="default", id="cancel_btn")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button clicks."""
        if event.button.id == "save_btn":
            self.save_settings()
        elif event.button.id == "cancel_btn":
            self.dismiss(False)

    def save_settings(self):
        """Save settings and dismiss modal."""
        # Update settings object
        self.settings.show_timestamps = self.query_one("#show_timestamps", Switch).value
        self.settings.show_backchannels = self.query_one(
            "#show_backchannels", Switch
        ).value
        self.settings.auto_scroll = self.query_one("#auto_scroll", Switch).value
        self.settings.output_dir = self.query_one("#output_dir", Input).value
        self.settings.api_base_url = self.query_one("#api_base_url", Input).value

        # Note: Settings are not persisted to disk in V1
        # Could be added in future with settings.model_dump() → JSON file

        self.dismiss(True)

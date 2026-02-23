"""
Main Textual application for MeetingScribe CLI frontend.

Entry point for the CLI testing tool.
"""

from textual.app import App
from textual.binding import Binding
from .screens.setup import SetupScreen
from .screens.settings import SettingsScreen
from .api.client import ASRClient
from .config import settings


class MeetingScribeApp(App):
    """MeetingScribe CLI Application."""

    # CSS styling file
    CSS_PATH = "app.tcss"

    # App metadata
    TITLE = "MeetingScribe"
    SUB_TITLE = "Multi-Source ASR Testing Tool"

    # Global keybindings
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+comma", "settings", "Settings"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.settings = settings
        self.client = ASRClient(self.settings)

    def on_mount(self):
        """Initialize app on mount."""
        # Start with setup screen
        self.push_screen(SetupScreen(self.client, self.settings))

    async def action_settings(self):
        """Show settings modal."""

        def on_dismiss(result):
            if result:
                # Settings were saved
                self.notify("Settings saved")

        await self.push_screen(SettingsScreen(self.settings), on_dismiss)

    async def action_help(self):
        """Show help information."""
        help_text = """
        MeetingScribe - Multi-Source ASR Testing Tool

        Keyboard Shortcuts:
        - Ctrl+Q / Ctrl+C: Quit
        - Ctrl+, : Settings
        - Ctrl+R: Stop recording (in recording screen)
        - F1: This help
        - Escape: Back / Cancel

        Workflow:
        1. Select audio devices (microphone, system audio)
        2. Click "Start Recording"
        3. View live transcription
        4. Click "Stop Recording" or press Ctrl+R
        5. Wait for final transcript processing

        Features:
        - Multi-source audio capture
        - Live transcription with MLX-Whisper
        - Overlap detection
        - Speaker diarization (cold path)
        - Source-specific coloring

        For more info, see the documentation.
        """
        self.notify(help_text, severity="information", timeout=15)

    async def action_quit(self):
        """Quit application and cleanup."""
        # Close HTTP client
        await self.client.close()
        self.exit()

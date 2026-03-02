"""
Settings screen for configuring CLI frontend preferences.

Modal screen with four sections: General, Audio Engine, Intelligence (LLM), Output.
Settings are persisted to ~/.meeting_scribe/cli_config.json on Save.
"""

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Label, Switch, Button, Input, Static, RadioButton, RadioSet
from cli_frontend.config import CLISettings
from cli_frontend.settings_persistence import persistence


class SettingsScreen(ModalScreen):
    """Settings modal screen."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, settings: CLISettings):
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="settings_modal"):
            yield Label("⚙ Settings", id="settings_title")
            with VerticalScroll(id="settings_scroll"):
                # ── General ───────────────────────────────────────────
                yield Label("General", classes="section_header")

                with Horizontal(classes="setting_row"):
                    yield Switch(value=self.settings.show_timestamps, id="show_timestamps")
                    yield Label("Show timestamps")

                with Horizontal(classes="setting_row"):
                    yield Switch(value=self.settings.auto_scroll, id="auto_scroll")
                    yield Label("Auto-scroll transcript")

                yield Label("BYT auto-refresh (seconds):", classes="field_label")
                yield Input(
                    value=str(self.settings.insight_auto_refresh_seconds),
                    id="insight_auto_refresh_seconds",
                    placeholder="60",
                )

                yield Label("Default context window (minutes):", classes="field_label")
                yield Input(
                    value=str(self.settings.insight_context_minutes),
                    id="insight_context_minutes",
                    placeholder="5",
                )

                # ── Audio Engine ──────────────────────────────────────
                yield Label("Audio Engine", classes="section_header")

                with Horizontal(classes="setting_row"):
                    yield Switch(value=self.settings.use_local_asr, id="use_local_asr")
                    yield Label("Local Processing (mlx-whisper · Apple Silicon)")

                # ── Intelligence ──────────────────────────────────────
                yield Label("Intelligence (LLM)", classes="section_header")

                with RadioSet(id="llm_backend"):
                    yield RadioButton(
                        "Local (Ollama)",
                        id="llm_local",
                        value=self.settings.use_local_llm,
                    )
                    yield RadioButton(
                        "Cloud (Gemini)",
                        id="llm_cloud",
                        value=not self.settings.use_local_llm,
                    )

                yield Label("Ollama URL (leave blank for localhost:11434):", classes="field_label")
                yield Input(
                    value=self.settings.ollama_host if self.settings.ollama_host != "http://localhost:11434" else "",
                    id="ollama_host",
                    placeholder="http://localhost:11434",
                )
                yield Label("Model:", classes="field_label")
                yield Input(value=self.settings.ollama_model, id="ollama_model", placeholder="gemma3:4b-it-qat")

                yield Label("Gemini API Key (leave blank to use .env):", classes="field_label")
                yield Input(
                    value="",
                    id="gemini_api_key",
                    placeholder="AIza… or leave blank to use GEMINI_API_KEY from .env",
                    password=True,
                )

                # ── Output ────────────────────────────────────────────
                yield Label("Output", classes="section_header")

                yield Label("Save directory:", classes="field_label")
                yield Input(value=self.settings.output_dir, id="output_dir", placeholder="~/.meeting_scribe/meetings/")

            # Buttons pinned outside scroll area
            with Horizontal(id="settings_buttons"):
                yield Button("Save", variant="primary", id="save_btn")
                yield Button("Cancel", variant="default", id="cancel_btn")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save_btn":
            self.save_settings()
        elif event.button.id == "cancel_btn":
            self.dismiss(False)

    def save_settings(self):
        """Save settings to memory and persist to disk."""
        def _int(widget_id: str, default: int) -> int:
            try:
                return int(self.query_one(f"#{widget_id}", Input).value)
            except (ValueError, TypeError):
                return default

        use_local = self.query_one("#llm_local", RadioButton).value

        self.settings.show_timestamps = self.query_one("#show_timestamps", Switch).value
        self.settings.auto_scroll = self.query_one("#auto_scroll", Switch).value
        self.settings.use_local_asr = self.query_one("#use_local_asr", Switch).value
        self.settings.insight_auto_refresh_seconds = _int("insight_auto_refresh_seconds", 60)
        self.settings.insight_context_minutes = _int("insight_context_minutes", 5)
        self.settings.use_local_llm = use_local
        self.settings.ollama_host = self.query_one("#ollama_host", Input).value or "http://localhost:11434"
        self.settings.ollama_model = self.query_one("#ollama_model", Input).value or "gemma3:4b-it-qat"
        self.settings.gemini_api_key = self.query_one("#gemini_api_key", Input).value  # empty = use .env
        self.settings.output_dir = self.query_one("#output_dir", Input).value

        persistence.save_general_settings({
            "show_timestamps": self.settings.show_timestamps,
            "auto_scroll": self.settings.auto_scroll,
            "use_local_asr": self.settings.use_local_asr,
            "insight_auto_refresh_seconds": self.settings.insight_auto_refresh_seconds,
            "insight_context_minutes": self.settings.insight_context_minutes,
            "use_local_llm": self.settings.use_local_llm,
            "ollama_host": self.settings.ollama_host,
            "ollama_model": self.settings.ollama_model,
            "output_dir": self.settings.output_dir,
        })

        self.dismiss(True)

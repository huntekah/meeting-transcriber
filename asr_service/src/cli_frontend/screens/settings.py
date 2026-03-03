"""
Settings screen for configuring CLI frontend preferences.

Modal screen with four sections: General, Audio Engine, Intelligence (LLM), Output.
Settings are persisted to ~/.meeting_scribe/cli_config.json on Save.
"""

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Label, Switch, Button, Input, Static, RadioButton, RadioSet, Select
from rich.text import Text
from textual import work
from cli_frontend.config import CLISettings
from cli_frontend.settings_persistence import persistence
from cli_frontend.api.insights_client import InsightsClient
from cli_frontend.models import ModelInfo


class SettingsScreen(ModalScreen):
    """Settings modal screen."""

    MODEL_LOADING = "__loading__"
    MODEL_ERROR = "__error__"

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, settings: CLISettings):
        super().__init__()
        self.settings = settings
        self._insights_client = InsightsClient(self.settings.insights_service_url)

    def on_mount(self) -> None:
        self._load_models()

    async def on_unmount(self) -> None:
        await self._insights_client.close()

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

                yield Label("Transcript / BYT split (transcript %):", classes="field_label")
                yield Select(
                    [
                        ("30% transcript", 30),
                        ("50% transcript", 50),
                        ("70% transcript (default)", 70),
                        ("80% transcript", 80),
                        ("90% transcript", 90),
                    ],
                    id="split_transcript_percent",
                    value=self.settings.split_transcript_percent,
                    allow_blank=False,
                )

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

                yield Label("Transcription language:", classes="field_label")
                yield Select(
                    [
                        ("Auto-detect", "auto"),
                        ("English", "en"),
                        ("Polish", "pl"),
                    ],
                    id="asr_language",
                    value=self.settings.asr_language,
                    allow_blank=False,
                )

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
                yield Label("Ollama model:", classes="field_label")
                yield Select(
                    [("Loading models...", self.MODEL_LOADING)],
                    id="ollama_model",
                    allow_blank=False,
                    disabled=True,
                )

                yield Label("Gemini model:", classes="field_label")
                yield Select(
                    [("Loading models...", self.MODEL_LOADING)],
                    id="gemini_model",
                    allow_blank=False,
                    disabled=True,
                )

                yield Label("Gemini API Key (leave blank to use .env):", classes="field_label")
                yield Input(
                    value="",
                    id="gemini_api_key",
                    placeholder="AIza... or leave blank to use GEMINI_API_KEY from .env",
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
    # Model loading
    # ------------------------------------------------------------------

    @work(exclusive=False)
    async def _load_models(self) -> None:
        await self._load_provider_models(
            provider="ollama",
            select_id="#ollama_model",
            current_value=self.settings.ollama_model,
            formatter=self._format_ollama_option,
        )
        await self._load_provider_models(
            provider="gemini",
            select_id="#gemini_model",
            current_value=self.settings.gemini_model,
            formatter=self._format_gemini_option,
        )

    async def _load_provider_models(
        self,
        provider: str,
        select_id: str,
        current_value: str,
        formatter,
    ) -> None:
        response = await self._insights_client.get_models(provider)
        select = self.query_one(select_id, Select)

        if response.errors:
            self._set_select_placeholder(select, "Failed to load models", self.MODEL_ERROR)
            return

        if not response.models:
            self._set_select_placeholder(select, "No models found", self.MODEL_ERROR)
            return

        options = [formatter(model) for model in response.models]
        options.sort(key=lambda opt: (opt[0].plain if isinstance(opt[0], Text) else str(opt[0])).lower())
        values = [value for _, value in options]

        if current_value and current_value not in values:
            options.append((Text(f"{current_value} (not found)"), current_value))

        select.set_options(options)
        select.disabled = False
        select.value = current_value if current_value in values else options[0][1]

    def _set_select_placeholder(self, select: Select, label: str, value: str) -> None:
        select.set_options([(Text(label), value)])
        select.value = value
        select.disabled = True

    def _format_size(self, size_bytes: int | None) -> str:
        if not size_bytes:
            return "unknown"
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _format_ollama_option(self, model: ModelInfo) -> tuple[Text, str]:
        size_label = self._format_size(model.size_bytes)
        return Text(f"{model.id} ({size_label})"), model.id

    def _format_gemini_option(self, model: ModelInfo) -> tuple[Text, str]:
        display = model.display_name or model.id
        limits = []
        if model.input_token_limit:
            limits.append(f"in={model.input_token_limit}")
        if model.output_token_limit:
            limits.append(f"out={model.output_token_limit}")
        limits_label = f"[{', '.join(limits)}]" if limits else ""
        return Text(f"{display} {limits_label}") if limits_label else Text(display), model.id

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
        self.settings.asr_language = self.query_one("#asr_language", Select).value
        self.settings.split_transcript_percent = int(
            self.query_one("#split_transcript_percent", Select).value
        )
        self.settings.insight_auto_refresh_seconds = _int("insight_auto_refresh_seconds", 60)
        self.settings.insight_context_minutes = _int("insight_context_minutes", 5)
        self.settings.use_local_llm = use_local
        self.settings.ollama_host = self.query_one("#ollama_host", Input).value or "http://localhost:11434"

        ollama_select = self.query_one("#ollama_model", Select)
        if not ollama_select.disabled and ollama_select.value not in (self.MODEL_LOADING, self.MODEL_ERROR):
            self.settings.ollama_model = ollama_select.value

        gemini_select = self.query_one("#gemini_model", Select)
        if not gemini_select.disabled and gemini_select.value not in (self.MODEL_LOADING, self.MODEL_ERROR):
            self.settings.gemini_model = gemini_select.value

        self.settings.gemini_api_key = self.query_one("#gemini_api_key", Input).value  # empty = use .env
        self.settings.output_dir = self.query_one("#output_dir", Input).value

        persistence.save_general_settings({
            "show_timestamps": self.settings.show_timestamps,
            "auto_scroll": self.settings.auto_scroll,
            "split_transcript_percent": self.settings.split_transcript_percent,
            "use_local_asr": self.settings.use_local_asr,
            "asr_language": self.settings.asr_language,
            "insight_auto_refresh_seconds": self.settings.insight_auto_refresh_seconds,
            "insight_context_minutes": self.settings.insight_context_minutes,
            "use_local_llm": self.settings.use_local_llm,
            "ollama_host": self.settings.ollama_host,
            "ollama_model": self.settings.ollama_model,
            "gemini_model": self.settings.gemini_model,
            "gemini_api_key": self.settings.gemini_api_key or "",
            "output_dir": self.settings.output_dir,
        })

        self.dismiss(True)

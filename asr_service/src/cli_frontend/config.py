"""
CLI frontend configuration settings.

Uses pydantic-settings to load from environment variables with CLI_ prefix.
"""

from pydantic_settings import BaseSettings


class CLISettings(BaseSettings):
    """CLI frontend settings loaded from environment."""

    api_base_url: str = "http://localhost:8000"
    output_dir: str = "~/.meeting_scribe/meetings/"
    auto_scroll: bool = True
    show_timestamps: bool = True

    # Recording layout: transcript vs BYT pane width (percent for transcript, rest is BYT)
    split_transcript_percent: int = 70  # 30–90; 70 = 70% transcript, 30% BYT
    show_backchannels: bool = False  # "uh-huh", "yeah"

    # Device selection persistence
    last_mic_device_index: int = 0
    last_system_audio_device_index: int = -1

    # LLM insights service
    insights_service_url: str = "http://localhost:8001"
    insight_auto_refresh_seconds: int = 60  # 0 = disabled
    insight_context_minutes: int = 5

    # Audio engine
    use_local_asr: bool = True  # mlx-whisper (requires Apple Silicon)
    asr_language: str = "auto"  # "auto" for auto-detect, or language code like "en", "pl"

    # LLM intelligence backend
    use_local_llm: bool = True  # True = Ollama, False = Gemini
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b-it-qat"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""

    class Config:
        env_prefix = "CLI_"
        extra = "ignore"


# Global settings instance (env defaults, then overridden by persisted values)
settings = CLISettings()


def _apply_persisted_settings() -> None:
    """Load settings from ~/.meeting_scribe/cli_config.json so all preferences are restored on startup."""
    try:
        from cli_frontend.settings_persistence import persistence  # avoid circular import at top
        saved = persistence.load_general_settings()
        for key, value in saved.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
    except Exception:
        pass


_apply_persisted_settings()

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
    show_backchannels: bool = False  # "uh-huh", "yeah"

    # Device selection persistence
    last_mic_device_index: int = 0  # Remember last microphone selection
    last_system_audio_device_index: int = (
        -1
    )  # Remember last system audio (default -1 = ScreenCaptureKit)

    # LLM insights service
    insights_service_url: str = "http://localhost:8001"
    insight_auto_refresh_seconds: int = 60  # 0 = disabled
    insight_context_minutes: int = 5  # default context window for LLM

    class Config:
        env_prefix = "CLI_"
        extra = "ignore"


# Global settings instance
settings = CLISettings()

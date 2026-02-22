"""
CLI frontend configuration settings.

Uses pydantic-settings to load from environment variables with CLI_ prefix.
"""

from pydantic_settings import BaseSettings


class CLISettings(BaseSettings):
    """CLI frontend settings loaded from environment."""

    api_base_url: str = "http://localhost:8000"
    output_dir: str = "~/Documents/MeetingScribe/"
    auto_scroll: bool = True
    show_timestamps: bool = True
    show_backchannels: bool = False  # "uh-huh", "yeah"

    class Config:
        env_prefix = "CLI_"
        extra = "ignore"


# Global settings instance
settings = CLISettings()

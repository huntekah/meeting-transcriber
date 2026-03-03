"""
Tests for settings model label formatting.
"""

from cli_frontend.config import CLISettings
from cli_frontend.models import ModelInfo, ModelProvider
from cli_frontend.screens.settings import SettingsScreen


def test_format_ollama_option_includes_size():
    screen = SettingsScreen(CLISettings())
    model = ModelInfo(id="llama3:8b", provider=ModelProvider.ollama, size_bytes=1024)
    label, value = screen._format_ollama_option(model)

    assert value == "llama3:8b"
    assert "llama3:8b" in label.plain
    assert "(" in label.plain and ")" in label.plain


def test_format_gemini_option_includes_limits():
    screen = SettingsScreen(CLISettings())
    model = ModelInfo(
        id="gemini-2.5-flash",
        provider=ModelProvider.gemini,
        display_name="Gemini 2.5 Flash",
        input_token_limit=123,
        output_token_limit=456,
    )
    label, value = screen._format_gemini_option(model)

    assert value == "gemini-2.5-flash"
    assert "Gemini 2.5 Flash" in label.plain
    assert "gemini-2.5-flash" in label.plain
    assert "in=123" in label.plain
    assert "out=456" in label.plain

"""
Settings screen layout tests.
"""

import pytest
from pathlib import Path
from textual.app import App
from textual.widgets import Input, Select

import cli_frontend
from cli_frontend.config import CLISettings
from cli_frontend.screens.settings import SettingsScreen


class _SettingsApp(App[None]):
    CSS_PATH = Path(cli_frontend.__file__).with_name("app.tcss")

    def __init__(self) -> None:
        super().__init__()
        self._settings = CLISettings()


@pytest.mark.asyncio
async def test_settings_inputs_visible_with_small_terminal():
    """Simulate larger font by shrinking terminal height."""
    app = _SettingsApp()

    async with app.run_test(size=(50, 16)) as pilot:
        await pilot.pause()
        await app.push_screen(SettingsScreen(app._settings))
        await pilot.pause()
        screen = app.screen
        scroll = screen.query_one("#settings_scroll")
        ollama_host = screen.query_one("#ollama_host", Input)
        ollama_model = screen.query_one("#ollama_model", Select)
        gemini_model = screen.query_one("#gemini_model", Select)

        assert scroll.size.height > 0
        assert ollama_host.display is True
        assert ollama_model.display is True
        assert gemini_model.display is True
        assert ollama_host.size.height > 0
        assert ollama_model.size.height > 0
        assert gemini_model.size.height > 0

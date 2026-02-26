"""
Tests for settings persistence — device name-based identity.
"""

import json
import pytest
from cli_frontend.settings_persistence import SettingsPersistence


@pytest.fixture
def persistence(tmp_path):
    """Create a SettingsPersistence instance pointing at a temp directory."""
    p = SettingsPersistence()
    p.config_dir = tmp_path
    p.config_file = tmp_path / "cli_config.json"
    return p


class TestDeviceNamePersistence:
    """Device selection is stored and retrieved by device name, not index."""

    def test_save_and_load_device_names(self, persistence):
        """Round-trip: saved names come back unchanged."""
        persistence.save_device_selection("MacBook Pro Microphone", "System Audio (ScreenCaptureKit)")
        mic_name, system_name = persistence.load_device_selection()

        assert mic_name == "MacBook Pro Microphone"
        assert system_name == "System Audio (ScreenCaptureKit)"

    def test_load_returns_none_none_when_file_missing(self, persistence):
        """No config file → (None, None), no crash."""
        assert not persistence.config_file.exists()
        mic_name, system_name = persistence.load_device_selection()

        assert mic_name is None
        assert system_name is None

    def test_old_index_format_not_returned_as_names(self, persistence):
        """Old config with integer indices does not get returned as device names."""
        persistence.config_file.write_text(
            json.dumps({"mic_device_index": 2, "system_device_index": -1})
        )
        mic_name, system_name = persistence.load_device_selection()

        assert mic_name is None
        assert system_name is None

    def test_save_none_system_saves_and_loads_correctly(self, persistence):
        """Saving None for system audio is valid (no system source)."""
        persistence.save_device_selection("USB Mic", None)
        mic_name, system_name = persistence.load_device_selection()

        assert mic_name == "USB Mic"
        assert system_name is None

    def test_save_overwrites_previous_names(self, persistence):
        """Second save replaces the first."""
        persistence.save_device_selection("Rode NT-USB", "System Audio (ScreenCaptureKit)")
        persistence.save_device_selection("Jabra Evolve2", None)
        mic_name, system_name = persistence.load_device_selection()

        assert mic_name == "Jabra Evolve2"
        assert system_name is None

    def test_save_preserves_other_settings(self, persistence):
        """Saving device names does not clobber unrelated settings like output_dir."""
        persistence.save_output_dir("/home/user/meetings")
        persistence.save_device_selection("My Mic", None)

        raw = json.loads(persistence.config_file.read_text())
        assert raw["output_dir"] == "/home/user/meetings"
        assert raw["mic_device_name"] == "My Mic"

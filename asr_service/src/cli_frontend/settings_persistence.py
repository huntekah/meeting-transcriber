"""
Settings persistence for CLI frontend.

Saves and loads device selections and other user preferences to/from a config file.
"""

import json
from pathlib import Path
from typing import Dict, Any
from .logging import logger


class SettingsPersistence:
    """Manages persistence of user settings to disk."""

    def __init__(self):
        """Initialize settings persistence."""
        # Settings file location: ~/.meeting_scribe/cli_config.json
        self.config_dir = Path.home() / ".meeting_scribe"
        self.config_file = self.config_dir / "cli_config.json"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_device_selection(
        self, mic_device_name: str | None, system_device_name: str | None
    ):
        """
        Save device selection to config file.

        Args:
            mic_device_name: Selected microphone device name (or None)
            system_device_name: Selected system audio device name (or None)
        """
        try:
            # Load existing settings
            settings = self._load_raw_settings()

            # Update device selections by name (robust against index shifts)
            settings["mic_device_name"] = mic_device_name
            settings["system_device_name"] = system_device_name

            # Save to file
            self._save_raw_settings(settings)
            logger.debug(
                f"Saved device selection: mic={mic_device_name}, system={system_device_name}"
            )
        except Exception as e:
            logger.error(f"Failed to save device selection: {e}", exc_info=True)

    def load_device_selection(self) -> tuple[str | None, str | None]:
        """
        Load saved device selection from config file.

        Returns:
            Tuple of (mic_device_name, system_device_name) or (None, None) if not found
        """
        try:
            settings = self._load_raw_settings()
            mic_name = settings.get("mic_device_name")
            system_name = settings.get("system_device_name")
            logger.debug(f"Loaded device selection: mic={mic_name}, system={system_name}")
            return mic_name, system_name
        except Exception as e:
            logger.error(f"Failed to load device selection: {e}")
            return None, None

    def save_output_dir(self, output_dir: str):
        """
        Save output directory preference.

        Args:
            output_dir: Path to output directory
        """
        try:
            settings = self._load_raw_settings()
            settings["output_dir"] = output_dir
            self._save_raw_settings(settings)
            logger.debug(f"Saved output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save output directory: {e}", exc_info=True)

    def load_output_dir(self) -> str | None:
        """
        Load saved output directory.

        Returns:
            Output directory path or None if not found
        """
        try:
            settings = self._load_raw_settings()
            output_dir = settings.get("output_dir")
            logger.debug(f"Loaded output directory: {output_dir}")
            return output_dir
        except Exception as e:
            logger.error(f"Failed to load output directory: {e}")
            return None

    def _load_raw_settings(self) -> Dict[str, Any]:
        """
        Load raw settings from file.

        Returns:
            Dictionary of settings, or empty dict if file doesn't exist
        """
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load settings file, using defaults: {e}")
            return {}

    def _save_raw_settings(self, settings: Dict[str, Any]):
        """
        Save raw settings to file.

        Args:
            settings: Dictionary of settings to save
        """
        with open(self.config_file, "w") as f:
            json.dump(settings, f, indent=2)


# Global persistence instance
persistence = SettingsPersistence()

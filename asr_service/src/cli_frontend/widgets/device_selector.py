"""
Device selector widget for choosing audio input devices.

Wraps Textual's Select widget with AudioDevice-specific formatting.
"""

from textual.widgets import Select
from typing import List, Optional, Tuple
from cli_frontend.models import AudioDevice


class DeviceSelector(Select):
    """Dropdown for audio device selection."""

    def __init__(
        self,
        devices: Optional[List[AudioDevice]] = None,
        label: str = "Select device",
        allow_blank: bool = True,
        **kwargs,
    ):
        """
        Initialize device selector.

        Args:
            devices: List of available devices (can be empty initially)
            label: Prompt label for selector
            allow_blank: Whether to allow no selection
            **kwargs: Additional arguments passed to Select widget (e.g., id, classes)
        """
        # Create options from devices
        if devices:
            options = self._create_options(devices)
        else:
            options = []

        super().__init__(
            options,
            prompt=label,
            allow_blank=allow_blank,
            **kwargs,
        )
        self.devices = devices or []

    def _create_options(self, devices: List[AudioDevice]) -> List[Tuple[str, int]]:
        """
        Create select options from device list.

        Args:
            devices: List of AudioDevice objects

        Returns:
            List of (label, value) tuples for Select widget
        """
        options = []
        for device in devices:
            # Format: "MacBook Pro Microphone (1ch, 48000Hz)"
            default_marker = " [DEFAULT]" if device.is_default else ""
            label = f"{device.name} ({device.channels}ch, {device.sample_rate}Hz){default_marker}"
            options.append((label, device.device_index))
        return options

    def set_devices(self, devices: List[AudioDevice], select_default: bool = True):
        """
        Update device list and refresh options.

        Args:
            devices: New list of devices
            select_default: Whether to auto-select default device
        """
        self.devices = devices
        options = self._create_options(devices)
        self.set_options(options)

        # Auto-select default device if requested
        if select_default:
            for device in devices:
                if device.is_default:
                    self.value = device.device_index
                    break

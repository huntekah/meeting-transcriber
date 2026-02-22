"""
Audio device enumeration service.

Provides functionality to list available audio input devices using sounddevice.
"""

import sounddevice as sd
from typing import List
from ..schemas.devices import AudioDevice
from ..core.logging import logger


def list_audio_devices() -> List[AudioDevice]:
    """
    Enumerate all available audio input devices.

    Returns:
        List of AudioDevice models containing device information

    Example:
        >>> devices = list_audio_devices()
        >>> for device in devices:
        ...     print(f"{device.device_index}: {device.name}")
    """
    try:
        # Query all devices
        all_devices = sd.query_devices()
        default_input_device = sd.default.device[0] if sd.default.device else None

        input_devices = []

        # Filter for input devices only
        for idx, device in enumerate(all_devices):
            # Only include devices with input channels
            if device['max_input_channels'] > 0:
                input_devices.append(
                    AudioDevice(
                        device_index=idx,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=int(device['default_samplerate']),
                        is_default=(idx == default_input_device),
                    )
                )

        logger.info(f"Found {len(input_devices)} audio input devices")
        return input_devices

    except Exception as e:
        logger.error(f"Failed to enumerate audio devices: {e}", exc_info=True)
        return []


def get_device_by_index(device_index: int) -> AudioDevice | None:
    """
    Get device information by index.

    Args:
        device_index: Device index to lookup

    Returns:
        AudioDevice if found, None otherwise
    """
    devices = list_audio_devices()
    for device in devices:
        if device.device_index == device_index:
            return device
    return None


def get_default_input_device() -> AudioDevice | None:
    """
    Get the default input device.

    Returns:
        Default AudioDevice if found, None otherwise
    """
    devices = list_audio_devices()
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None

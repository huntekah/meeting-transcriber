"""
Audio device enumeration service.

Provides functionality to list available audio input devices using sounddevice.
Includes virtual devices like ScreenCaptureKit system audio on macOS 13+.
"""

import sounddevice as sd
import platform
from typing import List
from ..schemas.devices import AudioDevice
from ..core.logging import logger
from ..utils.file_ops import get_project_root


def list_audio_devices() -> List[AudioDevice]:
    """
    Enumerate all available audio input devices.

    Includes:
    - All system sounddevice input devices
    - Virtual ScreenCaptureKit device on macOS 13+ (if available)

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
        logger.debug(f"Queried {len(all_devices)} total devices from sounddevice")
        for idx, device in enumerate(all_devices):
            # Only include devices with input channels
            if device["max_input_channels"] > 0:
                input_devices.append(
                    AudioDevice(
                        device_index=idx,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        sample_rate=int(device["default_samplerate"]),
                        is_default=(idx == default_input_device),
                        source_type="sounddevice",
                    )
                )
                logger.debug(f"  [{idx}] {device['name']} - {device['max_input_channels']}ch, {int(device['default_samplerate'])}Hz")
            else:
                logger.debug(f"  [{idx}] FILTERED OUT: {device['name']} (output-only, {device['max_input_channels']} input channels)")

        logger.info(f"Found {len(input_devices)} sounddevice input devices")

        # Add ScreenCaptureKit device on macOS 13+ if binary available
        is_macos_13_plus = _is_macos_13_or_later()
        is_binary_available = _is_screencapture_binary_available()

        logger.debug(f"ScreenCaptureKit availability check: macOS 13+={is_macos_13_plus}, binary={is_binary_available}")

        if is_macos_13_plus and is_binary_available:
            screencapture_device = AudioDevice(
                device_index=-1,  # Special marker for ScreenCaptureKit (not a real sounddevice index)
                name="System Audio (ScreenCaptureKit)",
                channels=1,  # Always mono
                sample_rate=16000,  # Whisper standard
                is_default=False,
                source_type="screencapture",
            )
            # Insert after microphone, before other devices
            # This makes it the second option in the UI
            if input_devices:
                input_devices.insert(1, screencapture_device)
            else:
                input_devices.append(screencapture_device)
            logger.info("✅ Added ScreenCaptureKit virtual device (macOS 13+, binary available)")
        elif not is_macos_13_plus:
            logger.debug("⏭️  ScreenCaptureKit not available: macOS version < 13.0")
        elif not is_binary_available:
            logger.debug("⏭️  ScreenCaptureKit not available: compiled binary not found")

        # Log final device list
        logger.info(f"Available audio devices ({len(input_devices)} total):")
        for device in input_devices:
            logger.info(f"  [{device.device_index}] {device.name} ({device.channels}ch, {device.sample_rate}Hz) - source_type={device.source_type}")

        return input_devices

    except Exception as e:
        logger.error(f"Failed to enumerate audio devices: {e}", exc_info=True)
        return []


def _is_macos_13_or_later() -> bool:
    """
    Check if running on macOS 13.0 or later.

    Returns:
        True if macOS 13+, False otherwise
    """
    system = platform.system()
    logger.debug(f"Detected OS: {system}")

    if system != "Darwin":
        logger.debug(f"Not macOS (system={system}), ScreenCaptureKit unavailable")
        return False

    try:
        version = platform.mac_ver()[0]
        major_version = int(version.split(".")[0])
        is_13_plus = major_version >= 13
        logger.debug(f"macOS version: {version} (major={major_version}, 13+={is_13_plus})")
        return is_13_plus
    except (IndexError, ValueError):
        logger.warning(f"Could not parse macOS version: {version}")
        return False


def _is_screencapture_binary_available() -> bool:
    """
    Check if the ScreenCaptureKit binary is available (compiled and executable).

    Returns:
        True if binary exists and is executable, False otherwise
    """
    try:
        repo_root = get_project_root()
    except FileNotFoundError as e:
        logger.warning(f"Could not find project root: {e}")
        return False

    binary_path = repo_root / "scripts" / "screencapture_audio"

    exists = binary_path.exists()
    is_file = binary_path.is_file() if exists else False
    available = exists and is_file

    logger.debug(f"ScreenCaptureKit binary path: {binary_path}")
    logger.debug(f"  - exists: {exists}")
    logger.debug(f"  - is_file: {is_file}")
    logger.debug(f"  - available: {available}")

    if not available:
        logger.debug(
            f"ScreenCaptureKit binary not found at {binary_path}\n"
            f"Compile with: swiftc scripts/screencapture_audio.swift -o scripts/screencapture_audio"
        )
    else:
        logger.debug(f"✅ ScreenCaptureKit binary found at {binary_path}")

    return available


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

"""
Device discovery endpoints.

Provides endpoints for listing available audio input devices.
"""

from fastapi import APIRouter
from typing import List

from ....schemas.devices import AudioDevice
from ....services.audio_devices import list_audio_devices

router = APIRouter()


@router.get("/devices", response_model=List[AudioDevice], tags=["devices"])
async def get_audio_devices():
    """
    List all available audio input devices.

    Returns:
        List of AudioDevice models with device information

    Example response:
        [
            {
                "device_index": 0,
                "name": "MacBook Pro Microphone",
                "channels": 1,
                "sample_rate": 48000,
                "is_default": true
            },
            {
                "device_index": 2,
                "name": "External USB Mic",
                "channels": 2,
                "sample_rate": 48000,
                "is_default": false
            }
        ]
    """
    return list_audio_devices()

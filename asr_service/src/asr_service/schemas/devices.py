"""
Audio device schemas.

Pydantic models for audio input device information.
"""

from pydantic import BaseModel, Field


class AudioDevice(BaseModel):
    """
    Audio input device information.

    Attributes:
        device_index: Index of the device in the system
        name: Human-readable device name
        channels: Number of input channels
        sample_rate: Default sample rate in Hz
        is_default: Whether this is the system default input device
        source_type: Type of audio source ("sounddevice" or "screencapture")
    """

    device_index: int = Field(..., description="Device index")
    name: str = Field(..., description="Device name")
    channels: int = Field(..., description="Number of input channels")
    sample_rate: int = Field(..., description="Default sample rate in Hz")
    is_default: bool = Field(False, description="Is default input device")
    source_type: str = Field(
        "sounddevice", description="Audio source type (sounddevice or screencapture)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "device_index": 0,
                "name": "MacBook Pro Microphone",
                "channels": 1,
                "sample_rate": 48000,
                "is_default": True,
                "source_type": "sounddevice",
            }
        }

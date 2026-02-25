"""
Tests for CLI frontend models.

Tests Pydantic model validation and serialization.
"""

from cli_frontend.models import (
    AudioDevice,
    SourceConfig,
    Utterance,
    WSUtteranceMessage,
    WSStateChangeMessage,
    WSErrorMessage,
)


class TestAudioDevice:
    """Test AudioDevice model."""

    def test_valid_device(self):
        """Test creating valid AudioDevice."""
        device = AudioDevice(
            device_index=0,
            name="Test Mic",
            channels=1,
            sample_rate=48000,
            is_default=True,
        )

        assert device.device_index == 0
        assert device.name == "Test Mic"
        assert device.is_default is True

    def test_device_from_dict(self):
        """Test creating AudioDevice from dict."""
        data = {
            "device_index": 1,
            "name": "USB Mic",
            "channels": 2,
            "sample_rate": 44100,
            "is_default": False,
        }

        device = AudioDevice(**data)

        assert device.name == "USB Mic"
        assert device.channels == 2


class TestSourceConfig:
    """Test SourceConfig model."""

    def test_valid_source_config(self):
        """Test creating valid SourceConfig."""
        config = SourceConfig(
            device_index=0,
            device_name="Test Device",
        )

        assert config.device_index == 0
        assert config.device_name == "Test Device"


class TestUtterance:
    """Test Utterance model."""

    def test_valid_utterance(self):
        """Test creating valid Utterance."""
        utterance = Utterance(
            source_id=0,
            start_time=1000.0,
            end_time=1002.5,
            text="Hello world",
        )

        assert utterance.source_id == 0
        assert utterance.text == "Hello world"
        assert utterance.confidence == 1.0  # Default value
        assert utterance.is_final is True  # Default value
        assert utterance.overlaps_with == []  # Default value

    def test_utterance_with_overlaps(self):
        """Test Utterance with overlap markers."""
        utterance = Utterance(
            source_id=0,
            start_time=1.0,
            end_time=2.0,
            text="Overlapping",
            overlaps_with=[1, 2],
        )

        assert len(utterance.overlaps_with) == 2
        assert 1 in utterance.overlaps_with

    def test_utterance_low_confidence(self):
        """Test Utterance with low confidence."""
        utterance = Utterance(
            source_id=0,
            start_time=1.0,
            end_time=2.0,
            text="Uncertain",
            confidence=0.3,
        )

        assert utterance.confidence == 0.3


class TestWebSocketMessages:
    """Test WebSocket message models."""

    def test_utterance_message(self):
        """Test WSUtteranceMessage."""
        utterance = Utterance(
            source_id=0,
            start_time=1.0,
            end_time=2.0,
            text="Test",
        )

        message = WSUtteranceMessage(
            type="utterance",
            data=utterance,
        )

        assert message.type == "utterance"
        assert message.data.text == "Test"

    def test_state_change_message(self):
        """Test WSStateChangeMessage."""
        message = WSStateChangeMessage(
            type="state_change",
            state="recording",
        )

        assert message.type == "state_change"
        assert message.state == "recording"

    def test_error_message(self):
        """Test WSErrorMessage."""
        message = WSErrorMessage(
            type="error",
            message="Connection failed",
            code="CONNECTION_ERROR",
        )

        assert message.type == "error"
        assert message.message == "Connection failed"
        assert message.code == "CONNECTION_ERROR"

    def test_message_serialization(self):
        """Test message can be serialized to JSON."""
        message = WSStateChangeMessage(
            type="state_change",
            state="completed",
        )

        # Should be able to serialize
        data = message.model_dump()

        assert data["type"] == "state_change"
        assert data["state"] == "completed"

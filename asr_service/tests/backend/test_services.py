"""
Tests for backend services.

Tests core service components like SessionManager, ModelManager, etc.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from asr_service.services.session_manager import SessionManager
from asr_service.services.model_manager import ModelManager
from asr_service.core.exceptions import SessionNotFoundError


class TestSessionManager:
    """Test SessionManager singleton."""

    def test_singleton_pattern(self):
        """Test SessionManager is a singleton."""
        manager1 = SessionManager()
        manager2 = SessionManager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a new session."""
        from asr_service.schemas.transcription import SourceConfig
        import uuid

        # Clear singleton
        if hasattr(SessionManager, '_instance'):
            SessionManager._instance = None
        manager = SessionManager()

        # Mock ModelManager with async load_models
        mock_model_manager = MagicMock()
        mock_model_manager.load_models = AsyncMock()
        mock_model_manager.vad_model = MagicMock()
        mock_model_manager.whisper_model_name = "test-model"

        sources = [SourceConfig(device_index=0, device_name="Test")]
        session = await manager.create_session(sources, mock_model_manager)

        # Verify session was created with valid UUID
        assert session.session_id is not None
        try:
            uuid.UUID(session.session_id)  # Validate UUID format
        except ValueError:
            pytest.fail("Session ID is not a valid UUID")

        # Verify session was added to registry
        assert session.session_id in manager._sessions
        assert len(manager._sessions) == 1

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test retrieving an existing session."""
        manager = SessionManager()

        # Clear sessions
        manager._sessions = {}

        # Add a mock session
        mock_session = MagicMock()
        mock_session.session_id = "test-123"
        manager._sessions["test-123"] = mock_session

        session = await manager.get_session("test-123")
        assert session.session_id == "test-123"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        """Test SessionNotFoundError when session doesn't exist."""
        manager = SessionManager()
        manager._sessions = {}

        with pytest.raises(SessionNotFoundError):
            await manager.get_session("nonexistent")

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing all session IDs."""
        manager = SessionManager()
        manager._sessions = {
            "session-1": MagicMock(),
            "session-2": MagicMock(),
        }

        sessions = await manager.list_sessions()
        assert set(sessions) == {"session-1", "session-2"}

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Test deleting a session."""
        manager = SessionManager()
        manager._sessions = {"test-123": MagicMock()}

        await manager.delete_session("test-123")

        assert "test-123" not in manager._sessions


class TestModelManager:
    """Test ModelManager singleton."""

    def test_singleton_pattern(self):
        """Test ModelManager is a singleton."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2

    def test_is_loaded_initially_false(self):
        """Test models are not loaded initially."""
        # Create a fresh instance by clearing singleton
        if hasattr(ModelManager, '_instance'):
            ModelManager._instance = None
        manager = ModelManager()
        # Initially, models should not be loaded
        assert manager._models_loaded is False

    @pytest.mark.asyncio
    async def test_load_models(self):
        """Test loading models (mocked to avoid actual model loading)."""
        # Create a fresh instance
        if hasattr(ModelManager, '_instance'):
            ModelManager._instance = None
        manager = ModelManager()

        with patch("asr_service.services.model_manager.torch") as mock_torch:
            # Mock torch hub load for VAD model
            mock_vad = MagicMock()
            mock_torch.hub.load.return_value = mock_vad

            # Mock the actual sync loading function
            with patch.object(manager, '_load_models_sync') as mock_load_sync:
                await manager.load_models()
                # Verify load was attempted
                mock_load_sync.assert_called_once()


class TestTranscriptMerger:
    """Test ChronologicalMerger."""

    def test_add_utterance(self):
        """Test adding utterances in chronological order."""
        from asr_service.services.transcript_merger import ChronologicalMerger
        from asr_service.schemas.transcription import Utterance

        merger = ChronologicalMerger()

        # Add utterances out of order
        utt1 = Utterance(source_id=0, start_time=1.0, end_time=2.0, text="First")
        utt2 = Utterance(source_id=1, start_time=0.5, end_time=1.5, text="Second")

        merger.add_utterance(utt1)
        merger.add_utterance(utt2)

        # Should be sorted by start_time
        utterances = merger.get_all_utterances()
        assert len(utterances) == 2
        assert utterances[0].text == "Second"  # Earlier start time
        assert utterances[1].text == "First"

    def test_overlap_detection(self):
        """Test overlapping utterance detection."""
        from asr_service.services.transcript_merger import ChronologicalMerger
        from asr_service.schemas.transcription import Utterance

        merger = ChronologicalMerger()

        # Create overlapping utterances
        utt1 = Utterance(source_id=0, start_time=1.0, end_time=3.0, text="First")
        utt2 = Utterance(source_id=1, start_time=2.0, end_time=4.0, text="Second")

        merger.add_utterance(utt1)
        merger.add_utterance(utt2)

        # Check overlaps were detected
        utterances = merger.get_all_utterances()
        assert 1 in utterances[0].overlaps_with  # utt1 overlaps with source 1
        assert 0 in utterances[1].overlaps_with  # utt2 overlaps with source 0

    def test_listener_callback(self):
        """Test listener callbacks are called."""
        from asr_service.services.transcript_merger import ChronologicalMerger
        from asr_service.schemas.transcription import Utterance

        merger = ChronologicalMerger()
        callback_called = []

        def listener(utterance):
            callback_called.append(utterance)

        merger.add_listener(listener)

        utt = Utterance(source_id=0, start_time=1.0, end_time=2.0, text="Test")
        merger.add_utterance(utt)

        assert len(callback_called) == 1
        assert callback_called[0].text == "Test"


class TestAudioDevices:
    """Test audio device discovery."""

    def test_list_audio_devices(self):
        """Test listing audio input devices."""
        from asr_service.services.audio_devices import list_audio_devices

        with patch("asr_service.services.audio_devices.sd.query_devices") as mock_query:
            mock_query.return_value = [
                {
                    "name": "Test Mic",
                    "index": 0,
                    "max_input_channels": 1,
                    "default_samplerate": 48000,
                },
                {
                    "name": "Output Device",
                    "index": 1,
                    "max_input_channels": 0,  # No input channels - should be filtered
                    "default_samplerate": 48000,
                },
            ]

            with patch("asr_service.services.audio_devices.sd.default.device", (0, -1)):
                # Disable ScreenCaptureKit for this test (different machines may have it or not)
                with patch("asr_service.services.audio_devices._is_screencapture_binary_available", return_value=False):
                    devices = list_audio_devices()

                    # Should only include input devices (output device should be filtered)
                    assert len(devices) >= 1

                    # Verify the test mic is included
                    device_names = [d.name for d in devices]
                    assert "Test Mic" in device_names

                    # Verify it's marked as default
                    test_mic = next((d for d in devices if d.name == "Test Mic"), None)
                    assert test_mic is not None
                    assert test_mic.is_default is True

                    # Verify output device is NOT included
                    assert "Output Device" not in device_names

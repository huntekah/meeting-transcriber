"""
Tests for CLI frontend widgets.

Tests StatusBar, TranscriptView, and DeviceSelector widgets.
"""

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from cli_frontend.widgets.status_bar import StatusBar
from cli_frontend.widgets.transcript_view import TranscriptView, LiveTranscriptView
from cli_frontend.widgets.device_selector import DeviceSelector
from cli_frontend.models import AudioDevice, Utterance


class _LiveTranscriptLayoutApp(App):
    """Minimal app to validate LiveTranscriptView layout."""

    def compose(self) -> ComposeResult:
        with Vertical(id="recording_container"):
            yield LiveTranscriptView(id="live_transcript")
            yield TranscriptView(id="transcript")


class TestStatusBar:
    """Test StatusBar widget."""

    def test_initial_state(self):
        """Test status bar initial state."""
        status_bar = StatusBar()

        assert status_bar.is_recording is False
        assert status_bar.start_time is None
        assert status_bar._timer_running is False

    def test_start_recording(self):
        """Test starting recording updates state."""
        from datetime import datetime
        from unittest.mock import patch, MagicMock

        status_bar = StatusBar()

        # Mock app.call_later to avoid NoActiveAppError
        mock_app = MagicMock()
        with patch('textual.message_pump.active_app') as mock_active_app:
            mock_active_app.get.return_value = mock_app

            status_bar.start_recording()

            assert status_bar.is_recording is True
            assert status_bar.start_time is not None
            assert isinstance(status_bar.start_time, datetime)
            assert status_bar._timer_running is True

    def test_stop_recording(self):
        """Test stopping recording updates state."""
        from unittest.mock import patch, MagicMock

        status_bar = StatusBar()

        # Mock app.call_later to avoid NoActiveAppError
        mock_app = MagicMock()
        with patch('textual.message_pump.active_app') as mock_active_app:
            mock_active_app.get.return_value = mock_app

            status_bar.start_recording()
            status_bar.stop_recording()

            assert status_bar.is_recording is False
            assert status_bar._timer_running is False

    def test_set_status(self):
        """Test setting custom status updates internal state."""
        from unittest.mock import patch

        status_bar = StatusBar()

        # Mock update method to avoid app context requirement
        with patch.object(status_bar, 'update'):
            status_bar.set_status("Test Status")

            # Verify update was called with correct message
            status_bar.update.assert_called_once_with("Test Status")


class TestTranscriptView:
    """Test TranscriptView widget."""

    def test_initial_state(self):
        """Test transcript view initial state."""
        view = TranscriptView()

        assert len(view.utterances) == 0

    def test_add_utterance(self):
        """Test adding utterance to transcript."""
        view = TranscriptView()

        utterance = Utterance(
            source_id=0,
            start_time=1000.0,
            end_time=1002.0,
            text="Hello world",
            confidence=0.95,
        )

        view.add_utterance(utterance)

        assert len(view.utterances) == 1
        assert view.utterances[0].text == "Hello world"

    def test_add_utterance_with_overlap(self):
        """Test adding utterance with overlap markers."""
        view = TranscriptView()

        utterance = Utterance(
            source_id=0,
            start_time=1000.0,
            end_time=1002.0,
            text="Overlapping speech",
            confidence=0.95,
            overlaps_with=[1],
        )

        view.add_utterance(utterance)

        assert len(view.utterances) == 1
        assert view.utterances[0].overlaps_with == [1]

    def test_clear_transcript(self):
        """Test clearing transcript."""
        view = TranscriptView()

        # Add some utterances
        view.add_utterance(
            Utterance(source_id=0, start_time=1.0, end_time=2.0, text="Test 1")
        )
        view.add_utterance(
            Utterance(source_id=1, start_time=2.0, end_time=3.0, text="Test 2")
        )

        view.clear_transcript()

        assert len(view.utterances) == 0


class TestDeviceSelector:
    """Test DeviceSelector widget."""

    def test_initial_state_empty(self):
        """Test device selector with no devices."""
        selector = DeviceSelector()

        assert len(selector.devices) == 0

    def test_initial_state_with_devices(self):
        """Test device selector with devices."""
        devices = [
            AudioDevice(
                device_index=0,
                name="Test Mic",
                channels=1,
                sample_rate=48000,
                is_default=True,
            ),
            AudioDevice(
                device_index=1,
                name="Test Speaker",
                channels=2,
                sample_rate=48000,
                is_default=False,
            ),
        ]

        selector = DeviceSelector(devices=devices)

        assert len(selector.devices) == 2

    def test_set_devices(self):
        """Test updating device list."""
        from unittest.mock import patch

        selector = DeviceSelector()

        devices = [
            AudioDevice(
                device_index=0,
                name="New Mic",
                channels=1,
                sample_rate=48000,
                is_default=True,
            ),
        ]

        # Mock set_options to avoid app context requirement
        with patch.object(selector, 'set_options'):
            selector.set_devices(devices, select_default=False)

            assert len(selector.devices) == 1
            assert selector.devices[0].name == "New Mic"

    def test_set_devices_auto_select_default(self):
        """Test auto-selecting default device."""
        from unittest.mock import patch, PropertyMock

        selector = DeviceSelector()

        devices = [
            AudioDevice(
                device_index=0,
                name="Non-default",
                channels=1,
                sample_rate=48000,
                is_default=False,
            ),
            AudioDevice(
                device_index=1,
                name="Default Mic",
                channels=1,
                sample_rate=48000,
                is_default=True,
            ),
        ]

        # Mock set_options and value setter to avoid app context requirement
        with patch.object(selector, 'set_options'):
            with patch.object(type(selector), 'value', new_callable=PropertyMock) as mock_value:
                selector.set_devices(devices, select_default=True)

                # Should store devices
                assert len(selector.devices) == 2

                # Verify value was set to default device index
                mock_value.return_value = 1
                mock_value.assert_called()

                # Verify devices were stored correctly
                default_device = next(d for d in selector.devices if d.is_default)
                assert default_device.device_index == 1


class TestDeviceSelectorNameBased:
    """DeviceSelector pre-selects by device name, not by index."""

    def _make_devices(self):
        return [
            AudioDevice(device_index=0, name="MacBook Pro Microphone", channels=1, sample_rate=48000, is_default=True),
            AudioDevice(device_index=1, name="Jabra Evolve2", channels=1, sample_rate=48000, is_default=False),
            AudioDevice(device_index=-1, name="System Audio (ScreenCaptureKit)", channels=1, sample_rate=16000, is_default=False),
        ]

    def test_set_devices_preselects_by_name(self):
        """When pre_select_name matches a device, that device's index is selected."""
        from unittest.mock import patch, PropertyMock

        selector = DeviceSelector()
        devices = self._make_devices()

        with patch.object(selector, "set_options"):
            with patch.object(type(selector), "value", new_callable=PropertyMock) as mock_value:
                selector.set_devices(devices, select_default=False, pre_select_name="Jabra Evolve2")
                mock_value.assert_called_with(1)

    def test_set_devices_preselects_screencapture_by_name(self):
        """ScreenCaptureKit sentinel (index=-1) is found by name match."""
        from unittest.mock import patch, PropertyMock

        selector = DeviceSelector()
        devices = self._make_devices()

        with patch.object(selector, "set_options"):
            with patch.object(type(selector), "value", new_callable=PropertyMock) as mock_value:
                selector.set_devices(devices, select_default=False, pre_select_name="System Audio (ScreenCaptureKit)")
                mock_value.assert_called_with(-1)

    def test_set_devices_name_not_found_no_crash(self):
        """Unknown name → no selection attempted, no exception raised."""
        from unittest.mock import patch, PropertyMock

        selector = DeviceSelector()
        devices = self._make_devices()

        with patch.object(selector, "set_options"):
            with patch.object(type(selector), "value", new_callable=PropertyMock) as mock_value:
                selector.set_devices(devices, select_default=False, pre_select_name="Ghost Device")
                mock_value.assert_not_called()

    def test_set_devices_name_none_falls_back_to_default(self):
        """pre_select_name=None with select_default=True picks the is_default device."""
        from unittest.mock import patch, PropertyMock

        selector = DeviceSelector()
        devices = self._make_devices()

        with patch.object(selector, "set_options"):
            with patch.object(type(selector), "value", new_callable=PropertyMock) as mock_value:
                selector.set_devices(devices, select_default=True, pre_select_name=None)
                mock_value.assert_called_with(0)  # MacBook Pro Microphone is_default=True


class TestLiveTranscriptView:
    """Test LiveTranscriptView widget."""

    @pytest.mark.asyncio
    async def test_update_and_clear_partial(self):
        """Test updating and clearing partials."""
        app = _LiveTranscriptLayoutApp()

        utterance = Utterance(
            source_id=0,
            start_time=1000.0,
            end_time=1002.0,
            text="Partial text",
            confidence=0.95,
            is_final=False,
        )

        async with app.run_test():
            view = app.query_one("#live_transcript", LiveTranscriptView)
            await view.update_partial(utterance)
            assert 0 in view._partials

            view.clear_partial(utterance.source_id)
            assert 0 not in view._partials

    async def test_layout_mount_does_not_crash(self):
        """Mount LiveTranscriptView in a layout to ensure no crash."""
        app = _LiveTranscriptLayoutApp()
        async with app.run_test():
            assert app.query_one("#live_transcript", LiveTranscriptView)

"""
Status bar widget with recording indicator and timer.

Shows current recording status and elapsed time.
"""

from textual.widgets import Static
from textual import work
from datetime import datetime
import asyncio


class StatusBar(Static):
    """Status bar with recording indicator and timer."""

    def __init__(self, **kwargs):
        super().__init__("Ready", **kwargs)
        self.is_recording = False
        self.start_time = None
        self._timer_running = False

    def start_recording(self):
        """Start recording timer."""
        self.is_recording = True
        self.start_time = datetime.now()
        self._timer_running = True
        self.update_timer()

    def stop_recording(self):
        """Stop recording timer."""
        self.is_recording = False
        self._timer_running = False

    @work(exclusive=True)
    async def update_timer(self):
        """Update timer display every second."""
        while self._timer_running and self.is_recording:
            elapsed = datetime.now() - self.start_time
            # Format as HH:MM:SS
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            timer_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            self.update(f"🎙 Recording • {timer_str}")

            await asyncio.sleep(1)

    def set_status(self, status: str):
        """
        Set custom status message.

        Args:
            status: Status message to display
        """
        self.update(status)

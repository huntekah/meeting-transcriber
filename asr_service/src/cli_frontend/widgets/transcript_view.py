"""
Transcript view widget for displaying live transcription.

Shows utterances in chronological order with source labels, timestamps, and overlap markers.
"""

from textual.widgets import RichLog
from datetime import datetime
from cli_frontend.models import Utterance
from cli_frontend.logging import logger


class TranscriptView(RichLog):
    """Live transcript view with auto-scroll."""

    def __init__(self, **kwargs):
        super().__init__(
            highlight=True,
            markup=True,
            auto_scroll=True,
            max_lines=1000,
            wrap=True,
            **kwargs,
        )
        self.utterances = []

    def add_utterance(self, utterance: Utterance):
        """
        Add new utterance to transcript.

        Args:
            utterance: Utterance to add to transcript
        """
        logger.info(f"TranscriptView.add_utterance called: source={utterance.source_id}, text='{utterance.text[:50]}...'")

        self.utterances.append(utterance)
        logger.debug(f"Utterance appended to list (total: {len(self.utterances)})")

        # Format: [Source 0] 12:34:56 - "Hello world" [overlapping]
        timestamp = datetime.fromtimestamp(utterance.start_time).strftime("%H:%M:%S")

        # Source label with color based on source_id
        source_colors = ["cyan", "green", "yellow", "magenta", "blue"]
        color = source_colors[utterance.source_id % len(source_colors)]
        source_label = f"[{color}][Source {utterance.source_id}][/{color}]"

        # Overlap marker
        overlap_marker = ""
        if utterance.overlaps_with:
            overlap_marker = " [yellow]⚠ [overlapping][/yellow]"

        # Confidence indicator (if low)
        confidence_marker = ""
        if utterance.confidence < 0.7:
            confidence_marker = f" [dim]({utterance.confidence:.2f})[/dim]"

        line = f"{source_label} {timestamp} - {utterance.text}{overlap_marker}{confidence_marker}"
        logger.info(f"Writing line to RichLog: {line}")
        self.write(line)
        logger.info("Line written successfully")

    def clear_transcript(self):
        """Clear all utterances from view."""
        self.utterances = []
        self.clear()

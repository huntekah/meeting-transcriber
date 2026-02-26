"""
Transcript view widget for displaying live transcription.

Shows utterances in chronological order with source labels, timestamps, and overlap markers.
"""

from textual.widgets import RichLog, Static
from datetime import datetime
from cli_frontend.models import Utterance
from cli_frontend.logging import logger


def _format_utterance_line(utterance: Utterance, *, show_partial: bool = False) -> str:
    """Format an utterance for display in the transcript UI."""
    timestamp = datetime.fromtimestamp(utterance.start_time).strftime("%H:%M:%S")

    source_colors = ["cyan", "green", "yellow", "magenta", "blue"]
    color = source_colors[utterance.source_id % len(source_colors)]
    source_label = f"[{color}][Source {utterance.source_id}][/{color}]"

    overlap_marker = ""
    if utterance.overlaps_with:
        overlap_marker = " [yellow]⚠ [overlapping][/yellow]"

    confidence_marker = ""
    if utterance.confidence < 0.7:
        confidence_marker = f" [dim]({utterance.confidence:.2f})[/dim]"

    partial_marker = " [dim]…[/dim]" if show_partial else ""
    return (
        f"{source_label} {timestamp} - {utterance.text}"
        f"{overlap_marker}{confidence_marker}{partial_marker}"
    )


class LiveTranscriptView(Static):
    """Live, updatable view for in-progress utterances."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._partials: dict[int, Utterance] = {}

    def update_partial(self, utterance: Utterance):
        """Update (or add) a partial utterance for a source."""
        self._partials[utterance.source_id] = utterance
        self._refresh_render()

    def clear_partial(self, source_id: int | None = None):
        """Clear one or all partial utterances."""
        if source_id is None:
            self._partials.clear()
        else:
            self._partials.pop(source_id, None)
        self._refresh_render()

    def _refresh_render(self):
        if not self._partials:
            self.update("")
            return

        lines = [
            _format_utterance_line(utterance, show_partial=True)
            for utterance in sorted(self._partials.values(), key=lambda u: u.source_id)
        ]
        self.update("\n".join(lines))


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
        logger.info(
            f"TranscriptView.add_utterance called: source={utterance.source_id}, text='{utterance.text[:50]}...'"
        )

        self.utterances.append(utterance)
        logger.debug(f"Utterance appended to list (total: {len(self.utterances)})")

        line = _format_utterance_line(utterance)
        logger.info(f"Writing line to RichLog: {line}")
        self.write(line)
        logger.info("Line written successfully")

    def clear_transcript(self):
        """Clear all utterances from view."""
        self.utterances = []
        self.clear()

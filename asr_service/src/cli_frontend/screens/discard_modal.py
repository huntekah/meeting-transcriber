"""
Discard confirmation modal.

Shown when the user presses Ctrl+X during recording.
Textual automatically dims the background when a ModalScreen is active.
"""

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Label, Static


class DiscardModal(ModalScreen[bool]):
    """Centered modal asking the user to confirm discarding the recording."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="discard_dialog"):
            yield Static("⚠", id="discard_icon")
            yield Label("Discard this recording?", id="discard_title")
            yield Static(
                "All audio and transcript data for this session will be deleted.",
                id="discard_body",
            )
            with Horizontal(id="discard_buttons"):
                yield Button("Yes, Discard", variant="error", id="discard_yes")
                yield Button("Cancel", variant="default", id="discard_no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "discard_yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

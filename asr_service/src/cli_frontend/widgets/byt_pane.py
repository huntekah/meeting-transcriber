"""
BYT Pane widget — live LLM-powered transcript insights.

"BYT" (Polish: Byt — Being / Entity) shows four insight views fed from the
current session transcript, side-by-side with the existing TranscriptView.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    TabbedContent,
    TabPane,
)

from cli_frontend.models import InsightType
from cli_frontend.logging import logger

if TYPE_CHECKING:
    pass

# Ordered list of insight tabs
INSIGHT_TABS: list[tuple[InsightType, str]] = [
    (InsightType.EMOTION_TRANSLATE, "🎭 Emotion"),
    (InsightType.CODE, "💻 Code"),
    (InsightType.KNOWLEDGE, "🧠 Knowledge"),
    (InsightType.PANIC_MODE, "🚨 Panic"),
]

CONTEXT_OPTIONS: list[tuple[str, float | None]] = [
    ("1 min", 1.0),
    ("3 min", 3.0),
    ("5 min", 5.0),
    ("10 min", 10.0),
    ("30 min", 30.0),
    ("Full session", None),
]

_PLACEHOLDER_MD = "*No insight yet — press ↻ Refresh or switch tabs to generate.*"
_STALE_NOTICE = "\n\n---\n*⚠ Context window changed — press ↻ Refresh to update.*"


class BytPane(Widget):
    """
    BYT insights panel with four sub-tabs and a context window selector.

    Sits to the right of TranscriptView in the RecordingScreen split layout.
    The parent screen is responsible for:
      - calling update_insight(type, markdown) when LLM results arrive
      - calling set_loading(type, bool) around fetch calls

    Messages posted to the app:
      - BytPane.RefreshRequested  — user wants a fresh insight for active tab
    """

    class RefreshRequested(Message):
        """User or timer requested a fresh insight for the given type."""

        def __init__(self, insight_type: InsightType, window_minutes: float | None) -> None:
            super().__init__()
            self.insight_type = insight_type
            self.window_minutes = window_minutes

    # Currently selected context window (None = full session)
    context_window: reactive[float | None] = reactive(5.0)

    def __init__(self, auto_refresh_seconds: int = 60, default_context_minutes: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._auto_refresh_seconds = auto_refresh_seconds
        self._cache: dict[InsightType, str] = {}
        self._stale: set[InsightType] = set()
        self._loading: dict[InsightType, bool] = {t: False for t, _ in INSIGHT_TABS}

        # Set initial context from config default
        default_opt = next(
            (v for _, v in CONTEXT_OPTIONS if v == float(default_context_minutes)),
            5.0,
        )
        self.context_window = default_opt

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="byt_inner"):
            with TabbedContent(id="byt_tabs"):
                for insight_type, label in INSIGHT_TABS:
                    with TabPane(label, id=f"tab_{insight_type.value}"):
                        yield LoadingIndicator(
                            id=f"loading_{insight_type.value}",
                            classes="byt_loading hidden",
                        )
                        yield Markdown(
                            _PLACEHOLDER_MD,
                            id=f"md_{insight_type.value}",
                            classes="byt_markdown",
                        )

            with Horizontal(id="byt_footer"):
                yield Button("↻ Refresh", id="byt_refresh_btn", variant="default")
                yield Label("Context:", id="byt_context_label")
                yield Select(
                    [(label, val) for label, val in CONTEXT_OPTIONS],
                    value=self.context_window,
                    id="byt_context_select",
                    allow_blank=False,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        if self._auto_refresh_seconds > 0:
            self.set_interval(self._auto_refresh_seconds, self._auto_refresh_tick)

    # ------------------------------------------------------------------
    # Public API (called by RecordingScreen)
    # ------------------------------------------------------------------

    def update_insight(self, insight_type: InsightType, markdown: str) -> None:
        """Store and display a new insight result."""
        self._cache[insight_type] = markdown
        self._stale.discard(insight_type)
        self.set_loading(insight_type, False)
        self._refresh_markdown_widget(insight_type)
        logger.info(f"BytPane: insight updated for {insight_type}")

    def set_loading(self, insight_type: InsightType, loading: bool) -> None:
        """Show or hide the loading indicator for a tab."""
        self._loading[insight_type] = loading
        try:
            indicator = self.query_one(f"#loading_{insight_type.value}", LoadingIndicator)
            md = self.query_one(f"#md_{insight_type.value}", Markdown)
            if loading:
                indicator.remove_class("hidden")
                md.add_class("hidden")
            else:
                indicator.add_class("hidden")
                md.remove_class("hidden")
        except Exception:
            pass  # Widget may not be mounted yet

    def mark_stale(self) -> None:
        """Mark the currently visible tab's cache as stale (context window changed)."""
        active = self._active_insight_type()
        if active in self._cache:
            self._stale.add(active)
            self._refresh_markdown_widget(active)

    def cache_is_empty(self, insight_type: InsightType) -> bool:
        """Return True if there is no cached content for this insight type."""
        return insight_type not in self._cache

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "byt_refresh_btn":
            self._request_refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "byt_context_select":
            self.context_window = event.value
            self.mark_stale()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Auto-fetch when switching to a tab with no cached content."""
        active = self._active_insight_type()
        if active is not None and self.cache_is_empty(active):
            self._request_refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_insight_type(self) -> InsightType | None:
        try:
            tabs = self.query_one("#byt_tabs", TabbedContent)
            active_pane_id = str(tabs.active)  # e.g. "tab_emotion_translate"
            for insight_type, _ in INSIGHT_TABS:
                if active_pane_id == f"tab_{insight_type.value}":
                    return insight_type
        except Exception:
            pass
        return None

    def _request_refresh(self) -> None:
        active = self._active_insight_type()
        if active is not None:
            self.post_message(self.RefreshRequested(active, self.context_window))

    def _auto_refresh_tick(self) -> None:
        """Called by set_interval — only refreshes if visible."""
        if not self.display:
            return
        self._request_refresh()

    def _refresh_markdown_widget(self, insight_type: InsightType) -> None:
        try:
            md_widget = self.query_one(f"#md_{insight_type.value}", Markdown)
            content = self._cache.get(insight_type, _PLACEHOLDER_MD)
            if insight_type in self._stale:
                content += _STALE_NOTICE
            md_widget.update(content)
        except Exception:
            pass

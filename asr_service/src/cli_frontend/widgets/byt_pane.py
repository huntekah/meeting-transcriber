"""
BYT Pane widget — live LLM-powered transcript insights.

"BYT" (Polish: Byt — Being / Entity) shows skill-driven insight views fed from
the current session transcript, side-by-side with the existing TranscriptView.

Skills are loaded dynamically from the LLM Intelligence service (GET /skills).
If no skills are available, a setup-guidance message is shown instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    LoadingIndicator,
    Markdown,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from cli_frontend.models import SkillInfo
from cli_frontend.logging import logger

if TYPE_CHECKING:
    pass

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
_NO_SKILLS_MSG = (
    "⚙ **No skills configured.**\n\n"
    "Add skill files to `llm_intelligence/skills/` and start the LLM Intelligence service:\n\n"
    "```\nmake run-insights\n```"
)


def _tab_id(skill_name: str) -> str:
    """Return the TabPane DOM ID for a skill name slug."""
    return f"tab_{skill_name.replace('-', '_')}"


def _loading_id(skill_name: str) -> str:
    return f"loading_{skill_name.replace('-', '_')}"


def _md_id(skill_name: str) -> str:
    return f"md_{skill_name.replace('-', '_')}"


class BytPane(Widget):
    """
    BYT insights panel with dynamic skill sub-tabs and a context window selector.

    Sits to the right of TranscriptView in the RecordingScreen split layout.
    Skills are passed in at construction time (fetched via InsightsClient.get_skills).
    If *skills* is empty, shows a "configure skills" guidance message.

    The parent screen is responsible for:
      - calling update_insight(skill_name, markdown) when LLM results arrive
      - calling set_loading(skill_name, bool) around fetch calls

    Messages posted to the app:
      - BytPane.RefreshRequested — user or timer wants a fresh insight for active tab
    """

    class RefreshRequested(Message):
        """User or timer requested a fresh insight for the given skill."""

        def __init__(self, skill_name: str, window_minutes: float | None) -> None:
            super().__init__()
            self.skill_name = skill_name
            self.window_minutes = window_minutes

    # Currently selected context window (None = full session)
    context_window: reactive[float | None] = reactive(5.0)

    def __init__(
        self,
        skills: list[SkillInfo],
        auto_refresh_seconds: int = 60,
        default_context_minutes: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._skills = skills
        self._auto_refresh_seconds = auto_refresh_seconds
        self._cache: dict[str, str] = {}
        self._stale: set[str] = set()
        self._loading: dict[str, bool] = {s.name: False for s in skills}

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
            if not self._skills:
                yield Static(_NO_SKILLS_MSG, id="byt_no_skills", markup=True)
            else:
                with TabbedContent(id="byt_tabs"):
                    for skill in self._skills:
                        with TabPane(skill.display, id=_tab_id(skill.name)):
                            yield LoadingIndicator(
                                id=_loading_id(skill.name),
                                classes="byt_loading hidden",
                            )
                            yield Markdown(
                                _PLACEHOLDER_MD,
                                id=_md_id(skill.name),
                                classes="byt_markdown",
                            )

                with Horizontal(id="byt_footer"):
                    yield Button("↻ Refresh", id="byt_refresh_btn", variant="default")
                yield ProgressBar(
                    total=float(self._auto_refresh_seconds),
                    show_percentage=False,
                    show_eta=False,
                    id="byt_progress",
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        if self._skills and self._auto_refresh_seconds > 0:
            self.set_interval(self._auto_refresh_seconds, self._auto_refresh_tick)
            self.set_interval(1, self._advance_progress)

    async def reload_skills(self, skills: list[SkillInfo]) -> None:
        """
        Replace the current skill set and rebuild the widget tree.

        Called by RecordingScreen after fetching skills from GET /skills.
        Uses Textual's recompose() to re-run compose() with the new skill list.
        Timers are re-established by on_mount() after recompose().
        """
        self._skills = skills
        self._cache.clear()
        self._stale.clear()
        self._loading = {s.name: False for s in skills}
        await self.recompose()

    # ------------------------------------------------------------------
    # Public API (called by RecordingScreen)
    # ------------------------------------------------------------------

    def update_insight(self, skill_name: str, markdown: str) -> None:
        """Store and display a new insight result."""
        self._cache[skill_name] = markdown
        self._stale.discard(skill_name)
        self.set_loading(skill_name, False)
        self._refresh_markdown_widget(skill_name)
        logger.info(f"BytPane: insight updated for skill={skill_name}")

    def set_loading(self, skill_name: str, loading: bool) -> None:
        """Show or hide the loading indicator for a skill tab."""
        self._loading[skill_name] = loading
        try:
            indicator = self.query_one(f"#{_loading_id(skill_name)}", LoadingIndicator)
            md = self.query_one(f"#{_md_id(skill_name)}", Markdown)
            if loading:
                indicator.remove_class("hidden")
                md.add_class("hidden")
            else:
                indicator.add_class("hidden")
                md.remove_class("hidden")
        except Exception:
            pass  # Widget may not be mounted yet

    def mark_stale(self) -> None:
        """Mark the active tab's cache as stale (context window changed)."""
        active = self._active_skill_name()
        if active is not None and active in self._cache:
            self._stale.add(active)
            self._refresh_markdown_widget(active)

    def cache_is_empty(self, skill_name: str) -> bool:
        """Return True if there is no cached content for this skill."""
        return skill_name not in self._cache

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "byt_refresh_btn":
            self._request_refresh()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Auto-fetch when switching to a tab with no cached content."""
        active = self._active_skill_name()
        if active is not None and self.cache_is_empty(active):
            self._request_refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_skill_name(self) -> str | None:
        """Return the skill name slug for the currently active tab, or None."""
        try:
            tabs = self.query_one("#byt_tabs", TabbedContent)
            active_pane_id = str(tabs.active)  # e.g. "tab_emotion_translate"
            for skill in self._skills:
                if active_pane_id == _tab_id(skill.name):
                    return skill.name
        except Exception:
            pass
        return None

    def _request_refresh(self) -> None:
        active = self._active_skill_name()
        if active is not None:
            self._reset_progress()
            self.post_message(self.RefreshRequested(active, self.context_window))

    def _auto_refresh_tick(self) -> None:
        """Called by set_interval — only refreshes if visible."""
        if not self.display:
            return
        self._reset_progress()
        self._request_refresh()

    def _advance_progress(self) -> None:
        """Tick the progress bar forward by 1 second."""
        try:
            self.query_one("#byt_progress", ProgressBar).advance(1)
        except Exception:
            pass

    def _reset_progress(self) -> None:
        """Reset the progress bar to zero (after a refresh fires)."""
        try:
            self.query_one("#byt_progress", ProgressBar).update(progress=0)
        except Exception:
            pass

    def _refresh_markdown_widget(self, skill_name: str) -> None:
        try:
            md_widget = self.query_one(f"#{_md_id(skill_name)}", Markdown)
            content = self._cache.get(skill_name, _PLACEHOLDER_MD)
            if skill_name in self._stale:
                content += _STALE_NOTICE
            md_widget.update(content)
        except Exception:
            pass


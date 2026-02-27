# BYT Pane — Transcript Insights Feature Plan

## Problem Statement

During active recordings the user sees a raw live transcript (`TranscriptView` RichLog + `LiveTranscriptView` provisional bar). We want to replace that static log with an intelligent **BYT** ("Byt" — Polish for *Being/Entity*) panel that shows LLM-generated insights organised across 4 content tabs, while keeping the live provisional transcript bar at the bottom.

---

## Scope

1. **Frontend (Textual)** — new BYT pane in RecordingScreen
2. **LLM Insights Service** — standalone FastAPI microservice
3. **Connection architecture** — how frontend + insights service are wired

---

## Architecture Decision: Connection Strategy

Three options considered:

| | Option A — ASR as Orchestrator | Option B — Middle Layer BFF | Option C — Frontend Direct ✅ |
|---|---|---|---|
| ASR changes | Heavy (add LLM caller, new WS msg type) | None | None |
| New services | None | BFF + LLM service | LLM service only |
| Frontend role | Passive receiver | Passive receiver | Active caller |
| Infrastructure | 2 services | 3 services | 2 services |
| Coupling | Tight (ASR ↔ LLM) | Clean | Clean |
| MVP effort | High | Highest | Lowest |

**Recommended: Option C — Frontend Direct**

The Textual frontend already holds the full transcript in memory (`TranscriptView.utterances[]`). It can:
- Collect final utterances from the existing WebSocket
- Call the LLM insights HTTP service directly (async, non-blocking)
- Trigger on tab switch, on demand (button), or on a timer
- Show loading state while awaiting LLM response

Zero changes to ASR service. Two-service topology. Correct for a spike.

> Option B (BFF/Orchestrator) is the recommended production evolution — documented as a future phase.

---

## Component Breakdown

### 1. Frontend — BYT Pane

**RecordingScreen layout — Dynamic Split-Screen:**

```
┌─ RecordingScreen ─────────────────────────────────────────────────┐
│ StatusBar (top dock)                                               │
│                                                                   │
│  ┌─ Horizontal ──────────────────────────────────────────────────┐ │
│  │                                                               │ │
│  │  ┌─ TranscriptView (~70%) ─┐  ┌─ BYT Pane (~30%) ─────────┐ │ │
│  │  │  RichLog (unchanged)    │  │ [🎭][💻][🧠][🚨]  sub-tabs │ │ │
│  │  │                         │  │ ┌───────────────────────┐  │ │ │
│  │  │                         │  │ │   Markdown output     │  │ │ │
│  │  │                         │  │ └───────────────────────┘  │ │ │
│  │  │                         │  │ [↻ Refresh]  ←──●── slider │ │ │
│  │  └─────────────────────────┘  └────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  LiveTranscriptView (100% width, always visible)                  │
│                                                                   │
│  Controls (bottom dock)                                           │
└───────────────────────────────────────────────────────────────────┘
```

**Toggle modes via hotkeys:**

| Mode | Hotkey | Result |
|------|--------|--------|
| Split (default) | — | Both panels visible (70% / 30%) |
| Focus Transcript | `Ctrl+T` | BYT hidden → Transcript expands to 100% |
| Focus BYT | `Ctrl+B` | Transcript hidden → BYT expands to 100% |

Toggling is implemented with `.hidden { display: none; }` CSS class in Textual — the complementary panel reflows to fill available space automatically. `LiveTranscriptView` and `Controls` are always visible regardless of mode.

**New / changed files:**
- `src/cli_frontend/widgets/byt_pane.py` — **NEW**
  - `BytPane(Widget)` — container with `TabbedContent` (4 insight sub-tabs) + footer bar
  - Each sub-tab: `Markdown` widget (Textual's built-in; renders GitHub-flavoured MD) + `LoadingIndicator`
  - `InsightType` enum: `emotion_translate`, `code`, `knowledge`, `panic_mode`
  - State: `_content: dict[InsightType, str]` — cached markdown per tab per session
  - **Context window slider:** `Slider` (or discrete `Select`) in footer to choose transcript window:
    - Values: `1m`, `3m`, `5m`, `10m`, `30m`, `Full` (default from config)
    - Changing value invalidates cache for active tab (stale state indicator)
  - **Refresh button:** `[↻ Refresh]` in footer — refreshes active insight sub-tab
  - Auto-refresh: `BytPane` owns a `set_interval` timer, fires every `insight_auto_refresh_seconds` (0 = off). Timer only refreshes the active sub-tab.
  - On sub-tab switch → auto-fetch if cache is empty for that `InsightType`
  - Exposes `update_insight(type, markdown)` and `set_loading(type, bool)`

- `src/cli_frontend/screens/recording.py` — **MODIFY**
  - Add a `Horizontal` container holding `TranscriptView` (left) and `BytPane` (right)
  - Keep `LiveTranscriptView` outside/below the `Horizontal` (always visible)
  - Add BINDINGS: `ctrl+t` → `action_toggle_transcript`, `ctrl+b` → `action_toggle_byt`
  - `action_toggle_transcript()`: toggles `.hidden` on `TranscriptView`
  - `action_toggle_byt()`: toggles `.hidden` on `BytPane`
  - Accumulate utterances in `self._utterances` list on final WS messages
  - `_fetch_insight(insight_type)` → formats transcript window → calls `InsightsClient` → calls `byt_pane.update_insight()`
  - `_format_transcript_window(minutes)` → slice `self._utterances` by timestamp, format as `[HH:MM:SS] Source N: text`

- `src/cli_frontend/api/insights_client.py` — **NEW**
  - `InsightsClient` (async httpx)
  - `POST /insights` → `{transcript_text: str, insight_type: str}` → `{markdown: str}`
  - Timeout: 60s (LLM can be slow)

- `src/cli_frontend/models.py` — **MODIFY**
  - Add `InsightType(str, Enum)`, `InsightRequest`, `InsightResponse`

- `src/cli_frontend/config.py` — **MODIFY**
  - Add `insights_service_url: str = "http://localhost:8001"`
  - Add `insight_auto_refresh_seconds: int = 60` (0 = disabled)

- `src/cli_frontend/app.tcss` — **MODIFY**
  - Styles for BYT pane, tab bar, loading spinner, Markdown widget

---

### 2. LLM Insights Service

**New top-level package: `src/llm_insights_service/`**

```
src/llm_insights_service/
├── __init__.py
├── main.py          # FastAPI app, /insights endpoint
├── schemas.py       # InsightType enum, InsightRequest, InsightResponse
├── service.py       # LLM call logic (Ollama / OpenAI-compatible HTTP)
└── prompts.py       # System prompt per InsightType (placeholder content)
```

**API contract:**
```
POST /insights
{
  "transcript": "Speaker 0: Hello...\nSpeaker 1: Yes, let's...",
  "insight_type": "emotion_translate" | "code" | "knowledge" | "panic_mode",
  "window_minutes": 10          // optional, for panic_mode
}

→ 200 OK
{
  "markdown": "## Emotional Tone\n...",
  "insight_type": "panic_mode",
  "token_count": 412
}
```

**LLM backend (pluggable):** configured via env var `LLM_PROVIDER`:
- `ollama` (default) — `POST http://localhost:11434/api/generate`
- `openai` — OpenAI-compatible, reads `OPENAI_API_KEY`

**Deployment:** runs as a separate process, e.g. `python -m llm_insights_service` on port 8001. Add a `Makefile` target.

---

## Insight Tab Details

| Tab | Emoji | `insight_type` | Purpose |
|-----|-------|----------------|---------|
| Emotion Translate | 🎭 | `emotion_translate` | Detect emotional tones, translate between communication styles |
| Code | 💻 | `code` | Surface code snippets, tech decisions, debug traces |
| Knowledge | 🧠 | `knowledge` | Key facts, named entities, action items, Q&A |
| Panic Mode | 🚨 | `panic_mode` | Summarize the last 10 minutes of transcript |

System prompt content is a deliberate TODO — the LLM service structure supports it but prompts are placeholder.

---

## Refresh Triggers (Frontend)

- **On BYT sub-tab switch** — if cache is empty for that insight type → auto-fetch
- **Manual** — "↻ Refresh" button in BYT pane footer (refreshes active sub-tab)
- **Auto-timer** (opt-in) — every `insight_auto_refresh_seconds` seconds (configurable, default 60, 0=off). Timer only fires when BYT tab is active.
- **On recording stop** — trigger one final refresh of the active insight tab

Loading state: each sub-tab shows `LoadingIndicator` while awaiting LLM. Cache is per-tab per session; switching insight sub-tabs doesn't re-fetch if already loaded unless context window has changed.

---

## Transcript Context Window (Slider)

A slider in the BYT pane footer lets the user choose how much transcript history to send to the LLM:

| Label | Meaning |
|-------|---------|
| 1m | Last 1 minute of utterances |
| 3m | Last 3 minutes |
| 5m | Last 5 minutes (default) |
| 10m | Last 10 minutes |
| 30m | Last 30 minutes |
| Full | All utterances from session start |

- Changing the slider value invalidates the current cache for the active tab (prompts a refresh notification)
- Default configurable via `insight_context_minutes` in CLISettings (default `5`)

---

## Transcript Formatting for LLM

Frontend formats accumulated utterances as:
```
[HH:MM:SS] Source 0: <text>
[HH:MM:SS] Source 1: <text>
...
```

Utterances are sliced by `start_time` to honour the context window slider value.

---

## Files Summary

| File | Action | Notes |
|------|--------|-------|
| `src/cli_frontend/widgets/byt_pane.py` | CREATE | Core BYT widget with 4 sub-tabs, slider, refresh btn |
| `src/cli_frontend/api/insights_client.py` | CREATE | HTTP client for insights service |
| `src/llm_insights_service/main.py` | CREATE | FastAPI service on port 8001 |
| `src/llm_insights_service/schemas.py` | CREATE | Pydantic models |
| `src/llm_insights_service/service.py` | CREATE | LLM call logic |
| `src/llm_insights_service/prompts.py` | CREATE | System prompts (placeholder) |
| `src/llm_insights_service/__init__.py` | CREATE | Package init |
| `src/cli_frontend/screens/recording.py` | MODIFY | Wrap Transcript + BYT in TabbedContent |
| `src/cli_frontend/models.py` | MODIFY | Add insight models |
| `src/cli_frontend/config.py` | MODIFY | Add insights URL, auto_refresh_seconds, context_minutes |
| `src/cli_frontend/app.tcss` | MODIFY | Styles for outer tabs + BYT sub-tabs + slider |
| `asr_service/Makefile` | MODIFY | Add `make insights` target |

**Not changed:** ASR service backend (session, pipelines, websocket, cold path) — zero modifications.

---

## Out of Scope (Future)

- Option B middle-layer BFF (for production real-time push)
- Actual LLM prompt engineering for each insight type
- Streaming LLM response (markdown progressively rendered)
- Insight history across sessions
- Export insights to markdown file alongside transcript

---

## Open Questions (resolved)

| # | Question | Decision |
|---|----------|----------|
| 1 | Should transcript RichLog be replaced? | No — side-by-side split-screen; hotkeys toggle each panel independently |
| 2 | Tab vs split screen for layout? | Split-screen Horizontal container; `Ctrl+T` hides Transcript, `Ctrl+B` hides BYT |
| 3 | Which LLM backend? | Ollama by default (local, free), OpenAI-compatible as alternative via env |
| 4 | How often to refresh? | On sub-tab-switch if empty + manual button + configurable auto-timer |
| 5 | Connection arch? | Option C (Frontend Direct) for MVP |
| 6 | How much transcript to send? | User-controlled context window slider (1m/3m/5m/10m/30m/Full) in BYT footer |

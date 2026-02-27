# Feature Requests & Implementation Estimates

                                                                                    
1. When choosing audio source user has no idea which sources in his computer actually process audio. Can we Show visual indicator,     of the actual audio loudness being processed? (eg with moving bars, or something). This would make it a lot easier for the user to     actually know which sources he can try.                                                                                            
2. A silence at the beginning of the video is captured as 'Thank you' for some random reason.                                      
5. Ability to change audio sources mid-recording. people can take off headphones, or change microphone. we dont want to force them     for a re-boot of the meeting recording. 
6. Startup health check per audio source: after starting each pipeline, verify audio chunks arrive within ~2s and warn/fail fast if a source produces nothing. Add per-source chunk counts to session stats and transcript metadata. Add live "receiving audio ✓/✗" indicator per source in the CLI frontend during recording. Pipe Swift subprocess stderr into the Python logger so ScreenCaptureKit errors surface immediately instead of silently.
7. Short/soft words don't appear until more speech follows. Single isolated words ("test") in mostly-silent audio produce empty or hallucinated Whisper output which is silently dropped, losing speech context. Fix: roll over on empty/hallucination result instead of dropping audio.

## 1. Visual Audio Loudness Indicator

**Description:**
When choosing audio sources in the setup screen, display real-time audio level meters (moving bars) for each available device. This gives users immediate visual feedback about which sources are actually capturing audio and at what volume levels.

**User Problem:**
Users have no way to know if a selected audio source is actually processing audio until they start recording and play back the transcript. Different devices (built-in mic, USB mic, system audio) may be capturing at very different levels or not at all.

**Suggested Implementation:**
- Add audio level visualization to SetupScreen widget (backend of Textual)
- Start a short background audio stream for each device when entering setup screen
- Use FFT or peak detection to calculate RMS (root mean square) energy
- Display animated progress bars (like a spectrum analyzer) for each device
- Update bars every ~200ms during device selection
- Stop streams when leaving setup screen

**Estimate:**
- **Complexity:** Medium (UI + audio analysis)
- **Backend:** 4-6 hours (background audio sampler, RMS calculation, thread management)
- **Frontend:** 3-4 hours (Textual widget with animated bars, state management)
- **Testing:** 2 hours (test different device types, edge cases like no audio)
- **Total:** 9-12 hours | **2-3 days** (with testing & fixes)

**Risks/Considerations:**
- Need to manage multiple concurrent audio streams safely
- Audio buffering/latency in setup screen
- May need to handle permission issues on some devices

---

## 2. Silence Being Transcribed as 'Thank You'

**Description:**
Silence at the beginning of recordings (or silent segments) is sometimes transcribed as 'Thank you' or other random phrases. This appears to be a Whisper model hallucination issue.

**User Problem:**
Final transcripts contain spurious text that was never actually spoken, cluttering the output and reducing transcript quality.

**Root Cause Analysis:**
Likely due to:
- Whisper's tendency to hallucinate when given silence/background noise
- VAD (voice activity detection) not filtering silent frames before transcription
- Model trained on speech but given non-speech audio

**Suggested Implementation Options:**

**Option A (Quick Fix - Recommended):**
- Post-process transcripts: remove segments with very low confidence scores
- Filter out segments matching common hallucination patterns ("thank you", "you're welcome", etc.)
- Enforce minimum segment duration (e.g., skip segments < 0.5s)
- **Effort:** 3-4 hours

**Option B (Proper Fix):**
- Improve VAD confidence threshold before sending to transcription
- Add silence detection in VADAudioProducer (detect dead silence before transcribing)
- Skip transcription for segments with very low energy levels
- **Effort:** 6-8 hours

**Option C (Advanced):**
- Fine-tune Whisper on domain-specific data
- Use whisper.cpp with better configuration flags
- **Effort:** 20+ hours (requires ML expertise)

**Recommended Approach:** Option A (quick) + Option B (proper)
**Total Estimate:** 9-12 hours | **2-3 days**

**Risks/Considerations:**
- May inadvertently filter legitimate speech if confidence scores are too strict
- Pattern matching could miss edge cases
- Requires tuning thresholds based on real recordings

---

## 5. Change Audio Sources Mid-Recording

**Description:**
Allow users to switch audio sources (microphone, system audio, etc.) during an active recording without stopping and restarting. Enable use cases like taking off headphones, switching to speaker phone, or adding system audio capture mid-call.

**User Problem:**
Users must stop recording, change source, and restart to capture from a different device. This is cumbersome and loses continuous context.

**Technical Complexity: HIGH**
This requires significant architectural changes:
- Session's source pipelines are initialized at creation time
- Audio from multiple start times needs to be combined
- State management for active/inactive sources
- Synchronization between newly added sources and existing ones

**Suggested Implementation:**
1. **Backend Changes:**
   - Modify `Session` to support dynamic pipeline management (add/remove/switch sources)
   - Track source activation times
   - Implement `add_source()`, `remove_source()`, `switch_source()` methods
   - Handle audio sync: buffer new sources, align with existing timeline
   - Update mixer to handle dynamic sources
   - Mark segments with source metadata for transcript clarity

2. **API Endpoint:**
   - Add `POST /api/v1/sessions/{session_id}/sources/add`
   - Add `POST /api/v1/sessions/{session_id}/sources/remove`
   - Add `POST /api/v1/sessions/{session_id}/sources/switch`

3. **Frontend Changes:**
   - Show active/inactive sources during recording (toggle buttons)
   - Allow switching without stopping
   - Visual indicator of source changes in live transcript

4. **Transcript Changes:**
   - Mark source changes in transcript metadata
   - Optionally show "[Source switched to Microphone]" annotations

**Estimate:**
- **Backend refactoring:** 12-16 hours (session management, source lifecycle, audio sync)
- **State management & synchronization:** 8-10 hours (tricky timing/buffering issues)
- **API endpoints & error handling:** 4-5 hours
- **Frontend UI:** 4-6 hours (source toggle, status indicators)
- **Integration testing:** 6-8 hours (complex edge cases, timing issues)
- **Total:** 34-45 hours | **1-2 weeks**

**Risks/Considerations:**
- **High complexity:** Multiple moving pieces that could interact badly
- **Audio sync challenges:** Keeping newly added sources time-synchronized with existing ones
- **State management:** Session state becomes more complex
- **User confusion:** Might create confusing transcripts if not clearly marked
- **Testing effort:** Hard to test all combinations (which source to switch, when, timing)
- **Potential for data loss:** If not handled carefully, could lose audio from transition period

**Recommended Phasing:**
- Phase 1 (easier): Add ability to START recording with a source (don't switch mid-recording)
- Phase 2 (harder): Switch between sources mid-recording
- Phase 3 (complex): Synchronize multiple sources started at different times

---

## Summary Table

| Feature | Effort | Timeline | Complexity | Risk | Priority | Notes |
|---------|--------|----------|-----------|------|----------|-------|
| 1. Audio Loudness Indicator | 9-12 hrs | 2-3 days | Medium | Low | High | Independent |
| 2. Fix Silence Hallucination | 9-12 hrs | 2-3 days | Medium | Medium | High | Independent |
| 5. Dynamic Audio Sources | 34-45 hrs | 1-2 weeks | Very High | High | Low | Independent (complex) |
| 6. Audio Source Health Check | 8-12 hrs | 2 days | Medium | Low | High | Independent |
| 7. Content-Aware Rollover | ~2 hrs | < 1 day | Low | Very Low | High | Independent |
| 8a. BYT Pane — UI Phase | 10-14 hrs | 2-3 days | Medium | Low | High | Depends on 8b API contract |
| 8b. BYT Pane — LLM Service Phase | 8-12 hrs | 2 days | Medium | Medium | High | Independent of 8a |

---

## 6. Audio Source Health Check & Silent Failure Detection

**Description:**
After starting each audio pipeline, verify that audio chunks are actually arriving within ~2 seconds. If a source produces no data, warn or fail fast rather than silently recording nothing. Extend this with per-source visibility throughout the recording.

**User Problem:**
A user can select screen capture + microphone, start recording a meeting, and only discover at the end that screen audio was never captured — because the ScreenCaptureKit subprocess failed silently (permissions issue, binary crash, etc.).

**Suggested Implementation:**

1. **Startup Health Check:**
   - After `pipeline.start()`, poll the audio queue for ~2s
   - If no chunks arrive, log a warning and surface it to the user immediately
   - Optionally: block session start or show a dismissible alert

2. **Per-Source Chunk Counts in Stats & Transcript:**
   - Add chunk counters to `SourcePipeline.get_stats()`
   - Include per-source chunk counts in the saved `transcript.json` metadata
   - Makes it easy to diagnose silent source issues in post

3. **Live CLI Indicator:**
   - Show a "receiving audio ✓ / ✗" status per source in the recording screen
   - Update every ~1s based on recent chunk activity

4. **ScreenCaptureKit Stderr → Logger:**
   - Pipe the Swift subprocess stderr into Python's logger
   - Errors from ScreenCaptureKit (permissions denied, device not found) currently disappear silently

**Estimate:**
- **Health check logic:** 2-3 hours
- **Chunk counters + stats:** 1-2 hours
- **CLI live indicator:** 2-3 hours
- **Subprocess stderr logging:** 1 hour
- **Testing:** 2 hours
- **Total:** 8-11 hours | **2 days**

**Risks/Considerations:**
- Health check polling adds slight startup latency (~2s)
- Some devices may legitimately have silence at start (should not false-alarm)
- ScreenCaptureKit requires macOS Screen Recording permission — needs clear user guidance

---

## 7. Content-Aware Rollover (Short/Soft Speech Context Preservation)

**Description:**
When a finalized audio segment produces an empty transcription or a hallucination-rejected result, roll the audio forward into the next segment instead of silently discarding it. This allows Whisper to accumulate enough speech context across multiple short utterances to produce a confident transcription.

**User Problem:**
Single short words ("test", "yes", "ok") spoken softly or in isolation are followed by 1.44s of silence, causing a semantic-silence commit. The resulting segment is mostly silence (~78% silence for a 0.3s word + 1.44s pause). Whisper either returns empty or hallucinates "." — both paths currently drop the audio. The user sees nothing. Only when subsequent richer speech arrives does text appear, and the earlier words may be lost entirely rather than included.

**Root Cause:**
`_handle_final_segment` in `live_transcriber.py` has two silent-drop paths:
```python
if not result["text"].strip():
    return  # audio lost

if self._is_hallucination(result, audio_duration):
    return  # audio lost
```
The rollover mechanism only activates on audio *duration* < `MIN_VALID_AUDIO_SECONDS`. It does not trigger on failed transcription content.

**Proposed Fix:**
Change both silent-drop paths to roll the audio over, capped at `MAX_UTTERANCE_SECONDS` to prevent unbounded growth:

```python
if not result["text"].strip():
    self._rollover_audio = audio_np[-max_rollover_samples:]
    return

if self._is_hallucination(result, audio_duration):
    self._rollover_audio = audio_np[-max_rollover_samples:]
    return
```

Since `audio_np` at that point already includes any previously rolled-over audio (prepended at the top of the function), this naturally chains: each failed attempt passes its full accumulated context to the next segment.

**Effect:**
```
"test" (soft) → Whisper: "" → ROLLOVER (1.74s saved)
"test" (soft) → prepend 1.74s → 3.48s total → Whisper: "" → ROLLOVER (3.48s saved)
"huh interesting" → prepend 3.48s → 6s → Whisper: "test test huh interesting" ✓
```

**Suggested Implementation:**
- `live_transcriber.py`: `_handle_final_segment` — ~6 lines changed (2 bare `return`s → rollover with cap)
- `live_transcriber.py`: add `_max_rollover_samples` property or compute inline from `MAX_UTTERANCE_SECONDS`
- `tests/backend/test_live_transcriber.py`: ~10 lines of new tests covering rollover-on-empty and rollover-on-hallucination paths

No config changes needed — reuses `MAX_UTTERANCE_SECONDS = 15.0`.

**Estimate:**
- **Backend change:** 1 hour (`_handle_final_segment`, 2 paths + cap logic)
- **Tests:** 1 hour (2 new test cases, update 1-2 existing assertions)
- **Total:** ~2 hours | **< 1 day**

**Risks/Considerations:**
- Rollover cap (15s) prevents memory growth; the `MAX_UTTERANCE_SECONDS` constant already exists for this purpose
- Rolled-over audio includes trailing silence from previous segment; Whisper handles leading silence well so this is acceptable
- Very aggressive speech (never empty/hallucinated) is unaffected — rollover only triggers on failed paths
- May cause a slight increase in final segment audio length for edge cases, but this is intentional

---

## Recommended Implementation Order

1. **Quick Win (< 1 day):**
   - Feature #7: Content-Aware Rollover (tiny change, high impact for soft/short speech)

2. **First Sprint (3-4 days):**
   - Feature #1: Audio Loudness Indicator (improves UX immediately)
   - Feature #2: Fix Silence Hallucination (improves quality)
   - Feature #6: Audio Source Health Check (independent, quick win)

3. **Future (1-2 weeks, lower priority):**
   - Feature #5: Dynamic Audio Sources (high complexity, lower user demand)

---

## Dependencies & Blockers

- **Feature #5 depends on:** Complete refactoring of session/pipeline architecture
- **Feature #6 depends on:** Nothing — fully independent, good candidate for next sprint
- **Feature #7 depends on:** Nothing — fully independent, < 1 day, no risk
- **Feature #1 may impact:** CPU usage during setup (need to monitor)
- **Feature #2 requires:** Dataset of real recordings to tune thresholds

---

## Notes for Implementation

- All features should include comprehensive logging (especially useful with new Loguru setup)
- Consider adding feature flags for gradual rollout
- Test on multiple machines with different hardware (mic quality varies)
- Get user feedback early and often (these are UX improvements)

---

## 8. BYT Pane — Live Transcript Insights (LLM-Powered)

**Description:**
During an active recording, show a live AI insights panel ("BYT" — Polish: *Byt*, meaning *Being / Entity*) side-by-side with the existing transcript. The panel provides four rotating LLM-generated insight views fed from the current session's transcript. A context window slider lets the user control how much history is sent to the model.

**User Problem:**
The raw transcript RichLog is good for reference, but gives no higher-order understanding during the meeting. Users want to immediately see what the code being discussed looks like, what the emotional dynamic is, what knowledge points emerged, or get a quick summary when joining late or returning from distraction — all without leaving the recording screen.

**Layout:**

```
┌─ RecordingScreen ──────────────────────────────────────────────────┐
│ StatusBar (top)                                                    │
│                                                                    │
│ ┌── Horizontal ─────────────────────────────────────────────────┐  │
│ │  ┌─ TranscriptView (~70%) ──┐  ┌─ BYT Pane (~30%) ─────────┐ │  │
│ │  │  RichLog (unchanged)     │  │ [🎭][💻][🧠][🚨]  sub-tabs │ │  │
│ │  │                          │  │ ┌───────────────────────┐  │ │  │
│ │  │                          │  │ │   Markdown output     │  │ │  │
│ │  │                          │  │ └───────────────────────┘  │ │  │
│ │  │                          │  │ [↻ Refresh]  ─●── slider  │ │  │
│ │  └──────────────────────────┘  └────────────────────────────┘ │  │
│ └───────────────────────────────────────────────────────────────┘  │
│  LiveTranscriptView (100% width, always visible)                   │
│  Controls (bottom dock)                                            │
└────────────────────────────────────────────────────────────────────┘
```

**Insight Tabs:**

| Tab | `insight_type` | Purpose |
|-----|----------------|---------|
| 🎭 Emotion Translate | `emotion_translate` | Detect emotional tones, translate between communication styles |
| 💻 Code | `code` | Surface code snippets, technical decisions, debug traces mentioned verbally |
| 🧠 Knowledge | `knowledge` | Key facts, named entities, open questions, action items |
| 🚨 Panic Mode | `panic_mode` | Rolling summary of the selected context window — perfect for joining late |

Each tab renders its response as **GitHub-flavoured Markdown** using Textual's built-in `Markdown` widget.

**Context Window Slider** (in BYT pane footer):

| Label | Utterances sent |
|-------|-----------------|
| 1m | Last 1 minute |
| 3m | Last 3 minutes |
| 5m | Last 5 minutes *(default)* |
| 10m | Last 10 minutes |
| 30m | Last 30 minutes |
| Full | Entire session |

Changing the slider invalidates the cache for the active insight tab and shows a stale-data indicator.

**"Zen Mode" Hotkey Toggles:**

| Hotkey | Effect |
|--------|--------|
| `Ctrl+T` | Toggle Transcript panel visibility (BYT expands to 100% width) |
| `Ctrl+B` | Toggle BYT panel visibility (Transcript expands to 100% width) |

`LiveTranscriptView` (provisional line) remains visible at all times regardless of toggle state.

**Refresh Triggers:**
- Sub-tab switch → auto-fetch if cache is empty for that insight type
- `↻ Refresh` button → refresh active sub-tab immediately
- Auto-timer → every N seconds (configurable `insight_auto_refresh_seconds`, default 60, 0 = off); only fires when BYT pane is visible
- Recording stop → final refresh of the currently visible insight tab

**Architecture — Frontend Direct (Option C):**
The Textual frontend formats the accumulated utterance list into timestamped plain text and calls the `llm-insights-service` directly via HTTP. Zero changes to the ASR backend. Two-service topology.

```
CLI Frontend  ──POST /insights──►  llm-insights-service (port 8001)
     │                                      │
     │ (WebSocket, unchanged)               │ (Ollama / Gemini)
     ▼                                      ▼
ASR Service                           LLM model
```

---

### Phase 8a — UI Phase

**Goal:** Ship the full BYT pane UI against a mock/stub insights service. The frontend can be developed and tested independently of the real LLM.

**Changes:**

| File | Action | Description |
|------|--------|-------------|
| `src/cli_frontend/widgets/byt_pane.py` | **CREATE** | `BytPane` widget: `TabbedContent` with 4 sub-tabs (each: `Markdown` + `LoadingIndicator`), footer with `↻ Refresh` button and context window `Select`/`Slider`. Owns auto-refresh `set_interval` timer. Caches content per `InsightType`. |
| `src/cli_frontend/api/insights_client.py` | **CREATE** | Async `httpx` client: `POST /insights {transcript, insight_type, window_minutes}` → `{markdown}`. 60 s timeout. |
| `src/cli_frontend/models.py` | **MODIFY** | Add `InsightType(str, Enum)`, `InsightRequest`, `InsightResponse`. |
| `src/cli_frontend/config.py` | **MODIFY** | Add `insights_service_url` (default `http://localhost:8001`), `insight_auto_refresh_seconds` (default `60`), `insight_context_minutes` (default `5`). |
| `src/cli_frontend/screens/recording.py` | **MODIFY** | Wrap `TranscriptView` + `BytPane` in `Horizontal`. Keep `LiveTranscriptView` below. Add `ctrl+t` / `ctrl+b` BINDINGS. Accumulate utterances in `self._utterances`. Implement `_fetch_insight(insight_type)` and `_format_transcript_window(minutes)`. |
| `src/cli_frontend/app.tcss` | **MODIFY** | Styles for `Horizontal` split, `BytPane` container, sub-tab bar, `Markdown` content area, `LoadingIndicator`, footer controls, `.hidden` reflow. |

**Estimate:**
- `BytPane` widget (tabs, cache, timer, loading state): 4-5 hrs
- `InsightsClient` + models + config: 1 hr
- `RecordingScreen` wiring (layout, toggle hotkeys, utterance accumulation, fetch logic): 3-4 hrs
- CSS/styling: 2-3 hrs
- Manual integration test against stub/mock: 1 hr
- **Total: 11-14 hrs | 2-3 days**

**Risks:**
- Textual's `Markdown` widget has limited CSS customisation — long code blocks may wrap awkwardly at narrow widths; mitigated by `Ctrl+B` focus mode
- `set_interval` timer must be properly cancelled on screen unmount to avoid leaked callbacks

---

### Phase 8b — LLM Service Phase

**Goal:** Implement the standalone `llm-insights-service` that accepts a transcript + insight type and returns rendered Markdown from an LLM.

**New package: `src/llm_insights_service/`**

```
src/llm_insights_service/
├── __init__.py
├── main.py       # FastAPI app on port 8001; POST /insights, GET /health
├── schemas.py    # InsightType enum, InsightRequest, InsightResponse
├── service.py    # InsightService.generate_insight() — async LLM call
└── prompts.py    # System prompt dict keyed by InsightType (placeholder content)
```

**API contract:**
```
POST /insights
{
  "transcript": "[10:04:01] Source 0: we should refactor...",
  "insight_type": "code",
  "window_minutes": 5        // optional; informational only, slicing done by frontend
}

200 OK
{
  "markdown": "## Code Discussion\n...",
  "insight_type": "code",
  "token_count": 387
}
```

**LLM backend** — pluggable via `LLM_PROVIDER` env var:

| Value | Backend | Auth |
|-------|---------|------|
| `ollama` *(default)* | `POST http://localhost:11434/api/generate` | None |
| `gemini` | Google Gemini API (`gemini-1.5-pro` or `gemini-2.0-flash`) | `GEMINI_API_KEY` env var |

Gemini Pro has a native context window of 1M+ tokens — no transcript truncation guard is needed for that provider. Ollama context limits vary by model; for Ollama a soft truncation cap (configurable, default `32k` chars of transcript) is applied in `service.py` to avoid overflowing smaller local models.

**Other files:**

| File | Action | Description |
|------|--------|-------------|
| `asr_service/Makefile` | **MODIFY** | Add `make insights` target: `uv run python -m llm_insights_service` on port 8001 |

**Estimate:**
- FastAPI app + schemas + health endpoint: 1-2 hrs
- `InsightService` with Ollama integration: 2-3 hrs
- Gemini API backend (`google-generativeai` SDK): 1-2 hrs
- Placeholder system prompts (one per insight type): 1 hr
- Makefile target + README update: 0.5 hr
- Integration test with real Ollama model + Gemini: 1-2 hrs
- **Total: 6.5-10.5 hrs | 1-2 days**

**Risks:**
- Ollama must be running locally; no graceful degradation if not available — mitigated by `GET /health` check the frontend can poll on startup
- Ollama context limits vary by local model; a configurable soft truncation cap (default `32k` chars) is applied for Ollama only
- LLM response quality is entirely prompt-dependent; prompt engineering is out of scope for this phase
- Gemini requires a valid `GEMINI_API_KEY`; no truncation guard needed (1M+ token context window)

**Not changed in either phase:** ASR service backend (sessions, pipelines, WebSocket, cold path).

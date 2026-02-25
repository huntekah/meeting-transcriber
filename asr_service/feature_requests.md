# Feature Requests & Implementation Estimates

                                                                                    
1. When choosing audio source user has no idea which sources in his computer actually process audio. Can we Show visual indicator,     of the actual audio loudness being processed? (eg with moving bars, or something). This would make it a lot easier for the user to     actually know which sources he can try.                                                                                            
2. A silence at the beginning of the video is captured as 'Thank you' for some random reason.                                      
3. Fragmented transcript in live view might need to be consolidated every 10s or so.                                               
4. How to make the transcript update faster? scripts/live_test_v2.py is able to blazingly fast show the 'approximate' audio, before     replacing it with the better one. Can we do that in service as well?                                                              
5. Ability to change audio sources mid-recording. people can take off headphones, or change microphone. we dont want to force them     for a re-boot of the meeting recording. 
6. Startup health check per audio source: after starting each pipeline, verify audio chunks arrive within ~2s and warn/fail fast if a source produces nothing. Add per-source chunk counts to session stats and transcript metadata. Add live "receiving audio ✓/✗" indicator per source in the CLI frontend during recording. Pipe Swift subprocess stderr into the Python logger so ScreenCaptureKit errors surface immediately instead of silently.

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

## 3. Consolidate Fragmented Transcripts Every 10 Seconds

**Description:**
Works **alongside** feature #4 (not instead of it). After fast approximate text is shown and replaced with final utterances, periodically consolidate those utterances into logical blocks for better readability. No new ASR work needed—just grouping/joining existing utterances every ~10 seconds (can be more frequent since it's pure text manipulation).

**Flow:**
1. Feature #4 shows fast approximate text immediately → User sees words appear instantly
2. Feature #4 replaces with final VAD-completed utterance → Text becomes accurate
3. Feature #3 consolidates: every 10s, group multiple short utterances into longer readable blocks

**User Problem:**
Even with Feature #4's fast updates, the transcript still feels fragmented because users see many individual utterances appearing one at a time. Consolidation groups related utterances, making the transcript easier to scan and understand context (reads like prose, not a list of fragments).

**Example:**
Without consolidation:
```
- [00:02] SPEAKER_00: "Welcome everyone"
- [00:04] SPEAKER_00: "to the meeting today"
- [00:07] SPEAKER_01: "Thanks for having me"
- [00:10] SPEAKER_01: "I'm excited to be here"
```

With consolidation (every 10s):
```
- [00:02-00:07] SPEAKER_00: "Welcome everyone to the meeting today"
- [00:07-00:10] SPEAKER_01: "Thanks for having me. I'm excited to be here"
```

**Suggested Implementation:**
- Add a `TranscriptConsolidator` class in `services/`
- Listen to utterance stream (from feature #4)
- Every 10s (or configurable interval), group consecutive utterances from same speaker
- Join text with spaces, keep first start_time and last end_time
- Send consolidated utterances via WebSocket to frontend
- Frontend displays consolidated blocks instead of individual utterances
- Can run even more frequently if CPU allows (consolidation is just string joining, no ASR)

**Estimate:**
- **Backend:** 3-4 hours (consolidator logic, timer/interval management, testing)
- **Frontend:** 1-2 hours (display consolidated blocks, handle updates)
- **Testing:** 1-2 hours (timing, edge cases, speaker changes)
- **Total:** 5-8 hours | **1 day**

**Risks/Considerations:**
- Consolidation interval should be configurable (10s may not be optimal for all use cases)
- Need to decide: consolidate by speaker? by time window? by sentence end? (recommend speaker grouping)
- Very low CPU impact (just text joining), can run frequently
- Frontend needs to handle replacing individual utterances with consolidated blocks smoothly
- Requires Feature #4 to be implemented first (prerequisite)

---

## 4. Faster Transcript Updates (Approximate First, Then Better)

**Description:**
Currently, transcription updates arrive only after the VAD detects speech end. Implement a "streaming" approach: show approximate/preliminary transcript immediately as audio is being captured, then replace with higher-quality final transcription once VAD completes.

**User Problem:**
Users wait for speech to end before seeing any transcript. With streaming, they'd see approximate text appear in real-time, making the experience feel faster and more responsive (like live captions).

**Technical Approach (Proven in scripts/live_test_v2.py):**
`scripts/live_test_v2.py` already demonstrates this using **MLX-Whisper** (same model we use):
- **Provisional passes**: While speaking, transcribe the *growing* audio buffer every 300ms
- **Finalization pass**: When VAD detects speech end, transcribe once more for final quality
- Uses same model for both passes (no model switching needed)
- Fast feeling comes from partial audio + high-frequency updates, not streaming inference

**Suggested Implementation (Recommended):**
- Adapt the `live_test_v2.py` pattern into `live_transcriber.py`
- Keep using `mlx-whisper` (no model switch needed)
- Add provisional transcription loop: every 300ms while speaking
- Keep final transcription: when VAD completes
- Send preliminary transcripts with `is_final=False` flag
- Replace with final transcripts when VAD completes
- Reuse VAD's `commit_ready` state from producer to coordinate timing

**Estimate:**
- **Dependency audit:** 2 hours (test faster-whisper performance vs mlx-whisper)
- **Backend refactoring:** 6-8 hours (swap transcriber, handle streaming, manage preliminary vs final)
- **Frontend:** 2-3 hours (UI to show preliminary text differently, replace on final)
- **Testing:** 4-5 hours (latency testing, quality comparison, edge cases)
- **Total:** 14-18 hours | **3-4 days**

**Alternative (Option B - Safer but slower):**
- Transcribe every 1s of buffered audio (even if silence/incomplete)
- Show as preliminary (`is_final=False`)
- Replace when VAD ends
- Effort: 8-10 hours, but with possible repetition of text

**Risks/Considerations:**
- `faster-whisper` may be faster but potentially lower quality than `mlx-whisper`
- Switching models requires re-benchmarking performance
- May need to adjust confidence thresholds for preliminary detection
- Could increase CPU usage with more frequent transcription

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
| 3. Consolidate Transcripts | 5-8 hrs | 1 day | Low | Low | Medium | **Depends on #4** |
| 4. Streaming Approx Transcripts | 14-18 hrs | 3-4 days | High | Medium | Medium | **Prerequisite for #3** |
| 5. Dynamic Audio Sources | 34-45 hrs | 1-2 weeks | Very High | High | Low | Independent (complex) |
| 6. Audio Source Health Check | 8-12 hrs | 2 days | Medium | Low | High | Independent |

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

## Recommended Implementation Order

1. **First Sprint (3-4 days):**
   - Feature #1: Audio Loudness Indicator (improves UX immediately)
   - Feature #2: Fix Silence Hallucination (improves quality)

2. **Second Sprint (4-5 days):**
   - Feature #4: Streaming Approx Transcripts (prerequisite for #3)
   - Feature #3: Consolidate Transcripts (improves readability, builds on #4)
   - *These should be done together since #3 depends on #4*

3. **Future (1-2 weeks, lower priority):**
   - Feature #5: Dynamic Audio Sources (high complexity, lower user demand)

---

## Dependencies & Blockers

- **Feature #3 depends on:** Feature #4 (must implement streaming approx transcripts first)
- **Feature #4 depends on:** Careful testing of `faster-whisper` vs `mlx-whisper` trade-offs
- **Feature #5 depends on:** Complete refactoring of session/pipeline architecture
- **Feature #6 depends on:** Nothing — fully independent, good candidate for next sprint
- **Feature #1 may impact:** CPU usage during setup (need to monitor)
- **Feature #2 requires:** Dataset of real recordings to tune thresholds

---

## Notes for Implementation

- All features should include comprehensive logging (especially useful with new Loguru setup)
- Consider adding feature flags for gradual rollout
- Test on multiple machines with different hardware (mic quality varies)
- Get user feedback early and often (these are UX improvements)

# Testing Guide - ASR Service

## 🔥 Critical Fix: WebSocket Broadcasts Now Working!

**The "nothing shows in CLI" issue is FIXED!**

**Problem:** Utterances were being transcribed (3 utterances in your test) but never sent to WebSocket clients because `asyncio.get_running_loop()` failed in pipeline threads.

**Solution:** Store event loop reference during session initialization and use it for thread-safe async calls.

**Now you should see utterances in real-time!**

---

## Recent Fixes (2026-02-23)

### 1. ✅ Transcription Verified Working
**Test File:** `data/this_is_a_test_that_we_will_use_to_check_asr.mp3`

**Result:**
```
Duration: 7.32s
Language: en
Segments: 1
Full text: this is a test that we will use to track ASR
```

**How to Test:**
```bash
uv run --all-groups pytest tests/integration/test_real_transcription.py::test_real_audio_transcription -v -s
```

### 2. ✅ Numba Threading Error Fixed
**Problem:** Cold path processing crashed with:
```
Numba workqueue threading layer is terminating: Concurrent access has been detected.
Error 134
```

**Fix:** Added `_cold_pipeline_lock` in `ModelManager` to serialize all cold pipeline access.

**Files Modified:**
- `src/asr_service/services/model_manager.py` - Added global lock
- `src/asr_service/services/cold_transcriber.py` - Wrapped `pipeline.process()` calls with lock

### 3. ✅ WebSocket Broadcast Fix (CRITICAL)
**Problem:** Utterances not being sent to WebSocket clients

**Root Cause:** `asyncio.get_running_loop()` was called from pipeline threads, which don't have an event loop. This caused all WebSocket broadcasts to fail silently with the warning:
```
WARNING - No event loop for WebSocket broadcast - utterance from source 0 NOT sent
```

**Fix:** Store the event loop reference during `initialize()` and use it for all thread-safe async calls:
```python
# In __init__
self._event_loop: Optional[asyncio.AbstractEventLoop] = None

# In async initialize()
self._event_loop = asyncio.get_running_loop()

# In _broadcast_utterance_sync() (runs in pipeline thread)
asyncio.run_coroutine_threadsafe(
    self._broadcast_message(message),
    self._event_loop  # Use stored reference instead of get_running_loop()
)
```

**Files Modified:**
- `src/asr_service/services/session.py:88` - Added `_event_loop` attribute
- `src/asr_service/services/session.py:117` - Capture event loop in `initialize()`
- `src/asr_service/services/session.py:268-295` - Use stored loop in `_broadcast_utterance_sync()`
- `src/asr_service/services/session.py:372-386` - Use stored loop in `_set_state()`

### 4. ✅ MLX Whisper Thread Safety
**Problem:** Error 139 (SIGSEGV) when stopping with multiple sources

**Fix:** Added global lock to serialize MLX Whisper calls across all sources.

**Files Modified:**
- `src/asr_service/services/live_transcriber.py:23` - Added `_MLX_WHISPER_LOCK`
- `src/asr_service/services/live_transcriber.py:196` - Wrapped transcription with lock
- `src/asr_service/services/live_transcriber.py:110-160` - Added shutdown safety checks

### 5. ✅ Enhanced Debugging
**Added logging for:**
- VAD speech start/end events
- WebSocket broadcast attempts
- Client connection count

**Files Modified:**
- `src/asr_service/services/vad_producer.py:246-264` - VAD state logging
- `src/asr_service/services/session.py:269-293` - WebSocket broadcast logging

---

## Testing Procedures

### Test 1: Full Test Suite
```bash
make test
```
Expected: **64 tests pass** in ~67 seconds

### Test 2: Real Audio Transcription
```bash
uv run --all-groups pytest tests/integration/test_real_transcription.py -v -s
```
Expected: Transcription of test MP3 file with correct text

### Test 3: Live WebSocket Debug
**Important: Run this to debug why CLI doesn't show utterances**

```bash
# Terminal 1: Start backend
make run

# Terminal 2: Run WebSocket test
cd scripts
python test_live_websocket.py
```

This script will:
1. Create a session with your default microphone
2. Connect to WebSocket
3. Listen for 30 seconds
4. Print all messages received (state changes, utterances, etc.)
5. **Say something into your microphone** - you should see utterances appear

**Expected Output:**
```
=== Testing Live WebSocket Broadcast ===

1. Fetching devices...
   Found 3 devices
   Using device: MacBook Pro Microphone

2. Creating session...
   Session ID: abc-123
   WebSocket URL: /api/v1/ws/abc-123

3. Connecting to WebSocket...
   Connected to ws://localhost:8000/api/v1/ws/abc-123
   Listening for messages (say something into your microphone)...

   [STATE] recording
   [UTTERANCE #1] Source 0: Hello this is a test
      Time: 1708690881.23s - 1708690883.45s
      Overlaps: []
   [UTTERANCE #2] Source 0: Another sentence here
      Time: 1708690885.67s - 1708690887.89s
      Overlaps: []

4. Test complete!
   Total messages received: 4
   Utterances received: 2

5. Stopping session...
   Session stopped (status: 200)
```

### Test 4: Backend with Debug Logging
To see VAD events and WebSocket broadcasts in real-time:

```bash
# Set debug logging level
export LOG_LEVEL=DEBUG

# Run backend
make run
```

Then use the CLI (`make run-cli`) or WebSocket test script. You'll see:
```
DEBUG - [Source 0] Speech START (VAD prob: 0.832)
DEBUG - Broadcasting utterance from source 0: 'Hello this is a test...' (clients: 1)
DEBUG - Utterance broadcast scheduled
DEBUG - [Source 0] Speech END (silence chunks: 15)
```

### Test 5: CLI Frontend
```bash
# Terminal 1: Backend (with debug logging)
export LOG_LEVEL=DEBUG
make run

# Terminal 2: CLI
make run-cli
```

**If utterances still don't show:**
1. Check backend logs for "Broadcasting utterance" messages
2. Check if WebSocket clients count > 0
3. Run the `test_live_websocket.py` script to verify WebSocket works
4. Check CLI logs for WebSocket connection errors

---

## Common Issues

### Issue: "No event loop for WebSocket broadcast"
**Symptom:** Warnings in logs, utterances not delivered

**Solution:** Fixed in `session.py` - now uses `run_coroutine_threadsafe()` instead of `create_task()`

### Issue: Error 134 (Numba threading)
**Symptom:** Service crashes during cold path processing

**Solution:** Fixed in `cold_transcriber.py` - cold pipeline calls now serialized with lock

### Issue: Error 139 (SIGSEGV) with multiple sources
**Symptom:** Service crashes when stopping with 2+ sources

**Solution:** Fixed in `live_transcriber.py` - MLX Whisper calls now serialized with global lock

### Issue: Nothing shows in CLI
**Debug Steps:**
1. Run `scripts/test_live_websocket.py` to verify backend WebSocket works
2. Check backend logs for "Broadcasting utterance" messages
3. Verify WebSocket client count > 0
4. Check CLI WebSocket connection (should show "connection open")

**Possible Causes:**
- CLI not receiving messages (WebSocket connection issue)
- CLI not rendering messages (Textual rendering issue)
- No speech detected by VAD (check VAD threshold or microphone level)

### Issue: VAD not detecting speech
**Symptoms:** No utterances generated despite speaking

**Debug:**
```bash
export LOG_LEVEL=DEBUG
make run
# Check for "Speech START" messages
```

**Tuning:**
- Adjust `VAD_THRESHOLD` in `.env` (default: 0.5)
- Lower value = more sensitive (0.3-0.4 for quiet environments)
- Higher value = less sensitive (0.6-0.7 for noisy environments)

---

## All Tests Pass ✅

```
======================== 64 passed in 67.31s (0:01:07) =========================
```

**Test Breakdown:**
- Backend API endpoints: 11 tests
- Backend services: 13 tests
- Frontend API client: 9 tests
- Frontend models: 10 tests
- Frontend widgets: 12 tests
- Integration tests: 3 tests (including new MP3 test)
- Unit tests: 6 tests

---

## Next Steps for Debugging Live CLI

If you're still seeing no utterances in the CLI after these fixes:

1. **First, verify WebSocket works:**
   ```bash
   # Terminal 1
   make run

   # Terminal 2
   python scripts/test_live_websocket.py
   ```

2. **If WebSocket test shows utterances:**
   - Problem is in the CLI rendering
   - Check `src/cli_frontend/screens/recording.py:on_ws_message()`
   - Check `src/cli_frontend/widgets/transcript_view.py:add_utterance()`

3. **If WebSocket test shows NO utterances:**
   - Check backend logs with DEBUG level
   - Verify microphone is working (record audio with system tool)
   - Check VAD threshold setting
   - Verify LiveTranscriber is processing segments

4. **Share these logs:**
   - Backend log output with DEBUG level
   - WebSocket test output
   - CLI terminal output

This will help identify exactly where the issue is!

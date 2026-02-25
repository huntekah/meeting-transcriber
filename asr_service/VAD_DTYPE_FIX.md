# VAD Dtype Mismatch Fix - Analysis & Solution

## The Problem

**Error encountered when processing 1-hour meeting:**
```
RuntimeError: expected scalar type Double but found Float
```

This error occurred in the Silero VAD model's STFT (Short-Time Fourier Transform) operation during cold path processing when audio duration exceeded `COLD_PATH_CHUNK_DURATION` (300 seconds).

## Root Cause Analysis

### Why The Error Happened

1. **Audio Format**: `soundfile.read()` returns **float32** audio arrays
2. **VAD Model Dtype**: Silero VAD model was in **float64** (Double) by default
3. **Tensor Creation**: `torch.from_numpy(chunk)` preserves input dtype
4. **Type Mismatch**: VAD model's STFT layer expected float64 but received float32

### Why Tests Didn't Catch It

The bug existed in `_find_silence_chunks()` but was never triggered by tests because:

1. **Short Test Audio**: The test MP3 file was < 300 seconds
2. **Direct Path**: Short audio (< chunk_duration) skipped VAD chunking:
   ```python
   if duration <= chunk_duration:  # Line 130 in cold_transcriber.py
       return self.pipeline.process(str(audio_path))  # Doesn't use VAD!
   ```
3. **Latent Bug**: The VAD chunking code path was untested for long audio

## Solution

### Fix 1: Ensure Audio is float32
```python
# In cold_transcriber.py:process_long_audio()
audio = audio.astype(np.float32)
```

### Fix 2: Convert VAD Model to float32
```python
# In cold_transcriber.py:_ensure_vad_model()
self._vad_model = self._vad_model.float()
```

## Verification of Coherence with Working Scripts

### `live_test_v2.py` (Live transcription - WORKS ✓)
```python
# Line 163: Audio explicitly float32
audio_chunk = indata[:, 0].astype(np.float32)

# Line 170-171: VAD call
audio_tensor = torch.from_numpy(audio_chunk)
vad_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
```

**Our fix is coherent**: Both work with float32 audio + float32 VAD model

### `interactive_test_v2.py` (Cold pipeline - WORKS ✓)
Uses `ColdPathPipeline_MLX` which doesn't directly call VAD for transcription.
Only our `ColdPathPostProcessor._find_silence_chunks()` uses VAD for chunking.

## Changes Made

### 1. Code Fix in `cold_transcriber.py`

**File**: `src/asr_service/services/cold_transcriber.py`

**Change 1 (Line 122-124)**: Ensure audio is float32
```python
# Ensure audio is float32 (Silero VAD requirement)
audio = audio.astype(np.float32)
```

**Change 2 (Line 87-89)**: Convert VAD model to float32
```python
# CRITICAL: Convert VAD model to float32 to match input audio dtype
self._vad_model = self._vad_model.float()
```

### 2. Test Coverage in `test_real_transcription.py`

**File**: `tests/integration/test_real_transcription.py`

**Added Test**: `test_long_audio_with_vad_chunking()`
- Creates synthetic audio longer than `COLD_PATH_CHUNK_DURATION`
- Forces VAD chunking code path
- Verifies VAD processes without dtype errors
- Tests on ~360 seconds of audio (1.2x chunk_duration)

**Result**: All 65 tests pass ✅

## Impact Assessment

### What Was Broken
- Any meeting longer than 300 seconds would fail during cold path processing
- Specifically when audio needed VAD-based chunking

### What Is Fixed
- 1-hour meetings can now be processed without dtype errors
- All long-form audio can be properly chunked and transcribed
- VAD model properly handles float32 audio input

### Backwards Compatibility
- ✅ No breaking changes
- ✅ Short audio (< 300s) still works
- ✅ All existing tests pass
- ✅ Code is coherent with working scripts

## Recovery

Two utility scripts provided to verify and recover your meeting:

1. **`verify_vad_fix.py`**: Quick verification (2-5 min runtime)
   - Tests VAD chunking on 1-hour audio
   - No full transcription, just VAD validation

2. **`recover_meeting.py`**: Full recovery (30-45 min runtime)
   - Complete transcription with Whisper + Diarization
   - Saves both JSON and Markdown transcripts

## Coherence with Working Solutions

✅ **Silero VAD Loading**: Same as `live_test_v2.py`
✅ **Audio Float32**: Consistent with all test scripts
✅ **Cold Path Integration**: Uses same `ColdPathPipeline_MLX` as `interactive_test_v2.py`
✅ **Float32 Model**: Now matches float32 audio for dtype consistency

---

**Status**: ✅ FIXED AND TESTED
- Fix verified on 1-hour meeting file
- All 65 unit/integration tests pass
- Coherent with existing working scripts

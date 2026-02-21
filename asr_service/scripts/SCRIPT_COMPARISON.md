# Script Comparison: Live vs Cold Path

## Quick Summary

| Script | Type | Backend | Model | Use Case |
|--------|------|---------|-------|----------|
| `live_test.py` | 🔴 **Live Streaming** | faster-whisper | large-v3-turbo | Real-time mic transcription with context |
| `live_test_v2.py` | 🔴 **Live Streaming** | mlx-whisper | large-v3-turbo | Real-time mic transcription (2x faster) |
| `interactive_test.py` | ⚫ **Cold Path** | faster-whisper | base (configurable) | Test audio files or mic (record then transcribe) |
| `cold_path_pipeline.py` | ⚫ **Cold Path** | faster-whisper | large-v3-turbo | Production batch processing with diarization |

---

## Detailed Comparison

### 🔴 Live Streaming Scripts (Real-Time)

#### `live_test.py` (faster-whisper)
**Architecture:** Producer/Consumer threads with VAD-based sentence detection

**Models:**
- **Whisper**: `large-v3-turbo` (faster-whisper, CPU)
- **VAD**: Silero VAD (chunk-by-chunk, 32ms)

**Processing:**
- Captures 512-sample chunks (32ms) continuously
- VAD detects speech boundaries (~480ms silence = sentence end)
- **Provisional updates**: Every 300ms while speaking (beam_size=1)
- **Finalization**: Accurate transcription after pause (beam_size=5)
- **Context chaining**: ✅ Last 50 words passed to `initial_prompt`

**Performance:**
- RTF: **0.86x** (faster than real-time)
- Model load: ~4s

**Pros:**
- ✅ Context awareness across sentences
- ✅ Proven stability
- ✅ Clean, accurate output

**Cons:**
- ❌ CPU-only (no GPU acceleration)
- ❌ Slower than MLX version

---

#### `live_test_v2.py` (mlx-whisper)
**Architecture:** Same producer/consumer as v1

**Models:**
- **Whisper**: `mlx-community/whisper-large-v3-turbo` (MLX, M4 Metal)
- **VAD**: Silero VAD (chunk-by-chunk, 32ms)

**Processing:**
- Same VAD-based sentence detection
- **Provisional updates**: Every 300ms while speaking
- **Finalization**: After ~480ms silence
- **Context chaining**: ❌ Disabled (causes hallucination loops)

**Performance:**
- RTF: **0.46x** (2x faster than faster-whisper!)
- Model load: <1s

**Pros:**
- ✅ 2x faster inference (native Metal acceleration)
- ✅ Instant model loading
- ✅ Clean output (with proper settings)

**Cons:**
- ❌ No context chaining (each sentence is independent)
- ❌ `initial_prompt` causes severe hallucinations

**Critical Settings:**
```python
# DO NOT USE initial_prompt with MLX - causes loops!
result = mlx_whisper.transcribe(
    audio_np,
    temperature=0.0,  # Greedy only
    # initial_prompt=None  # MUST be None
)
```

---

### ⚫ Cold Path Scripts (Batch Processing)

#### `interactive_test.py` (Testing Script)
**Architecture:** Record first, then process complete audio

**Models:**
- **Whisper**: `base` (hardcoded, fast for testing)
- **VAD**: Silero VAD (via faster-whisper's built-in)
- **Diarization**: Optional (Pyannote)

**Processing:**
1. **Record**: Capture audio to file OR extract from video
2. **Load models**: After recording completes
3. **Transcribe**: Process complete audio file via `ColdPathPipeline`

**Performance:**
- Model load: ~2s (cached)
- Transcription: ~11s per minute of audio

**Use Cases:**
- Testing first N minutes of a video file
- Quick mic tests (record then transcribe)
- Development/debugging

---

#### `cold_path_pipeline.py` (Production Pipeline)
**Architecture:** Complete post-meeting processing pipeline

**Models:**
- **Whisper**: Configurable (`base`, `small`, `large-v3-turbo`, etc.)
- **VAD**: Silero VAD (via faster-whisper)
- **Diarization**: Pyannote speaker-diarization-3.1 (optional)
- **Alignment**: WhisperX for word-level timestamps (optional)

**Processing:**
1. **Diarization** (optional): Identify speakers
2. **VAD**: Remove silence
3. **Transcription**: Whisper with anti-hallucination
4. **Alignment** (optional): Word-level timestamps + speaker assignment

**Features:**
- ✅ Speaker diarization
- ✅ Word-level alignment
- ✅ Comprehensive anti-hallucination
- ✅ Context preservation
- ✅ Production-ready

**Use Cases:**
- Post-meeting transcription
- Audio file batch processing
- High-accuracy requirements

---

## Key Differences Summary

### Live vs Cold Path

| Feature | Live Streaming | Cold Path |
|---------|---------------|-----------|
| **Latency** | Real-time (300ms updates) | Post-processing only |
| **Input** | Microphone stream | Complete audio file |
| **VAD** | Chunk-by-chunk (32ms) | Batch VAD over full file |
| **Updates** | Provisional + Finalized | Single final result |
| **Threading** | Producer/Consumer | Single-threaded |
| **Use Case** | Live meetings, demos | Post-meeting, archives |
| **Diarization** | ❌ Not implemented | ✅ Optional |
| **Context** | Last 50 words (v1 only) | Full document context |

### faster-whisper vs mlx-whisper

| Feature | faster-whisper | mlx-whisper |
|---------|---------------|-------------|
| **Backend** | CPU (CTranslate2) | M4 Metal (MLX) |
| **RTF** | 0.86x | **0.46x** (2x faster) |
| **Model Load** | ~4s | <1s |
| **Context Chaining** | ✅ Works | ❌ Causes hallucinations |
| **Stability** | ✅ Proven | ⚠️ Needs careful tuning |
| **Hallucination** | ✅ Well-controlled | ⚠️ Sensitive to parameters |

---

## Recommendations

### For Production Live Streaming:
1. **Start with `live_test_v2.py` (mlx-whisper)**
   - 2x faster inference
   - Acceptable quality without context
   - Monitor for accuracy issues

2. **Fallback to `live_test.py` (faster-whisper)**
   - If cross-sentence context is needed
   - If MLX stability issues arise
   - Still faster than real-time

### For Post-Meeting Processing:
- **Use `cold_path_pipeline.py`**
  - Full feature set (diarization, alignment)
  - Best accuracy
  - Not time-critical

---

## Configuration Notes

### Model Names
- **faster-whisper**: `large-v3-turbo`
- **mlx-whisper**: `mlx-community/whisper-large-v3-turbo`

### Environment Variables (.env)
Currently only used by `cold_path_pipeline.py` and `interactive_test.py`:
```bash
WHISPER_MODEL=large-v3-turbo
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
HF_TOKEN=your_token_here
```

**Note:** Live scripts hardcode model names. This should be unified in production (see `service_implementation_tasks.md`).

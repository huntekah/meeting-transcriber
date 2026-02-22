# ASR Testing Scripts

This directory contains experimental scripts for testing and debugging the ASR pipeline.

**📋 See [SCRIPT_COMPARISON.md](SCRIPT_COMPARISON.md) for detailed comparison of all scripts**

---

## Live Streaming Scripts (Real-Time)

### `live_test.py` - Real-time transcription with faster-whisper
Real-time microphone transcription with VAD-based sentence detection and context chaining.

**Usage:**
```bash
python live_test.py
# Speak into microphone
# Press ENTER to stop
```

**Features:**
- 💬 Provisional updates every 300ms while speaking
- ✅ Finalized transcription after pauses (~480ms silence)
- 🔗 Context chaining (last 50 words)
- 📊 Session statistics on exit

**Performance:** RTF 0.86x (faster than real-time)

### `live_test_v2.py` - Real-time transcription with mlx-whisper
Same as v1 but uses MLX for native M4 Metal acceleration (2x faster).

**Usage:**
```bash
python live_test_v2.py
# Speak into microphone
# Press ENTER to stop
```

**Features:**
- ⚡ 2x faster than v1 (RTF 0.46x)
- 🚀 Instant model loading (<1s)
- ❌ No context chaining (causes hallucinations in MLX)

**Note:** Use v1 if cross-sentence context is important, v2 for maximum speed.

---

## Cold Path Scripts (Batch Processing)

### `interactive_test.py` - Batch transcription with faster-whisper
Test the Cold Path pipeline with complete audio files using faster-whisper (CPU).

**Features:**
- ⚡ Parallel diarization + transcription
- 🎯 Full accuracy with context preservation
- 🔧 Configurable from `.env`

### `interactive_test_v2.py` - Batch transcription with mlx-whisper
Same as v1 but uses MLX for native M4 Metal acceleration (**2x faster**).

**Features:**
- ⚡⚡ Parallel diarization + transcription (MLX Metal)
- 🚀 ~8x faster than original cold path
- ⚠️ No context chaining (MLX limitation)

---

## Interactive Testing Script

`interactive_test.py` - Test the Cold Path pipeline with different inputs:

### Usage

```bash
# Test first N minutes of the meeting recording
python interactive_test.py --minutes 1   # Test 1 minute
python interactive_test.py --minutes 5   # Test 5 minutes

# Record from microphone and transcribe
python interactive_test.py --mic
```

### How It Works

The script extracts audio segments from `/data/full_meeting_audio.mp3` and processes them through the Cold Path pipeline.

**Processing Pipeline:**
1. Audio extraction (ffmpeg)
2. Model loading (Whisper + VAD)
3. Transcription with VAD filtering
4. Output with timestamps

**Performance (1 minute test):**
- Audio extraction: ~0.5s
- Model loading: ~2s (cached)
- Transcription: ~11s
- **Total: ~13s**

## Cold Path Pipeline

`cold_path_pipeline.py` - Production-quality transcription pipeline

### Features

- **Speaker Diarization** (Pyannote.audio 3.1+) - optional
- **Voice Activity Detection** (Silero VAD via faster-whisper)
- **ASR Transcription** (Whisper base/small/large-v3-turbo)
- **Word-level Alignment** (WhisperX)
- **Hallucination Prevention**:
  - `condition_on_previous_text=False` - prevents loops
  - Temperature fallback - retries with creativity if stuck
  - Compression ratio threshold - detects repetitive garbage
  - Log probability threshold - discards low-confidence segments

### Usage

```python
from cold_path_pipeline import ColdPathPipeline

# Initialize pipeline
pipeline = ColdPathPipeline(
    whisper_model="base",        # or "small", "medium", "large-v3", "large-v3-turbo"
    use_diarization=False,       # Enable for speaker identification
    hf_token=None,              # Required for diarization (HF_TOKEN env var)
    verbose=True
)

# Process audio file
result = pipeline.process("audio.mp3", language="en")

# Print formatted transcript
print(pipeline.format_transcript(result))

# Access segments
for segment in result['segments']:
    print(f"[{segment['start']:.1f}s] {segment['text']}")
```

### Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `base` | ⚡⚡⚡ | ⭐⭐ | Testing, development |
| `small` | ⚡⚡ | ⭐⭐⭐ | General purpose |
| `medium` | ⚡ | ⭐⭐⭐⭐ | High accuracy |
| `large-v3` | 🐌 | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| `large-v3-turbo` | ⚡⚡ | ⭐⭐⭐⭐⭐ | **Recommended** - Best balance |

### Current Configuration

The interactive test script uses **`base` model** for fast testing. To use `large-v3-turbo` for production accuracy, update:

```python
pipeline = ColdPathPipeline(
    whisper_model="large-v3-turbo",  # Change from "base"
    use_diarization=False,
    verbose=True
)
```

## Configuration

All scripts now use **unified configuration** from `.env`:

```bash
# Whisper Models
WHISPER_MODEL=large-v3-turbo  # For faster-whisper
MLX_WHISPER_MODEL=mlx-community/whisper-large-v3-turbo  # For mlx-whisper

# Diarization
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
HF_TOKEN=your_token_here

# Performance
COLD_PATH_PARALLEL_WORKERS=4  # Number of parallel transcription workers
```

Configuration is managed by `scripts/config.py` - no more hardcoded model names!

---

## Performance Improvements

### Cold Path Optimization
Both `cold_path_pipeline.py` and `cold_path_pipeline_v2.py` now include:
- **Parallel Diarization + Transcription**: Run simultaneously (~50% faster)
- **Configurable from .env**: Easy to switch models without code changes

**Expected Performance:**
- `interactive_test.py` (faster-whisper): ~4x faster than original
- `interactive_test_v2.py` (mlx-whisper): ~8x faster than original

---

## Notes

- All test audio files are in `../data/`
- Temporary segments created during testing are auto-cleaned
- The torchcodec warnings are harmless (optional Pyannote dependencies)
- Faster-whisper doesn't support MPS, automatically falls back to CPU
- MLX-Whisper does NOT support context chaining (causes hallucinations)

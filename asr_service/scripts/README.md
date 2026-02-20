# ASR Testing Scripts

This directory contains experimental scripts for testing and debugging the ASR pipeline.

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

## Legacy Scripts

- `test_mlx_whisper.py` - MLX-Whisper testing (replaced by faster-whisper)
- `debug_transcribe.py` - Original debug script
- `transcribe_file.py` - Simple HTTP API test script

## Notes

- All test audio files are in `../data/`
- Temporary segments created during testing are auto-cleaned
- The torchcodec warnings are harmless (optional Pyannote dependencies)
- Faster-whisper doesn't support MPS, automatically falls back to CPU

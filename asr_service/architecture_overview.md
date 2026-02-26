# MeetingScribe ASR Service — Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI Frontend (Textual)                                             │
│  src/cli_frontend/                                                  │
│                                                                     │
│  SetupScreen → RecordingScreen → (results)                          │
│  ├── TranscriptView (RichLog)     ← final utterances               │
│  └── LiveTranscriptView (Vertical) ← provisional per-source lines  │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ WebSocket (ws://localhost:8000)
                    │ + REST API (POST/GET sessions)
┌───────────────────▼─────────────────────────────────────────────────┐
│  FastAPI Backend  (src/asr_service/)                                │
│                                                                     │
│  ActiveSession                                                      │
│  ├── SourcePipeline × N  (one per source)                           │
│  │   ├── Producer (VADAudioProducer | ScreenCaptureAudioProducer)   │
│  │   │   └── VADStreamingBuffer (shared state machine)             │
│  │   └── LiveTranscriber (thread)                                   │
│  ├── ChronologicalMerger (fan-in)                                   │
│  ├── AudioMixer (mix to mono for cold path)                         │
│  └── ColdPathPostProcessor (runs in background after stop)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Full Session Flow: Two-Source Recording

### 1. Source Selection (Setup Screen)

User picks sources in the Textual UI:
- **Microphone** → `source_type = "sounddevice"`, `device_index = N`
- **System Audio** → `source_type = "screencapture"`, `device_name = "System Audio (ScreenCaptureKit)"`

### 2. Session Initialization (`POST /sessions`)

```
ActiveSession.initialize()
  ├── ModelManager.load_models()          # Loads Silero VAD + warms up MLX Whisper
  ├── _create_producer(0, mic_config)     → VADAudioProducer(source_id=0, device_index=N, vad_model=...)
  ├── _create_producer(1, sys_config)     → ScreenCaptureAudioProducer(source_id=1, vad_model=...)
  └── SourcePipeline × 2 (attaches transcriber)
```

**State machine:** `INITIALIZING → RECORDING`

### 3. Recording Starts (`start_recording()`)

Each `SourcePipeline.start()` launches two threads per source:
- **Producer thread** — captures audio, runs VAD
- **LiveTranscriber thread** — transcribes and emits utterances

#### Mic path (VADAudioProducer)
```
sounddevice callback (32ms chunks)
  └── VADStreamingBuffer.append_chunk(512-sample chunk)
        ├── Silero VAD → speech probability
        ├── is_speaking=True  → accumulate in _growing_buffer
        └── silence ≥ 15 chunks → finalize → push to audio_queue
```

#### System audio path (ScreenCaptureAudioProducer)
```
screencapture_audio Swift binary (stdout PCM stream)
  └── raw bytes → _vad_chunk_buffer → slice into 512-sample chunks
        └── VADStreamingBuffer.append_chunk(chunk)
              ├── same VAD state machine as mic
              └── segment pushed to audio_queue on silence
```

### 4. Provisional Transcription (Live Updates)

`LiveTranscriber._streaming_loop()` polls every 50ms:

```
while recording:
  ├── check audio_queue (final segments) → handle immediately
  └── if is_speaking and not commit_ready:
        └── if time_since_last > PROVISIONAL_INTERVAL (0.2s):
              ├── get_streaming_snapshot() → growing buffer snapshot
              ├── if audio ≥ MIN_AUDIO_LENGTH (0.5s):
              │     MLX Whisper (mlx-community/whisper-large-v3-turbo)
              │     → Utterance(is_final=False, source_id=N)
              └── output_callback → merger → WS broadcast
```

**Per-source independence:** Each source has its own `LiveTranscriber` thread, so Source 0 (mic) and Source 1 (system audio) emit provisional updates independently.

**Frontend rendering:** `LiveTranscriptView(Vertical)` holds a `Static` widget per source, identified by `source_id`. Provisional updates overwrite only that source's line.

```
RecordingScreen.on_ws_message
  ├── is_final=False → await live_transcript.update_partial(source_id, text)
  └── is_final=True  → transcript_view.write(text) + live_transcript.clear_partial(source_id)
```

### 5. Finalization (VAD detects silence end)

When `silence_counter ≥ SILENCE_CHUNKS`:
```
VADStreamingBuffer._finalize_segment()
  └── concatenate _growing_buffer → segment dict
        └── audio_queue.put(segment)

LiveTranscriber._streaming_loop
  └── segment = audio_queue.get_nowait()
        ├── _handle_final_segment(segment)
        │     MLX Whisper → Utterance(is_final=True)
        │     → output_callback
        ├── clear_commit_ready()
        └── _last_provisional_text = ""
```

**MLX Whisper lock:** `_MLX_WHISPER_LOCK` serializes all inference calls across both sources (Silero VAD has `_VAD_MODEL_LOCK` for the same reason).

### 6. Stop & Save (`POST /sessions/{id}/stop`)

```
ActiveSession.stop_recording()
  ├── pipeline.stop() × N (in ThreadPoolExecutor)
  │   ├── producer.stop() → drain queue
  │   └── transcriber.stop()
  ├── mixer.mix_to_mono([src0_audio, src1_audio]) → {session_id}_mixed.wav
  ├── mixer.save_multi_channel(...)              → {session_id}_multichannel.wav
  ├── merger.get_all_utterances()                → live_transcript
  └── asyncio.create_task(_run_cold_path_background())   ← non-blocking
```

**State machine:** `RECORDING → STOPPING → PROCESSING`

### 7. Cold Path Post-Processing (Background)

Runs after stop, concurrently with the user starting a new session if desired.

```
ColdPathPostProcessor.process_long_audio(mixed.wav, chunk_duration=300s)
  ├── VAD-based chunking (5-min chunks, 5s overlap)
  └── for each chunk:
        ├── [Parallel] mlx_whisper.transcribe()    → raw segments
        └── [Parallel] pyannote diarization         → speaker turns
              └── whisperx.align()                  → word-level timestamps
                    └── merge diarization + alignment → speaker-labelled segments

→ ColdTranscriptResult.segments saved as:
    {output_dir}/transcript.json
    {output_dir}/transcript.md       ← [HH:MM:SS] SPEAKER_X: text
→ WSFinalTranscriptMessage broadcast to WebSocket clients
```

**State machine:** `PROCESSING → COMPLETED` (or `FAILED`)

### 8. Cancel / Discard (`POST /sessions/{id}/cancel`)

```
ActiveSession.cancel_recording()
  ├── pipeline.stop() × N
  ├── shutil.rmtree(output_dir)      ← discard all audio and transcripts
  └── live_transcript = None
```

**State machine:** `RECORDING/STOPPING → CANCELLED`

---

## Models

| Step | Model | Library | Notes |
|------|-------|---------|-------|
| Voice Activity Detection | **Silero VAD** (v4) | `torch.hub` / `silero-vad` | Runs on every 512-sample chunk (32ms). Serialized via `_VAD_MODEL_LOCK`. |
| Live transcription (provisional + final) | **whisper-large-v3-turbo** | `mlx_whisper` (`mlx-community/whisper-large-v3-turbo`) | Apple Silicon GPU (MLX). Serialized via `_MLX_WHISPER_LOCK`. |
| Cold path transcription | **whisper-large-v3-turbo** | `mlx_whisper` (same model) | Applied per chunk of the full mixed audio. |
| Word-level alignment | **WhisperX alignment model** | `whisperx` | Loaded per-language on demand. |
| Speaker diarization | **pyannote/speaker-diarization-3.1** | `pyannote.audio` | Requires `HF_TOKEN`. Run once per chunk or globally. |

---

## Key Constants (`src/asr_service/core/config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | 16000 Hz | Audio sample rate throughout |
| `CHUNK_SIZE` | 512 samples | VAD granularity = 32ms per chunk |
| `VAD_THRESHOLD` | 0.5 | Silero speech probability threshold |
| `SILENCE_CHUNKS` | 15 | Silence chunks before segment finalization = ~480ms |
| `MIN_AUDIO_LENGTH` | 0.5 s | Minimum audio before attempting transcription |
| `PROVISIONAL_INTERVAL` | 0.2 s | Frequency of provisional transcription updates |
| `SCREENCAPTURE_MAX_DURATION_SECONDS` | 86400 | 24-hour max for Swift binary |
| `COLD_PATH_CHUNK_DURATION` | 300 s | 5-minute chunks for cold path |
| `COLD_PATH_OVERLAP` | 5 s | Overlap between cold path chunks |
| `COLD_PATH_PARALLEL` | True | Run diarization + transcription in parallel |
| `GLOBAL_DIARIZATION_ENABLED` | True | Single diarization pass on full audio |
| `OUTPUT_DIR` | `~/.meeting_scribe/meetings` | Root for session output directories |

---

## Session State Machine

```
INITIALIZING
    │
    ▼
RECORDING ──────────────────────► CANCELLED
    │  (cancel_recording)
    │ (stop_recording)
    ▼
STOPPING
    │
    ▼
PROCESSING   (cold path runs in background asyncio task)
    │
    ├──► COMPLETED
    └──► FAILED
```

---

## Threading Model

```
Event Loop (asyncio)              Worker Threads
─────────────────────────────     ──────────────────────────────────────
FastAPI request handlers          Source 0 Producer (sounddevice callback)
WebSocket broadcast               Source 0 LiveTranscriber
Session state transitions         Source 1 Producer (Swift binary reader)
Cold path executor wrapper        Source 1 LiveTranscriber
                                  Cold path executor (run_in_executor)
```

Cross-thread communication:
- **Producer → Transcriber**: `queue.Queue(maxsize=10)` per source
- **Transcriber → Merger**: direct callback (`_on_utterance`)
- **Merger → WebSocket**: `asyncio.run_coroutine_threadsafe()` with stored event loop

---

## WebSocket Protocol

All messages are JSON with a `type` field:

| Type | Direction | Payload |
|------|-----------|---------|
| `state_change` | Server → Client | `{ state: "recording" \| "processing" \| ... }` |
| `utterance` | Server → Client | `{ data: Utterance }` where `Utterance.is_final` distinguishes provisional vs final |
| `final_transcript` | Server → Client | `{ transcript: ColdTranscriptResult }` |

On connect, server **replays the last 20 utterances** (backlog) then streams live.

---

## Output Files

For each session saved to `OUTPUT_DIR/YYYY-MM-DD-HH-MM/`:

| File | Description |
|------|-------------|
| `{session_id}_mixed.wav` | Mono mix of all sources (cold path input) |
| `{session_id}_multichannel.wav` | Per-source channels for debugging |
| `transcript.json` | Structured `ColdTranscriptResult` with word-level timestamps |
| `transcript.md` | Human-readable `[HH:MM:SS] SPEAKER_X: text` lines |

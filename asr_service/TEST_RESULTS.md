# Language Selector - Test Results

**Date:** 2026-03-02
**Feature:** Language selector for ASR transcription (auto/en/pl)

## ✅ All Tests Passed

### Backend Tests (3/3)

**Test Suite:** `scripts/test_language_options.py`

| Test | Language Value | Result |
|------|----------------|--------|
| Auto-detect | `None` → defaults to `"en"` | ✓ PASS |
| English | `"en"` | ✓ PASS |
| Polish | `"pl"` | ✓ PASS |

**What was tested:**
- SessionConfig accepts language parameter
- SessionManager passes language to ActiveSession
- ActiveSession stores and uses language correctly
- SourcePipeline receives correct language
- Models load successfully for all options

---

### Frontend Tests (3/3)

**Test Suite:** `scripts/test_frontend_language_persistence.py`

| Test | Description | Result |
|------|-------------|--------|
| Default Setting | CLISettings defaults to `"auto"` | ✓ PASS |
| Save & Load | Settings persist to `~/.meeting_scribe/cli_config.json` | ✓ PASS |
| API Payload | Language flows correctly to API (auto→None, en→"en", pl→"pl") | ✓ PASS |

**What was tested:**
- Default `asr_language = "auto"` in CLISettings
- Settings UI saves language selection
- Saved settings load correctly on restart
- API client passes correct language value
- "auto" correctly omits language from payload (allows backend auto-detect)

---

## Implementation Summary

### Data Flow

```
User selects language in Settings UI
         ↓
CLISettings.asr_language = "auto" | "en" | "pl"
         ↓
Saved to ~/.meeting_scribe/cli_config.json
         ↓
Setup screen reads settings.asr_language
         ↓
ASRClient.create_session(language=...)
         ↓
API POST /api/v1/sessions {"language": "en"}
         ↓
SessionConfig(language="en")
         ↓
SessionManager.create_session(language="en")
         ↓
ActiveSession(language="en")
         ↓
SourcePipeline(language="en")
         ↓
LiveTranscriber(language="en")
         ↓
mlx_whisper.transcribe(..., language="en")
```

### Files Modified

#### Backend
- `src/asr_service/schemas/transcription.py` - Added `language` field to SessionConfig
- `src/asr_service/api/v1/endpoints/sessions.py` - Pass language to session creation
- `src/asr_service/services/session_manager.py` - Accept and forward language parameter
- `src/asr_service/services/session.py` - Store language and pass to pipelines

#### Frontend
- `src/cli_frontend/config.py` - Added `asr_language: str = "auto"` setting
- `src/cli_frontend/screens/settings.py` - Added language Select dropdown UI
- `src/cli_frontend/api/client.py` - Pass language to create_session API
- `src/cli_frontend/screens/setup.py` - Use language setting when creating sessions

---

## Verification

To manually verify the feature:

1. **Start the services:**
   ```bash
   # Terminal 1: Backend
   cd asr_service
   uv run uvicorn asr_service.main:app --reload

   # Terminal 2: LLM Intelligence
   cd llm_intelligence
   uv run python -m llm_intelligence.main

   # Terminal 3: Frontend
   cd asr_service
   uv run python -m cli_frontend.main
   ```

2. **Test in UI:**
   - Press `Ctrl+,` to open Settings
   - Find "Audio Engine" section
   - Change "Transcription language" dropdown
   - Click "Save"
   - Start a new recording
   - Verify transcription uses selected language

3. **Verify persistence:**
   ```bash
   cat ~/.meeting_scribe/cli_config.json | grep asr_language
   # Should show: "asr_language": "auto" (or "en" or "pl")
   ```

---

## Language Options

| UI Label | Value Sent to Backend | Behavior |
|----------|----------------------|----------|
| Auto-detect | `None` (omitted from payload) | MLX Whisper auto-detects language |
| English | `"en"` | Forces English transcription |
| Polish | `"pl"` | Forces Polish transcription |

---

## Adding More Languages

To add more languages, edit `src/cli_frontend/screens/settings.py` line 78-87:

```python
yield Select(
    [
        ("Auto-detect", "auto"),
        ("English", "en"),
        ("Polish", "pl"),
        ("Spanish", "es"),      # Add more languages here
        ("French", "fr"),
        ("German", "de"),
    ],
    id="asr_language",
    value=self.settings.asr_language,
    allow_blank=False,
)
```

Use standard ISO 639-1 language codes supported by Whisper.

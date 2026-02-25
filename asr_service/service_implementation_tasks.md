# Service Implementation Tasks

This document tracks issues and improvements needed when developing the production service based on the experimental scripts.

## Configuration Management

### Model Configuration Inconsistency
**Status:** Open
**Priority:** Medium
**Location:** `scripts/live_test.py`, `scripts/live_test_v2.py`

**Issue:**
The test scripts (`live_test.py`, `live_test_v2.py`) hardcode model names instead of reading from `.env` configuration:
- `live_test.py` hardcodes: `whisper_model = load_whisper_model("large-v3-turbo")`
- `live_test_v2.py` hardcodes: `whisper_model = load_whisper_model("mlx-community/whisper-large-v3-turbo")`

Other scripts like `interactive_test.py` and `cold_path_pipeline.py` properly read from settings:
```python
from asr_service.core.config import settings
pipeline = ColdPathPipeline(whisper_model=settings.WHISPER_MODEL, ...)
```

**Action Required:**
When developing the production service, ensure all model configurations are:
1. Read from `.env` / `settings.py`
2. Configurable via environment variables
3. Have sensible defaults
4. Document the difference between faster-whisper model names (`large-v3-turbo`) and mlx-whisper names (`mlx-community/whisper-large-v3-turbo`)

**Impact:**
- Makes it easier to switch models without code changes
- Consistency across the codebase
- Better for deployment and testing

---

## Implementation Notes

### MLX-Whisper: Fast but No Context Chaining
**Status:** ✅ RESOLVED
**Priority:** High
**Location:** `scripts/live_test_v2.py`

**Root Cause:**
MLX-Whisper's `initial_prompt` parameter causes severe hallucination loops in streaming scenarios. The parameter that works in faster-whisper triggers repetitive output in MLX.

**Solution:**
Remove `initial_prompt` entirely and use pure greedy decoding:
```python
result = mlx_whisper.transcribe(
    audio_np,
    path_or_hf_repo=whisper_model_name,
    language="en",
    # DO NOT USE: initial_prompt causes hallucination loops in MLX
    condition_on_previous_text=False,
    temperature=0.0,  # Greedy decoding only
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
)
```

**Final Benchmark Results:**
| Metric | faster-whisper | mlx-whisper |
|--------|----------------|-------------|
| **RTF** | 0.86x | **0.46x** ⚡ |
| **Quality** | ✅ Clean + context | ✅ Clean (no context) |
| **Context chaining** | ✅ Supported | ❌ Not supported |
| **Model load time** | ~4s | <1s |

**Production Decision:**
Two viable options depending on requirements:
1. **faster-whisper**: Use when context chaining is important (better accuracy across sentences)
2. **mlx-whisper**: Use when speed is critical and each utterance is independent (2x faster)

**Recommendation:** Start with **mlx-whisper** for production due to 2x speed improvement. Monitor quality. If cross-sentence context is needed, switch to faster-whisper.

---

## Known Issues (Non-Critical)

### Resource Tracker Warning on Shutdown
**Status:** Open (Low Priority)
**Priority:** Low
**Location:** `scripts/live_test.py`, `scripts/live_test_v2.py`

**Issue:**
On shutdown, you may see this warning:
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown: {'/mp-3lpzbiog'}
```

**Cause:**
PyTorch's `torch.hub.load()` (used for Silero VAD) creates multiprocessing resources that aren't explicitly cleaned up before program exit.

**Impact:**
- **Harmless** - The OS cleans up the semaphores automatically
- No memory leaks or functional issues
- Purely cosmetic warning

**Possible Solutions (if needed):**
1. Add explicit cleanup of torch hub cache on shutdown
2. Use `torch.multiprocessing.set_sharing_strategy('file_system')`
3. Call `torch.cuda.empty_cache()` before exit (if using CUDA)
4. Ignore - this is common with PyTorch-based applications

**Decision:** Low priority. Only address if it becomes a production issue.

---

## Future Tasks

(Add additional tasks here as they come up during development)

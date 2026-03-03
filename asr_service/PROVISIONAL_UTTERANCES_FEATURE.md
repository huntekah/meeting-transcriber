# Provisional Utterances in Insights

**Feature:** Include in-progress (provisional) utterances in LLM insights

**Why:** Sometimes you need an insight *right now* while someone is still talking. Previously, only finalized utterances were included in the transcript sent to the LLM, meaning the current speaker's words weren't considered.

---

## How It Works

### Before (Missing Current Speech)

```
Final utterances only:
[10:30:15] Microphone: We should probably consider the performance implications.
[10:30:22] Microphone: I think we need to benchmark this.

🤔 User requests insight while speaker is still talking...

Currently speaking: "And maybe we should look at the memory usage too..."
❌ This part is NOT included in the insight!
```

### After (Includes Current Speech)

```
Final + provisional utterances:
[10:30:15] Microphone: We should probably consider the performance implications.
[10:30:22] Microphone: I think we need to benchmark this.
[10:30:28] Microphone (speaking...): And maybe we should look at the memory usage too

🤔 User requests insight while speaker is still talking...

✓ Current speech IS included in the insight!
```

---

## Implementation Details

### Changes Made

**File:** `src/cli_frontend/screens/recording.py`

1. **Track provisional utterances** (line 69)
   ```python
   self._provisional_utterances: dict[int, Utterance] = {}
   ```
   Maps `source_id` → latest provisional utterance

2. **Store provisional utterances** (line 157)
   ```python
   elif msg.data.text.strip():
       await live_transcript.update_partial(msg.data)
       # Store/update provisional utterance for this source
       self._provisional_utterances[msg.data.source_id] = msg.data
   ```

3. **Clear on finalization** (line 152)
   ```python
   elif msg.data.is_final:
       self._utterances.append(msg.data)
       # Clear any provisional utterance for this source
       self._provisional_utterances.pop(msg.data.source_id, None)
   ```

4. **Include in formatted transcript** (lines 278-284)
   ```python
   # Add any provisional (in-progress) utterances at the end
   for source_id in sorted(self._provisional_utterances.keys()):
       u = self._provisional_utterances[source_id]
       ts = datetime.fromtimestamp(u.start_time, tz=timezone.utc).strftime("%H:%M:%S")
       # Mark as in-progress so the LLM knows this is still being spoken
       lines.append(f"[{ts}] {u.source_label} (speaking...): {u.text}")
   ```

---

## Transcript Format

The transcript sent to the LLM now looks like:

```
[10:30:15] Microphone: Hello everyone.
[10:30:18] Microphone: Let's discuss the architecture.
[10:30:23] System Audio: I agree, we need to plan this carefully.
[10:30:28] Microphone (speaking...): One thing I'm concerned about is
```

The `(speaking...)` marker tells the LLM that:
- This utterance is **in-progress**
- The speaker hasn't finished yet
- The text may be incomplete or updated soon

---

## Lifecycle

### State Transitions

```
1. Speaker starts talking
   └─→ Provisional utterance created
        └─→ Stored in _provisional_utterances[source_id]
             └─→ Included in insights with "(speaking...)"

2. Speaker continues (more words)
   └─→ Provisional utterance updated
        └─→ _provisional_utterances[source_id] replaced
             └─→ Updated text shown in insights

3. Speaker finishes (VAD detects silence)
   └─→ Final utterance received
        └─→ Added to _utterances
             └─→ Removed from _provisional_utterances
                  └─→ No longer marked as "(speaking...)"
```

### Example Timeline

```
Time    Event                           _utterances              _provisional_utterances
------  ------------------------------  -----------------------  ------------------------
10:30   "Hello" (final)                ["Hello"]                {}
10:31   "How are" (provisional)        ["Hello"]                {0: "How are"}
10:32   "How are you" (provisional)    ["Hello"]                {0: "How are you"}
10:33   "How are you?" (final)         ["Hello", "How are you?"] {}
```

---

## Multi-Speaker Support

When multiple sources are speaking simultaneously:

```
[10:30:15] Microphone: We need to decide on the framework.
[10:30:18] System Audio (speaking...): I think React would be best because
[10:30:20] Microphone (speaking...): Although we should also consider
```

Both provisional utterances are included, sorted by `source_id` for consistency.

---

## Testing

**Test script:** `scripts/test_provisional_in_insights.py`

Verifies:
- ✓ Provisional utterances tracked correctly
- ✓ Appear in formatted transcript with "(speaking...)"
- ✓ Update when speaker continues
- ✓ Cleared when finalized
- ✓ Multiple speakers handled correctly

**Run test:**
```bash
uv run python scripts/test_provisional_in_insights.py
```

---

## Usage

No configuration needed - this works automatically!

1. **Start a recording session**
2. **Begin speaking** (provisional utterances start appearing)
3. **Request an insight** while speaking (Ctrl+1, Ctrl+2, etc.)
4. **Insight includes your in-progress speech** ✨

The LLM sees:
```
Your completed sentences
+ What you're currently saying (marked with "speaking...")
```

---

## Benefits

1. **Real-time context** - Insights use the most current information
2. **No waiting** - Don't need to wait for speaker to finish
3. **Better accuracy** - LLM has full context including ongoing speech
4. **Multi-speaker aware** - Handles overlapping speakers correctly

---

## Technical Notes

- Provisional utterances come from VAD (Voice Activity Detection) during active speech
- They update in real-time as the ASR model processes audio chunks
- Memory footprint is minimal - only one provisional utterance per active source
- Automatically cleared when speech ends (VAD detects silence → final utterance)

---

## Future Enhancements

Possible improvements:
- Add visual indicator in UI when provisional utterances are active
- Option to exclude provisional utterances (if user prefers final-only)
- Show provisional utterance age/confidence in transcript

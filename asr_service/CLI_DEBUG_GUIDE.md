# CLI Debug Guide

## Where to Find Logs

The CLI now logs everything to a file in your home directory:

```bash
~/.asr_cli_debug.log
```

## How to Debug

### Step 1: Start Backend with Debug Logging
```bash
# Terminal 1
export LOG_LEVEL=DEBUG && make run
```

### Step 2: Start CLI
```bash
# Terminal 2
make run-cli
```

### Step 3: Watch the CLI Log File
```bash
# Terminal 3 (optional - to see CLI logs in real-time)
tail -f ~/.asr_cli_debug.log
```

### Step 4: Start Recording and Speak
1. Select your microphone in CLI
2. Click "Start Recording"
3. **Speak into your microphone**
4. Wait a few seconds
5. Stop recording

### Step 5: Check the Logs

**Backend log (Terminal 1) should show:**
```
DEBUG - Broadcasting utterance from source 0: 'hello world...' (clients: 1)
DEBUG - Utterance broadcast scheduled
```

**CLI log file should show:**
```
INFO - Connecting to WebSocket: ws://localhost:8000/api/v1/ws/...
INFO - WebSocket connected successfully
INFO - Received message #1: {"type":"state_change",...}
INFO - Received message #2: {"type":"utterance",...}
INFO - Parsed utterance: source=0, text='hello world...'
INFO - TranscriptView.add_utterance called: source=0, text='hello world...'
INFO - Writing line to RichLog: [cyan][Source 0][/cyan] 12:34:56 - hello world
INFO - Line written successfully
```

## What to Look For

### If You See Utterances in Backend but NOT in CLI Log:

**Backend shows:**
```
DEBUG - Broadcasting utterance from source 0...
DEBUG - Utterance broadcast scheduled
```

**CLI log shows nothing after "WebSocket connected successfully"**

→ **WebSocket connection issue**

### If CLI Log Shows Utterances Received but NOT Written:

**CLI log shows:**
```
INFO - Parsed utterance: source=0, text='hello'
```

**But no "Writing line to RichLog" messages**

→ **Error in message parsing** (check for exceptions)

### If CLI Log Shows Lines Written but Nothing on Screen:

**CLI log shows:**
```
INFO - Writing line to RichLog: [cyan][Source 0][/cyan] ...
INFO - Line written successfully
```

**But screen is blank**

→ **Textual rendering issue** (widget not visible or refresh problem)

## Quick Test

Run this to see everything at once:

```bash
# Terminal 1: Backend
export LOG_LEVEL=DEBUG && make run

# Terminal 2: CLI
make run-cli

# Terminal 3: Watch CLI logs
tail -f ~/.asr_cli_debug.log

# Terminal 4: Backend logs (filtered)
tail -f <backend_output> | grep "Broadcasting utterance"
```

## After Testing, Share These:

```bash
# 1. Last 100 lines of CLI log
tail -100 ~/.asr_cli_debug.log

# 2. Backend output (just the relevant part showing utterances)
# Copy from Terminal 1 starting from "Broadcasting utterance"
```

This will show us exactly where the messages are getting stuck!

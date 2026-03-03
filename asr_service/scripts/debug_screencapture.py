#!/usr/bin/env python3
"""
Debug ScreenCaptureKit system audio capture on this machine.

Runs the screencapture_audio binary with stderr streamed live so you can see
permission errors, "No displays", or stream failures immediately. Use this when
system audio works on another machine but not here.

Usage (from asr_service/):
  python scripts/debug_screencapture.py [duration_seconds]
  # or
  uv run python scripts/debug_screencapture.py 15

Example output when it fails:
  [stderr] ERROR: ...  → you see the exact Swift error
  Exit code: 1, stdout bytes: 0  → process died before sending audio
"""

import subprocess
import sys
import threading
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    swift_source = script_dir / "screencapture_audio.swift"
    swift_binary = script_dir / "screencapture_audio"

    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample_rate = 16000

    print("=" * 70)
    print("ScreenCaptureKit debug – system audio capture on THIS machine")
    print("=" * 70)
    print(f"\nBinary: {swift_binary}")
    print(f"Duration: {duration}s, sample rate: {sample_rate} Hz")
    print("\nStderr from the binary is printed below in real time.")
    print("Look for lines starting with ERROR or INFO.\n")

    # Compile if needed
    if not swift_binary.exists() or (
        swift_source.exists()
        and swift_binary.stat().st_mtime < swift_source.stat().st_mtime
    ):
        print("Compiling Swift binary...")
        r = subprocess.run(
            ["swiftc", str(swift_source), "-o", str(swift_binary)],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        if r.returncode != 0:
            print("Compilation failed:")
            print(r.stderr or r.stdout)
            return 1
        print("Compilation OK.\n")

    if not swift_binary.exists():
        print(f"Binary not found: {swift_binary}")
        return 1

    # Run binary with stderr streamed live; stdout we consume for byte count
    proc = subprocess.Popen(
        [str(swift_binary), str(sample_rate), str(duration)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=script_dir,
    )

    stdout_chunks = []
    stderr_done = threading.Event()

    def read_stderr():
        try:
            for line in iter(proc.stderr.readline, b""):
                if line:
                    sys.stderr.write(f"[screencapture stderr] {line.decode('utf-8', errors='replace')}")
                    sys.stderr.flush()
        finally:
            stderr_done.set()

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            stdout_chunks.append(chunk)
    except Exception as e:
        print(f"\nError reading stdout: {e}", file=sys.stderr)

    returncode = proc.wait()
    stderr_done.wait(timeout=1.0)
    t.join(timeout=0.5)

    total_bytes = sum(len(c) for c in stdout_chunks)
    total_samples = total_bytes // 4  # float32
    total_sec = total_samples / sample_rate if sample_rate else 0

    print("\n" + "-" * 70)
    print("Result")
    print("-" * 70)
    print(f"  Exit code:    {returncode}")
    print(f"  Stdout:      {total_bytes} bytes ({total_samples} samples, ~{total_sec:.2f}s)")
    if returncode != 0:
        print("\n  Process exited with an error. Check [screencapture stderr] lines above.")
    if returncode == 0 and total_bytes == 0:
        print("\n  No audio data received despite exit 0. Stream may have failed silently.")
    if total_bytes > 0:
        print("\n  Audio data received – capture is working on this machine.")

    print("\nChecklist on this machine:")
    print("  • macOS 13.0 or later (ScreenCaptureKit requirement)")
    print("  • System Settings → Privacy & Security → Screen Recording")
    print("    → Allow the app that runs this script (Terminal, VS Code, Cursor, etc.)")
    print("  • If you run the ASR backend (uvicorn), that process also needs Screen Recording")
    print("  • At least one display connected (no headless-only setups without a virtual display)")
    print("\nBackend logs: when using the full app, ScreenCaptureKit stderr is also")
    print("logged by the ASR service. Check logs/asr_service.log or console (LOG_LEVEL=DEBUG).")
    print("=" * 70)

    return 0 if returncode == 0 and total_bytes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

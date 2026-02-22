#!/usr/bin/env python3
import requests
import sys
import time

def transcribe_file(filepath):
    url = "http://localhost:8000/api/v1/transcribe_final"

    with open(filepath, 'rb') as f:
        files = {'file': (filepath, f, 'audio/mpeg')}

        print(f"Transcribing {filepath}...")
        start = time.time()
        response = requests.post(url, files=files)
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print(f"\nTranscription completed in {elapsed:.2f}s (server reported: {result['processing_time']:.2f}s)")
            print(f"\nText:\n{result['text']}\n")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1])
    else:
        print("Usage: python transcribe_file.py <audio_file>")

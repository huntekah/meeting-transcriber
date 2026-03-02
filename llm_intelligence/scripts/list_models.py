#!/usr/bin/env python3
"""
List available Ollama and Gemini models using the project SDKs.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def list_ollama() -> None:
    import ollama

    def _model_name(model: dict) -> str:
        for key in ("model", "name", "tag"):
            value = model.get(key)
            if value:
                return value
        return "unknown"

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)
    print(f"[ollama] host={host}")
    try:
        response = client.list()
    except Exception as exc:
        print(f"[ollama] error: {exc}")
        return

    models = response.get("models", [])
    if not models:
        print("[ollama] no local models found")
        return

    for model in models:
        name = _model_name(model)
        size = model.get("size")
        size_label = size if size is not None else "unknown"
        print(f"  - {name} (size={size_label})")


def list_gemini() -> None:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    print("[gemini]")
    try:
        pager = client.models.list()
    except Exception as exc:
        print(f"[gemini] error: {exc}")
        return

    found = False
    for model in pager:
        methods = getattr(model, "supported_generation_methods", None)
        if methods and "generateContent" not in methods:
            continue
        if methods and any(m in methods for m in ("generateImages", "generateVideos", "generateAudio")):
            continue

        name = getattr(model, "name", "unknown")
        display = getattr(model, "display_name", None)
        size = getattr(model, "size", None) or getattr(model, "model_size", None)
        size_label = f" (size={size})" if size else ""
        input_limit = getattr(model, "input_token_limit", None)
        output_limit = getattr(model, "output_token_limit", None)
        limits_label = ""
        if input_limit or output_limit:
            limits_label = f" (in={input_limit}, out={output_limit})"
        display_label = f" ({display})" if display and display != name else ""
        print(f"  - {name}{display_label}{size_label}{limits_label}")
        found = True

    if not found:
        print("[gemini] no models returned")


def main() -> None:
    _load_env()
    list_ollama()
    print()
    list_gemini()


if __name__ == "__main__":
    main()

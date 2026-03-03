"""Model catalog helpers for listing available LLM models by provider."""
from __future__ import annotations

import os
from typing import Iterable

from google import genai
import ollama

from llm_intelligence.schemas import (
    ListModelsResponse,
    ModelCapabilities,
    ModelInfo,
    ModelListError,
    ModelProvider,
)


def _ollama_model_name(model: dict) -> str:
    for key in ("model", "name", "tag"):
        value = model.get(key)
        if value:
            return value
    return "unknown"


def list_ollama_models(host: str | None = None) -> list[ModelInfo]:
    client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.list()
    models = response.get("models", [])
    results: list[ModelInfo] = []
    for model in models:
        name = _ollama_model_name(model)
        results.append(
            ModelInfo(
                id=name,
                provider=ModelProvider.ollama,
                display_name=name,
                size_bytes=model.get("size"),
                capabilities=ModelCapabilities(text=True),
                is_local=True,
            )
        )
    return results


def _capabilities_from_methods(methods: Iterable[str] | None) -> ModelCapabilities:
    if not methods:
        return ModelCapabilities(text=True)
    methods_set = set(methods)
    return ModelCapabilities(
        text="generateContent" in methods_set,
        image="generateImages" in methods_set,
        audio="generateAudio" in methods_set,
        video="generateVideos" in methods_set,
    )


def _normalize_gemini_id(name: str | None) -> str:
    if not name:
        return "unknown"
    return name.removeprefix("models/")


def list_gemini_models() -> list[ModelInfo]:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    results: list[ModelInfo] = []
    for model in client.models.list():
        methods = getattr(model, "supported_generation_methods", None)
        if methods and "generateContent" not in methods:
            continue
        if methods and any(m in methods for m in ("generateImages", "generateVideos", "generateAudio")):
            continue

        capabilities = _capabilities_from_methods(methods)
        results.append(
            ModelInfo(
                id=_normalize_gemini_id(getattr(model, "name", None)),
                provider=ModelProvider.gemini,
                display_name=getattr(model, "display_name", None),
                input_token_limit=getattr(model, "input_token_limit", None),
                output_token_limit=getattr(model, "output_token_limit", None),
                capabilities=capabilities,
                is_local=False,
            )
        )
    return results


def list_models(provider: str = "all", *, ollama_host: str | None = None) -> ListModelsResponse:
    models: list[ModelInfo] = []
    errors: list[ModelListError] = []

    if provider not in ("all", "ollama", "gemini"):
        raise ValueError(f"Unsupported provider: {provider}")

    if provider in ("all", "ollama"):
        try:
            models.extend(list_ollama_models(host=ollama_host))
        except Exception as exc:  # noqa: BLE001 - surface as error in response
            errors.append(ModelListError(provider=ModelProvider.ollama, message=str(exc)))

    if provider in ("all", "gemini"):
        try:
            models.extend(list_gemini_models())
        except Exception as exc:  # noqa: BLE001 - surface as error in response
            errors.append(ModelListError(provider=ModelProvider.gemini, message=str(exc)))

    return ListModelsResponse(models=models, errors=errors)

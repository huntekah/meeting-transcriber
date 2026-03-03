"""Tests for model catalog helpers."""

import pytest

from llm_intelligence.model_catalog import list_models
from llm_intelligence.schemas import ModelInfo, ModelProvider


def _model(provider: ModelProvider, name: str) -> ModelInfo:
    return ModelInfo(id=name, provider=provider, is_local=(provider == ModelProvider.ollama))


def test_list_models_provider_filter(monkeypatch):
    monkeypatch.setattr(
        "llm_intelligence.model_catalog.list_ollama_models",
        lambda host=None: [_model(ModelProvider.ollama, "ollama-a")],
    )
    monkeypatch.setattr(
        "llm_intelligence.model_catalog.list_gemini_models",
        lambda: [_model(ModelProvider.gemini, "gemini-a")],
    )

    ollama_only = list_models(provider="ollama")
    assert [m.id for m in ollama_only.models] == ["ollama-a"]

    gemini_only = list_models(provider="gemini")
    assert [m.id for m in gemini_only.models] == ["gemini-a"]

    both = list_models(provider="all")
    assert [m.id for m in both.models] == ["ollama-a", "gemini-a"]


def test_list_models_errors(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("llm_intelligence.model_catalog.list_ollama_models", _raise)
    monkeypatch.setattr(
        "llm_intelligence.model_catalog.list_gemini_models",
        lambda: [_model(ModelProvider.gemini, "gemini-a")],
    )

    response = list_models(provider="all")
    assert [m.id for m in response.models] == ["gemini-a"]
    assert response.errors[0].provider == ModelProvider.ollama
    assert "ollama down" in response.errors[0].message


def test_list_models_invalid_provider():
    with pytest.raises(ValueError):
        list_models(provider="other")

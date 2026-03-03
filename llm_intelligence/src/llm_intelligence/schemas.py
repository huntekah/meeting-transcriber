"""Pydantic schemas for the LLM Intelligence service."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SkillInfo(BaseModel):
    """Public metadata for a skill (returned by GET /skills)."""

    name: str
    description: str
    display: str


class SkillMeta(SkillInfo):
    """Full skill definition including the prompt template (internal use only)."""

    prompt_template: str


class InsightRequest(BaseModel):
    """Request body for POST /insights."""

    skill_name: str
    transcript: str
    model: str | None = None  # overrides server default; "gemini-*" → Gemini, else → Ollama


class InsightResponse(BaseModel):
    """Response from POST /insights."""

    markdown: str
    skill_name: str


class ModelProvider(str, Enum):
    ollama = "ollama"
    gemini = "gemini"


class ModelCapabilities(BaseModel):
    text: bool = True
    image: bool | None = None
    audio: bool | None = None
    video: bool | None = None


class ModelInfo(BaseModel):
    id: str  # provider-native id (ollama: "llama3.1:70b", gemini: "models/gemini-2.5-flash")
    provider: ModelProvider
    display_name: str | None = None
    size_bytes: int | None = None
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    capabilities: ModelCapabilities | None = None
    is_local: bool = False


class ModelListError(BaseModel):
    provider: ModelProvider
    message: str


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str | None = None
    errors: list[ModelListError] = []

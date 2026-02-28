"""Pydantic schemas for the LLM Intelligence service."""
from __future__ import annotations

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

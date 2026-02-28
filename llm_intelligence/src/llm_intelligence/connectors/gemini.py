"""Gemini backend — uses google-genai SDK for plain-text generation."""
from __future__ import annotations

import os

from google import genai
from google.genai.types import HttpOptions
from loguru import logger

from llm_intelligence.connectors.base import LLMBackend


class GeminiBackend(LLMBackend):
    """Calls the Gemini API via the google-genai SDK."""

    def __init__(self, model: str) -> None:
        self.model = model
        if api_key := os.getenv("GEMINI_API_KEY"):
            self.client = genai.Client(
                api_key=api_key,
                http_options=HttpOptions(timeout=180_000),
            )
        else:
            self.client = genai.Client(
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                http_options=HttpOptions(timeout=180_000),
            )

    async def generate_async(self, prompt: str) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = response.text or ""
        logger.info(
            "[LLM_USAGE] Gemini {} | input: {} chars | output: {} chars",
            self.model,
            len(prompt),
            len(text),
        )
        return text

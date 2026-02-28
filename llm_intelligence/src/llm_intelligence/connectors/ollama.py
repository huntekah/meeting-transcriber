"""Ollama backend — uses the ollama SDK for local model inference."""
from __future__ import annotations

import os

import ollama
from loguru import logger

from llm_intelligence.connectors.base import LLMBackend

_SOFT_CHAR_LIMIT = 32_000  # warning only; Ollama may silently truncate beyond this


class OllamaBackend(LLMBackend):
    """Calls a locally running Ollama instance."""

    def __init__(self, model: str) -> None:
        self.model = model
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.AsyncClient(host=host)

    async def generate_async(self, prompt: str) -> str:
        if len(prompt) > _SOFT_CHAR_LIMIT:
            logger.warning(
                "Prompt is {} chars — exceeds soft Ollama limit of {}. "
                "Local models may truncate context.",
                len(prompt),
                _SOFT_CHAR_LIMIT,
            )

        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        text: str = response["message"]["content"]
        logger.info(
            "[LLM_USAGE] Ollama {} | input: {} chars | output: {} chars",
            self.model,
            len(prompt),
            len(text),
        )
        return text

"""LLM connector base — ABC backend + LLMClient facade.

LLMClient routes based on model name:
  - model starts with "gemini" → GeminiBackend
  - anything else              → OllamaBackend

Retries are handled here via tenacity so backends stay simple.
Cache lookup / store wraps every call.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_intelligence.cache import create_cache_key, get_cached, set_cached


class LLMBackend(ABC):
    """Abstract base for LLM provider backends."""

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the plain-text response."""


class LLMClient:
    """
    Facade that routes to the correct backend, handles retry, and manages cache.

    Usage::

        client = LLMClient(model="gemini-2.5-flash")
        markdown = await client.generate(prompt)
    """

    def __init__(self, model: str | None = None) -> None:
        self.model: str = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")

        if self.model.lower().startswith("gemini"):
            from llm_intelligence.connectors.gemini import GeminiBackend

            logger.info("Using Gemini backend — model: {}", self.model)
            self.backend = GeminiBackend(self.model)
        else:
            from llm_intelligence.connectors.ollama import OllamaBackend

            logger.info("Using Ollama backend — model: {}", self.model)
            self.backend = OllamaBackend(self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_with_retry(self, prompt: str) -> str:
        return await self.backend.generate_async(prompt)

    async def generate(self, prompt: str) -> str:
        """Generate a response, using cache when possible."""
        key = create_cache_key(self.model, prompt)
        cached = get_cached(key)
        if cached is not None:
            return cached

        logger.debug("🔄 Cache miss — calling {} API", self.model)
        result = await self._generate_with_retry(prompt)
        set_cached(key, result)
        return result

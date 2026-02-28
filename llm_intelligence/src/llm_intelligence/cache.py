"""Disk-backed cache for LLM responses.

Adapted from anki_from_anything/src/utils/cache.py (diskcache).
Cache key is sha256(model + prompt) so identical calls are never repeated.
"""
from __future__ import annotations

import hashlib
import os

import diskcache as dc
from loguru import logger

_CACHE_DIR = os.getenv("LLM_CACHE_DIR", ".llm_cache")
_CACHE_SIZE_LIMIT = 2_000_000_000  # 2 GB

cache = dc.Cache(_CACHE_DIR, size_limit=_CACHE_SIZE_LIMIT)


def create_cache_key(model: str, prompt: str) -> str:
    """Return a deterministic sha256 key for (model, prompt)."""
    content = f"{model}|{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached(key: str) -> str | None:
    """Return cached string value or None if not present."""
    value = cache.get(key)
    if value is not None:
        logger.debug("🎯 Cache hit for key {}", key[:12])
    return value


def set_cached(key: str, value: str) -> None:
    """Store a string value in the cache."""
    try:
        cache.set(key, value)
        logger.debug("💾 Cached response for key {}", key[:12])
    except Exception as exc:
        logger.warning("Failed to cache response: {}", exc)

"""
Singleton base for shared managers.

Provides a thread-safe __new__ implementation with a per-class instance.
"""

from __future__ import annotations

import threading


class SingletonBase:
    """Thread-safe singleton base class."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

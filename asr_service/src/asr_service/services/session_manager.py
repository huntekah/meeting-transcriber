"""
Session manager service.

Singleton registry managing all active sessions.
Thread-safe with async lock for session operations.
"""

import asyncio
import uuid
from typing import Dict, List, Optional
from pathlib import Path

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import SessionNotFoundError
from ..schemas.transcription import SourceConfig
from .session import ActiveSession


class SessionManager:
    """
    Global singleton managing all active sessions.

    Thread-safe registry with async lock.
    Provides session lifecycle management.
    """

    _instance: Optional["SessionManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Session registry
        self._sessions: Dict[str, ActiveSession] = {}
        self._sessions_lock = asyncio.Lock()

        self._initialized = True
        logger.info("SessionManager singleton initialized")

    async def create_session(
        self,
        sources: List[SourceConfig],
        model_manager,
        output_dir: Path | str | None = None,
    ) -> ActiveSession:
        """
        Create new session with unique ID.

        Args:
            sources: List of audio sources to capture
            model_manager: ModelManager instance
            output_dir: Optional output directory (default from settings)

        Returns:
            ActiveSession instance
        """
        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Use default output dir if not specified
        output_dir = Path(output_dir) if output_dir else settings.OUTPUT_DIR

        # Create session
        session = ActiveSession(
            session_id=session_id,
            sources=sources,
            model_manager=model_manager,
            output_dir=output_dir,
        )

        # Add to registry
        async with self._sessions_lock:
            self._sessions[session_id] = session

        logger.info(
            f"Created session {session_id} with {len(sources)} sources "
            f"(total active: {len(self._sessions)})"
        )

        # Initialize session
        await session.initialize()

        return session

    async def get_session(self, session_id: str) -> ActiveSession:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ActiveSession instance

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise SessionNotFoundError(session_id)

        return session

    async def get_session_or_none(self, session_id: str) -> Optional[ActiveSession]:
        """
        Get session by ID without raising exception.

        Args:
            session_id: Session identifier

        Returns:
            ActiveSession instance or None
        """
        async with self._sessions_lock:
            return self._sessions.get(session_id)

    async def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        async with self._sessions_lock:
            return list(self._sessions.keys())

    async def delete_session(self, session_id: str):
        """
        Remove session from registry.

        Args:
            session_id: Session identifier
        """
        async with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(
                    f"Deleted session {session_id} (remaining: {len(self._sessions)})"
                )

    async def get_all_sessions(self) -> Dict[str, ActiveSession]:
        """
        Get all sessions.

        Returns:
            Dictionary mapping session_id → ActiveSession
        """
        async with self._sessions_lock:
            return dict(self._sessions)

    def get_stats(self) -> dict:
        """
        Get session manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_sessions": len(self._sessions),
            "session_ids": list(self._sessions.keys()),
        }

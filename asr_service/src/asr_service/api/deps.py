"""
FastAPI dependency injection.

Provides dependency injection for shared resources (singletons).
"""

from ..services.session_manager import SessionManager
from ..services.model_manager import ModelManager


def get_session_manager() -> SessionManager:
    """
    Dependency: SessionManager singleton.

    Returns:
        SessionManager instance

    Usage:
        @app.post("/sessions")
        async def create_session(
            manager: SessionManager = Depends(get_session_manager)
        ):
            ...
    """
    return SessionManager()


def get_model_manager() -> ModelManager:
    """
    Dependency: ModelManager singleton.

    Returns:
        ModelManager instance

    Usage:
        @app.get("/health")
        async def health(
            model_manager: ModelManager = Depends(get_model_manager)
        ):
            ...
    """
    return ModelManager()

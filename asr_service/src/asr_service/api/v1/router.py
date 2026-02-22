"""
API v1 router configuration.

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from .endpoints import devices, sessions, websocket

# Create API v1 router
api_router = APIRouter(prefix="/api/v1")

# Include endpoint routers
api_router.include_router(devices.router)
api_router.include_router(sessions.router)
api_router.include_router(websocket.router)

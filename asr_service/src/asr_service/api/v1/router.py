from fastapi import APIRouter
from asr_service.api.v1.endpoints import health, transcribe

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(transcribe.router, tags=["transcription"])

from fastapi import APIRouter, Depends
from asr_service.api.deps import get_asr_engine
from asr_service.schemas.transcription import HealthResponse
from asr_service.services.model_loader import ASREngine
from asr_service.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(engine: ASREngine = Depends(get_asr_engine)) -> HealthResponse:
    """
    Health check endpoint.

    Returns system status, model loading state, and device information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=engine.is_loaded,
        device=engine.device,
        model_id=settings.MODEL_ID,
    )

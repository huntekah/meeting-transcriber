from contextlib import asynccontextmanager
from fastapi import FastAPI
from asr_service.api.v1.router import api_router
from asr_service.core.config import settings
from asr_service.services.model_loader import ASREngine
from asr_service.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.

    Loads the ASR model on startup and cleans up on shutdown.
    """
    # Startup: Load the heavy ASR model
    logger.info("Starting up ASR service...")
    logger.info(f"Device: {settings.get_device()}")
    logger.info(f"Model: {settings.MODEL_ID}")

    engine = ASREngine()
    try:
        engine.load_model()
        logger.info("ASR service ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown: Clean up resources if needed
    logger.info("Shutting down ASR service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ASR meeting transcriber microservice using Whisper",
    version="0.1.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "status": "running",
    }

"""
FastAPI application entry point.

Main application with lifespan management, CORS, and API routing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import logger
from .api.v1.router import api_router
from .services.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=redefined-outer-name
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Multi-Source ASR Service starting...")
    logger.info(f"MLX-Whisper model: {settings.MLX_WHISPER_MODEL}")
    logger.info(f"Diarization model: {settings.DIARIZATION_MODEL}")
    logger.info(f"Device: {settings.get_device()}")
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")
    logger.info("=" * 80)

    # Pre-load models on startup for better UX (no delay on first utterance)
    model_manager = ModelManager()
    await model_manager.load_models()
    logger.info("Models pre-loaded successfully")

    yield

    # Shutdown
    logger.info("Multi-Source ASR Service shutting down...")

    # Cleanup models
    model_manager = ModelManager()
    model_manager.unload_models()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Multi-Source ASR Service",
    description="Real-time multi-source speech recognition with speaker diarization",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint.

    Returns service information.
    """
    return {
        "service": "Multi-Source ASR Service",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/sessions",
    }


@app.get("/health", tags=["health"])
async def health():
    """
    Health check endpoint.

    Returns basic health status.
    """
    model_manager = ModelManager()

    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded(),
        "whisper_model": settings.MLX_WHISPER_MODEL,
        "device": settings.get_device(),
    }

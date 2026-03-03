"""
FastAPI application entry point.

Main application with lifespan management, CORS, and API routing.
"""

import asyncio
import platform
import subprocess  # nosec B404
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import logger
from .api.v1.router import api_router
from .services.model_manager import ModelManager
from .services import audio_devices
from .utils.file_ops import get_project_root


def _run_screencapture_startup_probe() -> None:
    """
    Run the ScreenCaptureKit binary briefly so this process triggers the
    macOS Screen Recording permission dialog. Ensures "make run" prompts
    for access the same way the debug script does.
    """
    if platform.system() != "Darwin":
        return
    if not audio_devices._is_screencapture_binary_available():
        logger.debug(
            "ScreenCaptureKit binary not available, skipping startup probe"
        )
        return
    try:
        root = get_project_root()
        binary = root / "scripts" / "screencapture_audio"
        # Run for 1 second so SCShareableContent is called and permission is requested
        proc = subprocess.Popen(  # nosec B603
            [str(binary), "16000", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=str(root / "scripts"),
        )
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        if proc.returncode != 0 and proc.stderr:
            err = proc.stderr.read().decode("utf-8", errors="replace").strip()
            if err:
                logger.debug("ScreenCaptureKit probe stderr: %s", err)
        logger.info(
            "ScreenCaptureKit startup probe finished (permission may have been requested)"
        )
    except Exception as e:
        logger.warning("ScreenCaptureKit startup probe failed: %s", e)


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

    # Trigger Screen Recording permission for this process (same as debug script)
    # so "make run" shows the macOS dialog instead of failing silently later
    asyncio.create_task(
        asyncio.to_thread(_run_screencapture_startup_probe)
    )

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

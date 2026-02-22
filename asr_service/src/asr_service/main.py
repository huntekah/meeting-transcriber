"""
FastAPI application entry point.

SHOULD:
- Create FastAPI app instance
- Configure CORS, middleware
- Include API v1 router
- Handle startup/shutdown events for model lifecycle
- Configure OpenAPI documentation

CONTRACTS:
- Export: app (FastAPI instance)
- Startup behavior:
  - Load ASR models (or lazy load on first request)
  - Log startup info (model, device, etc.)
- Shutdown behavior:
  - Cleanup models
  - Close connections

STRUCTURE:
app = FastAPI(
    title="ASR Service",
    description="Speech-to-text transcription with speaker diarization",
    version="2.0.0"  # Bump version for MLX rewrite
)

@app.on_event("startup")
async def startup():
    # Optional: Pre-load models
    # Or let them load lazily on first request
    logger.info("ASR Service starting...")

@app.on_event("shutdown")
async def shutdown():
    # Cleanup
    logger.info("ASR Service shutting down...")

app.include_router(api_router)

CONFIGURATION:
- CORS: Allow configured origins from settings
- Max upload size: From settings
- Request timeout: From settings
- Logging: Use configured logger

RUN:
uvicorn asr_service.main:app --host 0.0.0.0 --port 8000
"""

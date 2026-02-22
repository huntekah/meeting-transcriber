"""
API v1 router configuration.

SHOULD:
- Create APIRouter for /api/v1 prefix
- Include all v1 endpoint routers (health, transcribe)
- Define route tags for OpenAPI documentation

CONTRACTS:
- Export: api_router (APIRouter instance)
- Includes:
  - health.router (tag: "health")
  - transcribe.router (tag: "transcription")

STRUCTURE:
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(transcribe.router, tags=["transcription"])

NOTE:
- This file is likely fine as-is, just need to ensure imports are correct
- Keep it simple - just route aggregation
"""

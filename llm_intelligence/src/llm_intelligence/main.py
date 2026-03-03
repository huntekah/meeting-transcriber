"""FastAPI application for the LLM Intelligence service."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/llm_intelligence/ → src/ → project root
load_dotenv(_PROJECT_ROOT / ".env", override=True)  # .env wins over shell — ensures GOOGLE_GENAI_USE_VERTEXAI=false is honoured

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from llm_intelligence.model_catalog import list_models as catalog_list_models  # noqa: E402
from llm_intelligence.schemas import (  # noqa: E402
    InsightRequest,
    InsightResponse,
    ListModelsResponse,
    SkillInfo,
)
from llm_intelligence.service import InsightsService, SkillNotFoundError  # noqa: E402


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")  # reads from os.environ (populated by load_dotenv above)

    llm_model: str = "gemini-2.5-flash"
    skills_dir: str = "skills"
    ollama_host: str = "http://localhost:11434"


settings = Settings()
_service: InsightsService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _service
    # Resolve skills_dir relative to this file: src/llm_intelligence/main.py → ../../skills
    _here = Path(__file__).resolve().parent          # src/llm_intelligence/
    _project_root = _here.parent.parent              # llm_intelligence/ (project root)
    skills_dir = str(_project_root / settings.skills_dir)
    _service = InsightsService.from_skills_dir(skills_dir)
    logger.info("LLM Intelligence service started. Skills: {}", [s.name for s in _service.list_skills()])
    yield
    logger.info("LLM Intelligence service shutting down.")


app = FastAPI(
    title="LLM Intelligence Service",
    description="Generates LLM-powered meeting transcript insights.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_service() -> InsightsService:
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}


@app.get("/skills", response_model=list[SkillInfo])
async def list_skills() -> list[SkillInfo]:
    """Return all available skill definitions."""
    return _get_service().list_skills()


@app.get("/models", response_model=ListModelsResponse)
def list_models(provider: str = Query("all", pattern="^(ollama|gemini|all)$")) -> ListModelsResponse:
    """Return available LLM models for the requested provider."""
    response = catalog_list_models(provider=provider, ollama_host=settings.ollama_host)
    response.default_model = settings.llm_model
    return response


@app.post("/insights", response_model=InsightResponse)
async def get_insight(request: InsightRequest) -> InsightResponse:
    """Generate an LLM insight for the given skill and transcript."""
    svc = _get_service()
    try:
        return await svc.get_insight(
            skill_name=request.skill_name,
            transcript=request.transcript,
            model=request.model,
        )
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error generating insight: {}", exc)
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

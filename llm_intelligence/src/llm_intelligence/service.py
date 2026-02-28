"""InsightsService — loads skills and dispatches LLM calls."""
from __future__ import annotations

from loguru import logger

from llm_intelligence.connectors.base import LLMClient
from llm_intelligence.schemas import InsightResponse, SkillInfo, SkillMeta
from llm_intelligence.skills_loader import load_skills


class SkillNotFoundError(KeyError):
    """Raised when a requested skill_name is not in the loaded skills."""


class InsightsService:
    """
    Service layer: loads skills from disk and handles insight generation.

    Instantiate once at application startup via ``from_skills_dir()``.
    """

    def __init__(self, skills: dict[str, SkillMeta]) -> None:
        self._skills = skills

    @classmethod
    def from_skills_dir(cls, skills_dir: str) -> "InsightsService":
        skills = load_skills(skills_dir)
        return cls(skills)

    def list_skills(self) -> list[SkillInfo]:
        """Return public metadata for all loaded skills."""
        return [SkillInfo(name=s.name, description=s.description, display=s.display) for s in self._skills.values()]

    async def get_insight(self, skill_name: str, transcript: str, model: str | None = None) -> InsightResponse:
        """
        Generate an LLM insight for the given skill and transcript.

        Args:
            skill_name: Slug matching a loaded skill file.
            transcript: Formatted transcript text to inject into the prompt.
            model: Optional model override (e.g. "gemini-2.5-pro", "llama3.2").

        Raises:
            SkillNotFoundError: If skill_name is not loaded.
        """
        if skill_name not in self._skills:
            available = list(self._skills.keys())
            raise SkillNotFoundError(f"Skill {skill_name!r} not found. Available: {available}")

        skill = self._skills[skill_name]
        prompt = skill.prompt_template.replace("{TRANSCRIPT}", transcript)

        logger.info("Generating insight for skill={!r} model={!r} transcript_chars={}", skill_name, model, len(transcript))

        client = LLMClient(model=model)
        markdown = await client.generate(prompt)

        return InsightResponse(markdown=markdown, skill_name=skill_name)

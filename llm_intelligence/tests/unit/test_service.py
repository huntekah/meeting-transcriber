"""Unit tests for InsightsService."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from llm_intelligence.service import InsightsService, SkillNotFoundError


@pytest.fixture
def service(skills_dir: Path) -> InsightsService:
    return InsightsService.from_skills_dir(str(skills_dir))


def test_list_skills_returns_skill_info(service: InsightsService) -> None:
    skills = service.list_skills()
    assert len(skills) == 1
    assert skills[0].name == "test-skill"
    assert skills[0].display == "🧪 Test Skill"
    # SkillInfo must NOT expose prompt_template
    assert not hasattr(skills[0], "prompt_template")


@pytest.mark.asyncio
async def test_get_insight_substitutes_transcript(service: InsightsService) -> None:
    with patch("llm_intelligence.service.LLMClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.generate = AsyncMock(return_value="## Result\nsome text")

        response = await service.get_insight("test-skill", "Speaker 0: Hello", model=None)

    assert response.skill_name == "test-skill"
    assert response.markdown == "## Result\nsome text"

    # Verify transcript was substituted into the prompt
    call_args = mock_instance.generate.call_args[0][0]
    assert "Speaker 0: Hello" in call_args
    assert "{TRANSCRIPT}" not in call_args


@pytest.mark.asyncio
async def test_get_insight_passes_model_override(service: InsightsService) -> None:
    with patch("llm_intelligence.service.LLMClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.generate = AsyncMock(return_value="md")

        await service.get_insight("test-skill", "transcript", model="gemini-2.5-pro")

    MockClient.assert_called_once_with(model="gemini-2.5-pro")


@pytest.mark.asyncio
async def test_get_insight_raises_for_unknown_skill(service: InsightsService) -> None:
    with pytest.raises(SkillNotFoundError):
        await service.get_insight("nonexistent-skill", "transcript")

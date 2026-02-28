"""
HTTP client for LLM insights service.

Sends transcript windows to the insights service and returns Markdown.
"""

import httpx
from cli_frontend.models import InsightRequest, InsightResponse, SkillInfo
from cli_frontend.logging import logger


class InsightsClient:
    """Async HTTP client for the LLM insights service."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        # LLM inference can be slow — generous timeout
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def get_skills(self) -> list[SkillInfo]:
        """
        Fetch the list of available skills from the insights service.

        Returns:
            List of SkillInfo objects, or empty list if service is unreachable.
        """
        try:
            response = await self._client.get("/skills", timeout=5.0)
            response.raise_for_status()
            return [SkillInfo.model_validate(s) for s in response.json()]
        except Exception as exc:
            logger.warning(f"Failed to fetch skills from insights service: {exc}")
            return []

    async def get_insight(
        self,
        transcript: str,
        skill_name: str,
        model: str | None = None,
    ) -> InsightResponse:
        """
        Request an insight for the given transcript window.

        Args:
            transcript: Formatted transcript text ([HH:MM:SS] Source N: text)
            skill_name: Name slug of the skill to apply
            model: Optional LLM model override (e.g. "gemini-2.5-pro")

        Returns:
            InsightResponse with rendered Markdown
        """
        payload = InsightRequest(
            transcript=transcript,
            skill_name=skill_name,
            model=model,
        )
        logger.info(
            f"Requesting insight: skill={skill_name}, "
            f"transcript_len={len(transcript)}, model={model}"
        )
        response = await self._client.post(
            "/insights", content=payload.model_dump_json(), headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return InsightResponse.model_validate(response.json())

    async def health_check(self) -> bool:
        """Return True if the insights service is reachable."""
        try:
            r = await self._client.get("/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()


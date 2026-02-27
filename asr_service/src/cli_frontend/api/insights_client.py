"""
HTTP client for LLM insights service.

Sends transcript windows to the insights service and returns Markdown.
"""

import httpx
from cli_frontend.models import InsightRequest, InsightResponse, InsightType
from cli_frontend.logging import logger


class InsightsClient:
    """Async HTTP client for the LLM insights service."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        # LLM inference can be slow — generous timeout
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def get_insight(
        self,
        transcript: str,
        insight_type: InsightType,
        window_minutes: float | None = None,
    ) -> InsightResponse:
        """
        Request an insight for the given transcript window.

        Args:
            transcript: Formatted transcript text ([HH:MM:SS] Source N: text)
            insight_type: Which insight type to generate
            window_minutes: How many minutes of transcript are included (informational)

        Returns:
            InsightResponse with rendered Markdown
        """
        payload = InsightRequest(
            transcript=transcript,
            insight_type=insight_type,
            window_minutes=window_minutes,
        )
        logger.info(
            f"Requesting insight: type={insight_type}, "
            f"transcript_len={len(transcript)}, window={window_minutes}m"
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

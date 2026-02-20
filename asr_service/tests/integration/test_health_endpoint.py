import pytest


async def test_health_endpoint_returns_200(async_client):
    """Test that the health endpoint returns 200 OK with status information."""
    response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data

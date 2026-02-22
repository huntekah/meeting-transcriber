"""
Health check endpoint.

SHOULD:
- Provide /health endpoint to check service status
- Return model loading status
- Include basic system info

CONTRACTS:
- Endpoint: GET /api/v1/health
- Response: HealthResponse schema
- Returns:
  - status: "healthy" | "unhealthy"
  - model_loaded: bool
  - model_name: str
  - device: str (e.g., "mps", "cuda", "cpu")

EXAMPLE RESPONSE:
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "mlx-community/whisper-large-v3-turbo",
    "device": "mps"
}

BEHAVIOR:
- Should not fail even if model not loaded
- Quick response, no heavy operations
- Used for k8s/docker health probes
"""

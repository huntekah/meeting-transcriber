import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from asr_service.main import app
from asr_service.services.model_loader import ASREngine


@pytest.fixture(scope="session")
def test_audio_file():
    """Path to the test audio file."""
    return Path(__file__).parent.parent / "this_is_a_test_that_we_will_use_to_check_asr.mp3"


@pytest.fixture(scope="session")
def expected_transcription():
    """Expected transcription from the test audio file."""
    return "This is a test that we will use to check ASR"


@pytest.fixture(scope="session")
def loaded_model():
    """Ensure the model is loaded once for all tests."""
    engine = ASREngine()
    if not engine.is_loaded:
        engine.load_model()
    return engine


@pytest.fixture
async def async_client(loaded_model):
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(loaded_model):
    """Synchronous test client for WebSocket testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_audio_bytes():
    """Sample raw PCM audio bytes for testing."""
    import numpy as np

    # Generate 1 second of sine wave at 16kHz (Float32 PCM)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    return audio.tobytes()

# Test Suite

Comprehensive test coverage for the ASR Service backend and CLI frontend.

## Test Structure

```
tests/
├── backend/              # Backend API and service tests
│   ├── test_api_endpoints.py
│   └── test_services.py
├── frontend/             # CLI frontend tests
│   ├── test_widgets.py
│   ├── test_api_client.py
│   └── test_models.py
└── conftest.py          # Shared fixtures
```

## Running Tests

### Run All Tests

```bash
make test
```

Or with uv directly:

```bash
uv run pytest
```

### Run Backend Tests Only

```bash
uv run pytest tests/backend/
```

### Run Frontend Tests Only

```bash
uv run pytest tests/frontend/
```

### Run Specific Test File

```bash
uv run pytest tests/backend/test_api_endpoints.py
```

### Run with Coverage

```bash
uv run pytest --cov=asr_service --cov=cli_frontend --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

### Run with Verbose Output

```bash
uv run pytest -v
```

### Run Specific Test

```bash
uv run pytest tests/backend/test_api_endpoints.py::TestRootEndpoints::test_root_endpoint -v
```

## Test Categories

### Backend Tests

**API Endpoints** (`test_api_endpoints.py`):
- Root and health endpoints
- Device discovery
- Session CRUD operations
- Error handling

**Services** (`test_services.py`):
- SessionManager singleton
- ModelManager lifecycle
- ChronologicalMerger overlap detection
- Audio device discovery

### Frontend Tests

**Widgets** (`test_widgets.py`):
- StatusBar recording timer
- TranscriptView utterance display
- DeviceSelector device management

**API Client** (`test_api_client.py`):
- HTTP REST client
- WebSocket client
- Session management

**Models** (`test_models.py`):
- Pydantic model validation
- WebSocket message serialization

## Fixtures

Shared fixtures in `conftest.py`:
- `async_client` - Async HTTP client for FastAPI testing
- `sync_client` - Synchronous test client for WebSocket testing
- `test_audio_file` - Path to test audio file
- `sample_audio_bytes` - Generated test audio data

## Mocking

Tests use `unittest.mock` to mock:
- Model loading (avoid loading heavy ML models)
- Hardware devices (sounddevice)
- WebSocket connections
- HTTP requests

## CI/CD

Tests should be run in CI pipeline:

```yaml
- name: Run tests
  run: uv run pytest --cov --cov-report=xml
```

## Writing New Tests

Follow these patterns:

**1. Class-based organization:**
```python
class TestFeature:
    """Test feature description."""

    def test_specific_behavior(self):
        """Test docstring."""
        # Arrange
        # Act
        # Assert
```

**2. Async tests:**
```python
@pytest.mark.asyncio
async def test_async_feature(async_client):
    response = await async_client.get("/endpoint")
    assert response.status_code == 200
```

**3. Mocking:**
```python
from unittest.mock import patch

def test_with_mock():
    with patch("module.function") as mock_func:
        mock_func.return_value = "test"
        # test code
```

## Coverage Goals

Target: **80%+ code coverage**

Current coverage:
- Backend API: Good coverage of endpoints
- Backend services: Core components tested
- Frontend widgets: Basic functionality tested
- Frontend API client: HTTP and WebSocket tested

## Known Limitations

- Model loading tests use mocks (no actual ML model testing)
- Audio processing tests use generated audio (no real recording)
- WebSocket tests mock connections (no full integration test)
- Thread safety not extensively tested

For full integration testing, see `integration/` directory.

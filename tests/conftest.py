import pytest
from httpx import ASGITransport, AsyncClient

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.main import app


@pytest.fixture(autouse=True)
def override_settings(tmp_path):
    """Ensure safe isolated test environments."""
    settings.default_workspace = str(tmp_path)
    yield


@pytest.fixture
async def async_client():
    """Async test client for FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

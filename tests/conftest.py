import pytest
from httpx import ASGITransport, AsyncClient

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.executors import agy as agy_mod
from cli_agent_dispatch.main import app


@pytest.fixture(autouse=True)
def override_settings(tmp_path):
    """Ensure safe isolated test environments."""
    settings.default_workspace = str(tmp_path)
    settings.session_db_path = str(tmp_path / "sessions.db")
    settings.litellm_sync_enabled = False
    agy_mod._AGY_MODELS_CACHE = None
    agy_mod._AGY_MODELS_CACHE_LIVE = False
    agy_mod._AGY_MODELS_CACHE_TS = 0.0
    yield
    agy_mod._AGY_MODELS_CACHE = None
    agy_mod._AGY_MODELS_CACHE_LIVE = False
    agy_mod._AGY_MODELS_CACHE_TS = 0.0


@pytest.fixture
async def async_client():
    """Async test client for FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

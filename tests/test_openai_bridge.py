from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import TaskResult


@pytest.mark.asyncio
async def test_health_check(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "CLIAgentDispatch"


@pytest.mark.asyncio
async def test_diagnostics_endpoint(async_client):
    resp = await async_client.get("/api/diagnostics")
    assert resp.status_code == 200
    assert "events" in resp.json()


@pytest.mark.asyncio
async def test_list_models(async_client):
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "opencode" in ids
    assert "agy" in ids


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(async_client):
    mock_res = TaskResult(
        success=True,
        executor="opencode",
        output="Mocked OpenCode response",
        duration_seconds=0.5,
    )
    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_res
        payload = {
            "model": "opencode",
            "messages": [{"role": "user", "content": "Explain async await"}],
            "stream": False,
        }
        resp = await async_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Mocked OpenCode response"


@pytest.mark.asyncio
async def test_chat_completions_streaming(async_client):
    mock_res = TaskResult(
        success=True,
        executor="agy",
        output="Streaming agy output",
        duration_seconds=0.2,
    )
    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_res
        payload = {
            "model": "agy-gemini",
            "messages": [{"role": "user", "content": "Review PR"}],
            "stream": True,
        }
        resp = await async_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "Streaming agy output" in text
        assert "[DONE]" in text


@pytest.mark.asyncio
async def test_chat_completions_empty_messages(async_client):
    resp = await async_client.post(
        "/v1/chat/completions", json={"model": "opencode", "messages": []}
    )
    assert resp.status_code == 400

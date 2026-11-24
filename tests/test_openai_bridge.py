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
    cards_by_id = {m["id"]: m for m in data["data"]}
    assert "opencode" in cards_by_id
    assert "agy" in cards_by_id
    assert "agy-hydrogen" in cards_by_id
    assert "agy-neon" in cards_by_id
    assert "agy-carbon" in cards_by_id
    assert "agy-argon" in cards_by_id
    assert "agy-opus" in cards_by_id
    assert "agy-sonnet" in cards_by_id
    assert "agy-flash" in cards_by_id
    assert cards_by_id["agy-hydrogen"]["reasoning_effort_levels"] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert cards_by_id["agy-hydrogen"]["default_reasoning_effort"] == "medium"
    # Tier-suffixed clutter must not be advertised in /v1/models
    assert "agy-argon-high" not in cards_by_id
    assert "agy-argon-low" not in cards_by_id
    assert "agy-hydrogen-high" not in cards_by_id
    assert "t3-agy-argon-high" not in cards_by_id
    assert "reasoning_effort_levels" not in cards_by_id["opencode"]


def test_resolve_executor_and_model_argon():
    from cli_agent_dispatch.api.openai_routes import _resolve_executor_and_model
    from cli_agent_dispatch.core.models import ExecutorType

    assert _resolve_executor_and_model("agy-argon") == (
        ExecutorType.AGY,
        "agy-argon",
    )
    assert _resolve_executor_and_model("agy-gemini-5-titanium") == (
        ExecutorType.AGY,
        "agy-gemini-5-titanium",
    )
    assert _resolve_executor_and_model("openai/agy-hydrogen") == (
        ExecutorType.AGY,
        "agy-hydrogen",
    )
    assert _resolve_executor_and_model("opencode") == (
        ExecutorType.OPENCODE,
        "opencode-default",
    )


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
    from cli_agent_dispatch.core.models import StreamChunk

    async def mock_stream_gen(task_req):
        yield StreamChunk(
            delta="Streaming agy output",
            session_id="conv-1",
            is_final=True,
            finish_reason="stop",
        )

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch_stream", side_effect=mock_stream_gen
    ):
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

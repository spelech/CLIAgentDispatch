from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.core.session_store import session_store


@pytest.mark.asyncio
async def test_anthropic_messages_non_streaming(async_client):
    mock_result = TaskResult(
        success=True,
        executor="agy",
        output="Hello from Anthropic Messages API via agy!",
        session_id="conv-anthropic-123",
        duration_seconds=0.25,
    )

    with patch(
        "cli_agent_dispatch.api.anthropic_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_result

        payload = {
            "model": "claude-sonnet-5.5",
            "system": "You are a coding assistant.",
            "messages": [{"role": "user", "content": "How do I define a Pydantic model?"}],
            "max_tokens": 1024,
            "stream": False,
        }
        headers = {"x-session-id": "client-thread-anthropic-1"}

        resp = await async_client.post("/v1/messages", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()

        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "claude-sonnet-5.5"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert "Hello from Anthropic Messages API" in data["content"][0]["text"]
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["output_tokens"] > 0
        assert resp.headers.get("x-session-id") == "client-thread-anthropic-1"

        # Verify session is persisted
        record = await session_store.get("client-thread-anthropic-1")
        assert record is not None
        assert record.native_session_id == "conv-anthropic-123"


@pytest.mark.asyncio
async def test_anthropic_messages_multi_turn_delta(async_client):
    # Pre-seed session
    await session_store.get_or_create(
        client_id="openclaw-thread-99",
        executor_type="agy",  # type: ignore
        native_session_id="conv-openclaw-persisted",
    )

    mock_result = TaskResult(
        success=True,
        executor="agy",
        output="Next step is running pytest.",
        session_id="conv-openclaw-persisted",
        duration_seconds=0.1,
    )

    with patch(
        "cli_agent_dispatch.api.anthropic_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_result

        payload = {
            "model": "claude-3-5-sonnet",
            "messages": [
                {"role": "user", "content": "Step 1"},
                {"role": "assistant", "content": "Done with step 1"},
                {"role": "user", "content": "What is step 2?"},
            ],
            "metadata": {"session_id": "openclaw-thread-99"},
        }

        resp = await async_client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"][0]["text"] == "Next step is running pytest."

        # Verify delta turn was extracted
        call_args: TaskRequest = mock_dispatch.call_args[0][0]
        assert call_args.prompt == "What is step 2?"
        assert call_args.session_id == "conv-openclaw-persisted"


@pytest.mark.asyncio
async def test_anthropic_messages_streaming(async_client):
    async def mock_stream_gen(task_req: TaskRequest):
        yield StreamChunk(delta="Streaming ", session_id="conv-anth-stream")
        yield StreamChunk(delta="tokens ", session_id="conv-anth-stream")
        yield StreamChunk(
            delta="safely", is_final=True, session_id="conv-anth-stream", finish_reason="stop"
        )

    with patch(
        "cli_agent_dispatch.api.anthropic_routes.engine.dispatch_stream",
        side_effect=mock_stream_gen,
    ):
        payload = {
            "model": "claude-sonnet-5.5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Stream test"}]}],
            "stream": True,
        }

        resp = await async_client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        text = resp.text
        assert "event: message_start" in text
        assert "event: content_block_start" in text
        assert "event: content_block_delta" in text
        assert "Streaming " in text
        assert "tokens " in text
        assert "safely" in text
        assert "event: content_block_stop" in text
        assert "event: message_delta" in text
        assert "event: message_stop" in text


@pytest.mark.asyncio
async def test_anthropic_messages_empty_validation(async_client):
    resp = await async_client.post(
        "/v1/messages", json={"model": "claude-sonnet-5.5", "messages": []}
    )
    assert resp.status_code == 400

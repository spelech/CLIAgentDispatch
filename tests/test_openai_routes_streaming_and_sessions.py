from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.core.session_store import session_store


@pytest.mark.asyncio
async def test_openai_chat_completions_multi_turn_session_flow(async_client):
    # Turn 1: Starts a new session with full history
    turn1_result = TaskResult(
        success=True,
        executor="agy",
        output="I can help you build an agent.",
        session_id="conv-agy-session-001",
        duration_seconds=0.3,
    )

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = turn1_result

        payload = {
            "model": "agy-claude",
            "messages": [
                {"role": "system", "content": "You are a helpful software engineer."},
                {"role": "user", "content": "Hello! Let's build an agent."},
            ],
            "stream": False,
        }
        headers = {"x-session-id": "client-thread-001"}
        resp = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "I can help you build an agent."
        assert resp.headers.get("x-session-id") == "client-thread-001"
        assert resp.headers.get("x-native-session-id") == "conv-agy-session-001"

        # Verify full prompt was passed on turn 1
        call_args: TaskRequest = mock_dispatch.call_args[0][0]
        assert "helpful software engineer" in call_args.system_prompt
        assert "Hello! Let's build an agent." in call_args.prompt
        assert call_args.session_id is None

    # Verify session is stored in session_store
    record = await session_store.get("client-thread-001")
    assert record is not None
    assert record.native_session_id == "conv-agy-session-001"

    # Turn 2: Follow-up turn passes the same session ID
    turn2_result = TaskResult(
        success=True,
        executor="agy",
        output="Step 1 is defining the models.",
        session_id="conv-agy-session-001",
        duration_seconds=0.2,
    )

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = turn2_result

        payload = {
            "model": "agy-claude",
            "messages": [
                {"role": "system", "content": "You are a helpful software engineer."},
                {"role": "user", "content": "Hello! Let's build an agent."},
                {"role": "assistant", "content": "I can help you build an agent."},
                {"role": "user", "content": "What is step 1?"},
            ],
            "stream": False,
        }
        headers = {"x-session-id": "client-thread-001"}
        resp = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Step 1 is defining the models."

        # Verify only the delta prompt was sent, preventing duplicate prompt growth
        call_args2: TaskRequest = mock_dispatch.call_args[0][0]
        assert call_args2.prompt == "What is step 1?"
        assert call_args2.system_prompt is None
        assert call_args2.session_id == "conv-agy-session-001"


@pytest.mark.asyncio
async def test_openai_chat_completions_incremental_streaming(async_client):
    async def mock_stream_gen(task_req: TaskRequest):
        yield StreamChunk(delta="Hello", session_id="conv-stream-1")
        yield StreamChunk(delta=" world", session_id="conv-stream-1")
        yield StreamChunk(
            delta="!", is_final=True, session_id="conv-stream-1", finish_reason="stop"
        )

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch_stream", side_effect=mock_stream_gen
    ):
        payload = {
            "model": "agy-gemini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        resp = await async_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        text = resp.text
        assert '{"role": "assistant", "content": "Hello"}' in text
        assert '{"role": "assistant", "content": " world"}' in text
        assert '{"role": "assistant", "content": "!"}' in text
        assert '"finish_reason": "stop"' in text
        assert "data: [DONE]" in text

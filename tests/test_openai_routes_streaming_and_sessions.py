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


@pytest.mark.asyncio
async def test_openai_chat_completions_x_session_affinity_reuses_thread(async_client):
    async def mock_stream_turn1(task_req: TaskRequest):
        assert task_req.session_id is None
        yield StreamChunk(delta="", session_id="t3-thread-affinity-1")
        yield StreamChunk(
            delta="First answer",
            is_final=True,
            session_id="t3-thread-affinity-1",
            finish_reason="stop",
        )

    async def mock_stream_turn2(task_req: TaskRequest):
        assert task_req.session_id == "t3-thread-affinity-1"
        assert task_req.prompt == "Follow-up question in same Slack thread"
        assert task_req.system_prompt is None
        yield StreamChunk(delta="", session_id="t3-thread-affinity-1")
        yield StreamChunk(
            delta="Second answer",
            is_final=True,
            session_id="t3-thread-affinity-1",
            finish_reason="stop",
        )

    calls = [0]

    def dispatch_stream_side_effect(task_req: TaskRequest):
        idx = calls[0]
        calls[0] += 1
        return mock_stream_turn1(task_req) if idx == 0 else mock_stream_turn2(task_req)

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch_stream",
        side_effect=dispatch_stream_side_effect,
    ):
        headers = {"x-session-affinity": "openclaw-slack-thread-uuid-1"}
        resp1 = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "t3",
                "messages": [
                    {"role": "system", "content": "System instructions"},
                    {"role": "user", "content": "Initial Slack thread prompt"},
                ],
                "stream": True,
            },
            headers=headers,
        )
        assert resp1.status_code == 200
        assert resp1.headers.get("x-session-id") == "openclaw-slack-thread-uuid-1"

        resp2 = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "t3",
                "messages": [
                    {"role": "system", "content": "System instructions"},
                    {"role": "user", "content": "Initial Slack thread prompt"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up question in same Slack thread"},
                ],
                "stream": True,
            },
            headers=headers,
        )
        assert resp2.status_code == 200
        assert "Second answer" in resp2.text


@pytest.mark.asyncio
async def test_openai_resumed_turn_preserves_user_prompt_before_internal_context_and_fallback(
    async_client,
):
    await session_store.get_or_create(
        client_id="openclaw-slack-thread-archived",
        executor_type="t3",  # type: ignore[arg-type]
        native_session_id="mcp:archived-thread-999",
    )

    captured_reqs: list[TaskRequest] = []

    async def mock_stream(task_req: TaskRequest):
        captured_reqs.append(task_req)
        yield StreamChunk(delta="", session_id="mcp:replacement-thread-1")
        yield StreamChunk(
            delta="Recovered full context",
            is_final=True,
            session_id="mcp:replacement-thread-1",
            finish_reason="stop",
        )

    with patch(
        "cli_agent_dispatch.api.openai_routes.engine.dispatch_stream",
        side_effect=mock_stream,
    ):
        payload = {
            "model": "t3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a personal assistant running inside OpenClaw.",
                },
                {"role": "user", "content": "Review my Molly messages"},
                {"role": "assistant", "content": "Her interest level is high."},
                {"role": "user", "content": "Trying to work in the Formula 1 lego photo"},
                {
                    "role": "user",
                    "content": (
                        "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\n"
                        "Active exec sessions:\nnone\n"
                        "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>"
                    ),
                },
            ],
            "stream": True,
        }
        resp = await async_client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"x-session-id": "openclaw-slack-thread-archived"},
        )
        assert resp.status_code == 200
        assert "Recovered full context" in resp.text

    assert len(captured_reqs) == 1
    req = captured_reqs[0]
    assert req.session_id == "mcp:archived-thread-999"
    # Delta prompt must include BOTH the user message AND trailing internal context
    assert "Trying to work in the Formula 1 lego photo" in req.prompt
    assert "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" in req.prompt
    assert req.system_prompt is None
    # Fallback prompt & system prompt must retain the full conversation history
    assert req.fallback_system_prompt == "You are a personal assistant running inside OpenClaw."
    assert req.fallback_prompt is not None
    assert "Review my Molly messages" in req.fallback_prompt
    assert "Her interest level is high." in req.fallback_prompt
    assert "Trying to work in the Formula 1 lego photo" in req.fallback_prompt

    # Session store should now point to the replacement T3 thread
    updated = await session_store.get("openclaw-slack-thread-archived")
    assert updated is not None
    assert updated.native_session_id == "mcp:replacement-thread-1"

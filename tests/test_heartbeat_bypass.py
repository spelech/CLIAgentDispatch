"""Tests for OpenClaw heartbeat bypass routing in OpenAI compatibility endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from cli_agent_dispatch.api.openai_routes import is_heartbeat_turn
from cli_agent_dispatch.core.models import (
    ChatCompletionMessage,
    ExecutorType,
    TaskRequest,
    TaskResult,
)
from cli_agent_dispatch.main import app


def test_is_heartbeat_turn_detection():
    # Detects explicit [OpenClaw heartbeat poll]
    msgs1 = [
        ChatCompletionMessage(role="system", content="System instructions"),
        ChatCompletionMessage(role="user", content="[OpenClaw heartbeat poll] 12:00:00"),
    ]
    assert is_heartbeat_turn(msgs1) is True

    # Detects scratch context marker
    msgs2 = [
        ChatCompletionMessage(
            role="user",
            content="Follow the heartbeat monitor scratch context for pending triggers",
        )
    ]
    assert is_heartbeat_turn(msgs2) is True

    # Normal user chat does not trigger heartbeat
    msgs3 = [ChatCompletionMessage(role="user", content="Investigate container logs")]
    assert is_heartbeat_turn(msgs3) is False

    assert is_heartbeat_turn([]) is False


@pytest.mark.asyncio
async def test_heartbeat_bypass_immediate_no_reply_non_streaming():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "t3",
            "messages": [
                {"role": "user", "content": "[OpenClaw heartbeat poll] check status"},
            ],
            "stream": False,
        }

        with patch("cli_agent_dispatch.core.engine.engine.dispatch") as mock_dispatch:
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "NO_REPLY"
            # Engine dispatch must be completely bypassed!
            mock_dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_bypass_immediate_no_reply_streaming():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "t3-agy-flash",
            "messages": [
                {"role": "user", "content": "Follow the heartbeat monitor scratch context"},
            ],
            "stream": True,
        }

        with patch("cli_agent_dispatch.core.engine.engine.dispatch_stream") as mock_stream:
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            content = resp.text
            assert "NO_REPLY" in content
            assert "[DONE]" in content
            mock_stream.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_bypass_with_fast_inference_env_var(monkeypatch):
    monkeypatch.setenv("OPENCLAW_HEARTBEAT_FAST_INFERENCE", "1")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "t3",
            "messages": [
                {"role": "user", "content": "[OpenClaw heartbeat poll] check queues"},
            ],
            "stream": False,
        }

        mock_result = TaskResult(
            success=True,
            executor="agy",
            output="NO_REPLY",
            session_id="ses_fast_123",
        )

        with patch(
            "cli_agent_dispatch.core.engine.engine.dispatch", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = mock_result
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200

            # Verify that executor was switched from T3 to AGY fast inference model
            mock_dispatch.assert_called_once()
            task_req: TaskRequest = mock_dispatch.call_args[0][0]
            assert task_req.executor == ExecutorType.AGY
            assert task_req.model == "gemini-3.8-flash-low"

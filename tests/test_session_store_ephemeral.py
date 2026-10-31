"""Tests for ephemeral session tracking, archive endpoints, 24h reaper, and SRE lifecycle."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from cli_agent_dispatch.core.models import ExecutorType, TaskResult
from cli_agent_dispatch.core.session_store import SessionStore, session_store
from cli_agent_dispatch.main import app, reap_expired_ephemeral_sessions


@pytest.mark.asyncio
async def test_session_record_ephemeral_defaults():
    store = SessionStore()
    rec = await store.get_or_create("client-1", ExecutorType.AGY)
    assert rec.ephemeral is False
    assert rec.expires_at is None
    assert rec.archived is False


@pytest.mark.asyncio
async def test_session_record_ephemeral_and_expiration():
    store = SessionStore()
    now = time.time()
    rec = await store.get_or_create(
        "client-sre",
        ExecutorType.T3,
        native_session_id="mcp:thread-123",
        ephemeral=True,
        expires_at=now + 3600,
    )
    assert rec.ephemeral is True
    assert rec.expires_at == now + 3600
    assert rec.archived is False

    # Not expired yet
    expired = await store.get_expired_ephemeral_sessions(now=now)
    assert len(expired) == 0

    # Expired in future
    expired_future = await store.get_expired_ephemeral_sessions(now=now + 3601)
    assert len(expired_future) == 1
    assert expired_future[0].client_id == "client-sre"

    # Mark archived
    await store.mark_archived("client-sre")
    rec_after = await store.get("client-sre")
    assert rec_after.archived is True

    # No longer returned by get_expired_ephemeral_sessions once archived
    expired_after_archive = await store.get_expired_ephemeral_sessions(now=now + 3601)
    assert len(expired_after_archive) == 0


@pytest.mark.asyncio
async def test_archive_session_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Archive non-existent session
        resp_404 = await client.post("/v1/sessions/non_existent_session_id/archive")
        assert resp_404.status_code == 404

        # 2. Archive T3 session
        await session_store.get_or_create(
            "client_t3_sre",
            ExecutorType.T3,
            native_session_id="mcp:t3_thread_999",
            ephemeral=True,
        )

        mock_t3_client = AsyncMock()
        mock_t3_client.call_tool.return_value = {"organized": True}

        with patch("cli_agent_dispatch.executors.t3.T3McpClient") as MockClientCls:
            MockClientCls.return_value.__aenter__.return_value = mock_t3_client
            MockClientCls.return_value.__aexit__.return_value = None

            resp = await client.post("/v1/sessions/client_t3_sre/archive")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "archived"
            assert data["thread_id"] == "mcp:t3_thread_999"

            mock_t3_client.call_tool.assert_called_once_with(
                "t3_thread_organize",
                {"threadId": "mcp:t3_thread_999", "action": "archive"},
            )

            # Verify session marked archived in store
            record = await session_store.get("client_t3_sre")
            assert record.archived is True


@pytest.mark.asyncio
async def test_reap_expired_ephemeral_sessions():
    now = time.time()
    await session_store.get_or_create(
        "reaper_expired_t3",
        ExecutorType.T3,
        native_session_id="mcp:t3_expired_1",
        ephemeral=True,
        expires_at=now - 100,  # Already expired
    )
    await session_store.get_or_create(
        "reaper_active_t3",
        ExecutorType.T3,
        native_session_id="mcp:t3_active_2",
        ephemeral=True,
        expires_at=now + 5000,  # Not expired
    )

    mock_t3_client = AsyncMock()
    mock_t3_client.call_tool.return_value = {"organized": True}

    with patch("cli_agent_dispatch.executors.t3.T3McpClient") as MockClientCls:
        MockClientCls.return_value.__aenter__.return_value = mock_t3_client
        MockClientCls.return_value.__aexit__.return_value = None

        count = await reap_expired_ephemeral_sessions(now=now)
        assert count == 1

        mock_t3_client.call_tool.assert_called_once_with(
            "t3_thread_organize",
            {"threadId": "mcp:t3_expired_1", "action": "archive"},
        )

        rec_expired = await session_store.get("reaper_expired_t3")
        assert rec_expired.archived is True

        rec_active = await session_store.get("reaper_active_t3")
        assert rec_active.archived is False


@pytest.mark.asyncio
async def test_openai_route_registers_sre_session_as_ephemeral():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        sre_prompt = """=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: homebox
Exit Code / Status: 137 (OOMKilled)
"""
        payload = {
            "model": "t3",
            "messages": [{"role": "user", "content": sre_prompt}],
            "stream": False,
        }

        mock_result = TaskResult(
            success=True,
            executor="t3",
            output='{"root_cause": "OOM"}',
            session_id="mcp:thread_sre_ephemeral_1",
        )

        with patch(
            "cli_agent_dispatch.core.engine.engine.dispatch", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = mock_result
            resp = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "client_sre_turn_1"},
            )
            assert resp.status_code == 200

            # Session should be downgraded from T3 to AGY and registered as ephemeral in session_store
            record = await session_store.get("client_sre_turn_1")
            assert record is not None
            assert record.executor_type == ExecutorType.AGY
            assert record.ephemeral is True
            assert record.expires_at is not None
            assert record.expires_at > time.time() + 86000


@pytest.mark.asyncio
async def test_reaper_handles_t3_error_gracefully():
    now = time.time()
    await session_store.get_or_create(
        "reaper_fail_t3",
        ExecutorType.T3,
        native_session_id="mcp:t3_fail_1",
        ephemeral=True,
        expires_at=now - 50,
    )

    mock_t3_client = AsyncMock()
    mock_t3_client.call_tool.side_effect = RuntimeError("T3 MCP disconnect")

    with patch("cli_agent_dispatch.executors.t3.T3McpClient") as MockClientCls:
        MockClientCls.return_value.__aenter__.return_value = mock_t3_client
        MockClientCls.return_value.__aexit__.return_value = None

        count = await reap_expired_ephemeral_sessions(now=now)
        # Exception was caught, count is 0
        assert count == 0


@pytest.mark.asyncio
async def test_lifespan_starts_and_cancels_reaper():
    from cli_agent_dispatch.main import lifespan

    test_app = MagicMock()
    async with lifespan(test_app):
        pass

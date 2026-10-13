import asyncio
import time

import pytest

from cli_agent_dispatch.core.models import ExecutorType
from cli_agent_dispatch.core.session_store import SessionStore


@pytest.mark.asyncio
async def test_session_store_create_and_get():
    store = SessionStore(default_ttl_seconds=60)
    record = await store.get_or_create(
        client_id="client-123",
        executor_type=ExecutorType.AGY,
        native_session_id="conv-456",
        metadata={"user": "alice"},
    )

    assert record.client_id == "client-123"
    assert record.executor_type == ExecutorType.AGY
    assert record.native_session_id == "conv-456"
    assert record.metadata["user"] == "alice"
    assert record.turn_count == 0

    fetched = await store.get("client-123")
    assert fetched is not None
    assert fetched.client_id == "client-123"
    assert fetched.turn_count == 1  # get() touches and increments turn


@pytest.mark.asyncio
async def test_session_store_update_native_id():
    store = SessionStore(default_ttl_seconds=60)
    await store.get_or_create(
        client_id="thread-abc",
        executor_type=ExecutorType.OPENCODE,
    )

    await store.update_native_id("thread-abc", "ses-real-999")
    fetched = await store.get("thread-abc")
    assert fetched is not None
    assert fetched.native_session_id == "ses-real-999"


@pytest.mark.asyncio
async def test_session_store_ttl_and_prune():
    store = SessionStore(default_ttl_seconds=1)
    await store.get_or_create("expiring-client", ExecutorType.AGY)
    assert await store.count() == 1

    # Sleep slightly past TTL
    await asyncio.sleep(1.1)

    # get() on expired session returns None and cleans it up
    assert await store.get("expiring-client") is None
    assert await store.count() == 0

    # Test prune_expired
    await store.get_or_create("session-1", ExecutorType.AGY)
    await store.get_or_create("session-2", ExecutorType.OPENCODE)
    assert await store.count() == 2

    # Manually backdate session-1
    async with store._lock:
        store._sessions["session-1"].last_active_at = time.time() - 10

    pruned = await store.prune_expired()
    assert pruned == 1
    assert await store.count() == 1
    assert "session-2" in store._sessions


@pytest.mark.asyncio
async def test_session_store_delete():
    store = SessionStore(default_ttl_seconds=60)
    await store.get_or_create("del-me", ExecutorType.AGY)
    assert await store.delete("del-me") is True
    assert await store.delete("nonexistent") is False

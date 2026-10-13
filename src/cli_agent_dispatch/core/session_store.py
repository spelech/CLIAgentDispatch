import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from cli_agent_dispatch.core.models import ExecutorType


class SessionRecord(BaseModel):
    """Represents an active multi-turn agent session."""

    client_id: str = Field(..., description="External client or conversation ID.")
    executor_type: ExecutorType = Field(..., description="Executor engine handling the session.")
    native_session_id: str | None = Field(
        default=None,
        description="Internal executor session/conversation ID (e.g. agy conversation ID).",
    )
    created_at: float = Field(default_factory=time.time)
    last_active_at: float = Field(default_factory=time.time)
    turn_count: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.last_active_at = time.time()
        self.turn_count += 1


class SessionStore:
    """Thread-safe, in-memory session registry with TTL management."""

    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def get(self, client_id: str) -> SessionRecord | None:
        """Retrieve a session by its client ID, updating touch if valid."""
        async with self._lock:
            record = self._sessions.get(client_id)
            if not record:
                return None
            if time.time() - record.last_active_at > self.default_ttl:
                del self._sessions[client_id]
                return None
            record.touch()
            return record

    async def get_or_create(
        self,
        client_id: str,
        executor_type: ExecutorType,
        native_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Retrieve existing session or create a new one."""
        async with self._lock:
            record = self._sessions.get(client_id)
            now = time.time()
            if record and (now - record.last_active_at <= self.default_ttl):
                record.touch()
                if native_session_id and not record.native_session_id:
                    record.native_session_id = native_session_id
                if metadata:
                    record.metadata.update(metadata)
                return record

            new_record = SessionRecord(
                client_id=client_id,
                executor_type=executor_type,
                native_session_id=native_session_id,
                metadata=metadata or {},
            )
            self._sessions[client_id] = new_record
            return new_record

    async def update_native_id(self, client_id: str, native_session_id: str) -> None:
        """Update the underlying native executor session ID for a client ID."""
        async with self._lock:
            if client_id in self._sessions:
                self._sessions[client_id].native_session_id = native_session_id
                self._sessions[client_id].touch()

    async def delete(self, client_id: str) -> bool:
        """Remove a session from the store."""
        async with self._lock:
            return bool(self._sessions.pop(client_id, None))

    async def prune_expired(self) -> int:
        """Remove all expired sessions according to default_ttl."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._sessions.items() if now - v.last_active_at > self.default_ttl
            ]
            for k in expired_keys:
                del self._sessions[k]
            return len(expired_keys)

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)


# Global singleton instance
session_store = SessionStore()

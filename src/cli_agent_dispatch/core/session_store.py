import asyncio
import time
from collections.abc import Awaitable, Callable
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
    ephemeral: bool = Field(
        default=False, description="Whether this session is ephemeral/temporary."
    )
    expires_at: float | None = Field(
        default=None, description="Timestamp after which ephemeral session expires."
    )
    archived: bool = Field(default=False, description="Whether this session has been archived.")

    def touch(self) -> None:
        self.last_active_at = time.time()
        self.turn_count += 1


class SessionStore:
    """Thread-safe, in-memory session registry with TTL management."""

    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        self._sessions: dict[str, SessionRecord] = {}
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancel_callbacks: dict[str, Callable[[], Awaitable[bool]]] = {}
        self._cancelled_runs: set[str] = set()
        self._lock = asyncio.Lock()

    def register_process(self, client_id: str, proc: asyncio.subprocess.Process) -> None:
        """Register a running OS subprocess associated with a session or run ID."""
        self._active_processes[client_id] = proc
        self._cancelled_runs.discard(client_id)

    def unregister_process(self, client_id: str) -> None:
        """Unregister an OS subprocess once execution completes."""
        self._active_processes.pop(client_id, None)

    def register_cancel_callback(
        self, client_id: str, callback: Callable[[], Awaitable[bool]]
    ) -> None:
        """Register an async cancellation callback (e.g., for T3 thread interruption)."""
        self._cancel_callbacks[client_id] = callback
        self._cancelled_runs.discard(client_id)

    def unregister_cancel_callback(self, client_id: str) -> None:
        """Unregister an async cancellation callback once execution completes."""
        self._cancel_callbacks.pop(client_id, None)

    def was_cancelled(self, client_id: str) -> bool:
        """Return True if a run/session was explicitly cancelled by user."""
        return client_id in self._cancelled_runs

    def active_process_ids(self) -> list[str]:
        """Return list of currently active process or run keys."""
        proc_ids = [
            k for k, proc in list(self._active_processes.items()) if proc.returncode is None
        ]
        cb_ids = [k for k in self._cancel_callbacks if k not in proc_ids]
        return proc_ids + cb_ids

    async def kill_active_process(self, client_id: str) -> bool:
        """Terminate and kill any running OS subprocess or active callback for a session."""
        cb = self._cancel_callbacks.pop(client_id, None)
        proc = self._active_processes.pop(client_id, None)
        if not proc and not cb:
            return False
        self._cancelled_runs.add(client_id)
        killed = False
        if cb:
            try:
                killed = await cb() or True
            except Exception:
                killed = True
        if proc:
            try:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.5)
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
                killed = True
            except Exception:
                pass
        return killed

    async def cancel_all_active(self) -> list[str]:
        """Terminate and kill all currently running OS subprocesses and callbacks across active turns."""
        cancelled: list[str] = []
        active_keys = list(
            dict.fromkeys(list(self._active_processes.keys()) + list(self._cancel_callbacks.keys()))
        )
        for key in active_keys:
            if await self.kill_active_process(key):
                cancelled.append(key)
        return cancelled

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
        ephemeral: bool = False,
        expires_at: float | None = None,
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
                if ephemeral:
                    record.ephemeral = True
                if expires_at is not None:
                    record.expires_at = expires_at
                return record

            new_record = SessionRecord(
                client_id=client_id,
                executor_type=executor_type,
                native_session_id=native_session_id,
                metadata=metadata or {},
                ephemeral=ephemeral,
                expires_at=expires_at,
            )
            self._sessions[client_id] = new_record
            return new_record

    async def get_expired_ephemeral_sessions(self, now: float | None = None) -> list[SessionRecord]:
        """Return list of ephemeral sessions whose expiration time has passed and are not yet archived."""
        current_time = now if now is not None else time.time()
        async with self._lock:
            return [
                rec
                for rec in self._sessions.values()
                if rec.ephemeral
                and not rec.archived
                and rec.expires_at is not None
                and rec.expires_at <= current_time
            ]

    async def mark_archived(self, client_id: str) -> None:
        """Mark a session record as archived."""
        async with self._lock:
            record = self._sessions.get(client_id)
            if record:
                record.archived = True
            for rec in self._sessions.values():
                if rec.native_session_id == client_id:
                    rec.archived = True

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

    async def list_all(self) -> list[SessionRecord]:
        """List all active, unexpired session records."""
        async with self._lock:
            now = time.time()
            return [
                v for v in self._sessions.values() if now - v.last_active_at <= self.default_ttl
            ]

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)


# Global singleton instance
session_store = SessionStore()

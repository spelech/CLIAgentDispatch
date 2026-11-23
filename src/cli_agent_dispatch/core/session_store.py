import asyncio
import json
import os
import sqlite3
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

    def __init__(self, default_ttl_seconds: int = 86400, db_path: str | None = None):
        self.default_ttl = default_ttl_seconds
        self._sessions: dict[str, SessionRecord] = {}
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancel_callbacks: dict[str, Callable[[], Awaitable[bool]]] = {}
        self._cancelled_runs: set[str] = set()
        self._lock = asyncio.Lock()
        self._db_path: str | None = None
        if db_path:
            self.attach_db(db_path)

    def attach_db(self, db_path: str) -> None:
        """Attach a SQLite database file and load unexpired sessions into memory."""
        self._db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    client_id TEXT PRIMARY KEY,
                    executor_type TEXT NOT NULL,
                    native_session_id TEXT,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    ephemeral INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            now = time.time()
            rows = conn.execute(
                "SELECT client_id, executor_type, native_session_id, created_at, last_active_at, "
                "turn_count, metadata_json, ephemeral, expires_at, archived FROM sessions"
            ).fetchall()
            expired_ids: list[str] = []
            for row in rows:
                (
                    cid,
                    exec_val,
                    native_id,
                    created_at,
                    last_active_at,
                    turn_count,
                    metadata_json,
                    ephemeral,
                    expires_at,
                    archived,
                ) = row
                if now - last_active_at > self.default_ttl:
                    expired_ids.append(cid)
                    continue
                try:
                    meta = json.loads(metadata_json) if metadata_json else {}
                except Exception:
                    meta = {}
                self._sessions[cid] = SessionRecord(
                    client_id=cid,
                    executor_type=ExecutorType(exec_val),
                    native_session_id=native_id,
                    created_at=created_at,
                    last_active_at=last_active_at,
                    turn_count=turn_count,
                    metadata=meta,
                    ephemeral=bool(ephemeral),
                    expires_at=expires_at,
                    archived=bool(archived),
                )
            if expired_ids:
                conn.executemany(
                    "DELETE FROM sessions WHERE client_id = ?", [(cid,) for cid in expired_ids]
                )
            conn.commit()

    def _persist_record(self, rec: SessionRecord) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    client_id, executor_type, native_session_id, created_at, last_active_at,
                    turn_count, metadata_json, ephemeral, expires_at, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    executor_type = excluded.executor_type,
                    native_session_id = excluded.native_session_id,
                    last_active_at = excluded.last_active_at,
                    turn_count = excluded.turn_count,
                    metadata_json = excluded.metadata_json,
                    ephemeral = excluded.ephemeral,
                    expires_at = excluded.expires_at,
                    archived = excluded.archived
                """,
                (
                    rec.client_id,
                    rec.executor_type.value,
                    rec.native_session_id,
                    rec.created_at,
                    rec.last_active_at,
                    rec.turn_count,
                    json.dumps(rec.metadata),
                    1 if rec.ephemeral else 0,
                    rec.expires_at,
                    1 if rec.archived else 0,
                ),
            )
            conn.commit()

    def _delete_from_db(self, client_ids: list[str]) -> None:
        if not self._db_path or not client_ids:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "DELETE FROM sessions WHERE client_id = ?", [(cid,) for cid in client_ids]
            )
            conn.commit()

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
                self._delete_from_db([client_id])
                return None
            record.touch()
            self._persist_record(record)
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
                self._persist_record(record)
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
            self._persist_record(new_record)
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
                self._persist_record(record)
            for rec in self._sessions.values():
                if rec.native_session_id == client_id:
                    rec.archived = True
                    self._persist_record(rec)

    async def update_native_id(self, client_id: str, native_session_id: str) -> None:
        """Update the underlying native executor session ID for a client ID."""
        async with self._lock:
            if client_id in self._sessions:
                self._sessions[client_id].native_session_id = native_session_id
                self._sessions[client_id].touch()
                self._persist_record(self._sessions[client_id])

    async def get_by_any_id(self, session_id: str) -> SessionRecord | None:
        """Look up an unexpired session by client_id or native_session_id."""
        async with self._lock:
            now = time.time()
            rec = self._sessions.get(session_id)
            if rec and (now - rec.last_active_at <= self.default_ttl):
                return rec
            for candidate in self._sessions.values():
                if candidate.native_session_id == session_id and (
                    now - candidate.last_active_at <= self.default_ttl
                ):
                    return candidate
            return None

    async def update_metadata(self, client_id: str, updates: dict[str, Any]) -> None:
        """Merge metadata fields into an existing session record."""
        async with self._lock:
            rec = self._sessions.get(client_id)
            if rec:
                rec.metadata.update(updates)
                rec.last_active_at = time.time()
                self._persist_record(rec)

    def record_step_snapshot(self, client_id: str, step: dict[str, Any]) -> None:
        """Record a recent tool step in memory on the session record for T3 handoff context."""
        rec = self._sessions.get(client_id)
        if not rec:
            for candidate in self._sessions.values():
                if candidate.native_session_id == client_id:
                    rec = candidate
                    break
        if not rec:
            return
        steps = rec.metadata.setdefault("recent_steps", [])
        if not isinstance(steps, list):
            steps = []
            rec.metadata["recent_steps"] = steps
        step_idx = step.get("step_index")
        for i, existing in enumerate(steps):
            if (
                isinstance(existing, dict)
                and step_idx is not None
                and existing.get("step_index") == step_idx
            ):
                steps[i] = dict(step)
                return
        steps.append(dict(step))
        if len(steps) > 20:
            del steps[:-20]

    async def upgrade_executor(
        self,
        client_id: str,
        executor_type: ExecutorType,
        native_session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Upgrade/bind an existing or new session record to a new executor and native session ID."""
        async with self._lock:
            rec = self._sessions.get(client_id)
            if rec:
                rec.executor_type = executor_type
                rec.native_session_id = native_session_id
                rec.archived = False
                if metadata:
                    rec.metadata.update(metadata)
                rec.touch()
                self._persist_record(rec)
                return rec
            new_rec = SessionRecord(
                client_id=client_id,
                executor_type=executor_type,
                native_session_id=native_session_id,
                metadata=metadata or {},
            )
            self._sessions[client_id] = new_rec
            self._persist_record(new_rec)
            return new_rec

    async def delete(self, client_id: str) -> bool:
        """Remove a session from the store."""
        async with self._lock:
            existed = bool(self._sessions.pop(client_id, None))
            if existed:
                self._delete_from_db([client_id])
            return existed

    async def prune_expired(self) -> int:
        """Remove all expired sessions according to default_ttl."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._sessions.items() if now - v.last_active_at > self.default_ttl
            ]
            for k in expired_keys:
                del self._sessions[k]
            self._delete_from_db(expired_keys)
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

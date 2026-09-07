from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DiagnosticEvent(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    event_type: str
    executor: str
    duration_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticRingBuffer:
    """Thread-safe ring buffer for recording recent execution events."""

    def __init__(self, capacity: int = 100) -> None:
        self._buffer: deque[DiagnosticEvent] = deque(maxlen=capacity)

    def record(self, event: DiagnosticEvent) -> None:
        self._buffer.append(event)

    def get_events(self, limit: int = 50) -> list[DiagnosticEvent]:
        return list(self._buffer)[-limit:]


diagnostic_buffer = DiagnosticRingBuffer(capacity=100)

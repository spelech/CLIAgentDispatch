from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult


class BaseExecutor(ABC):
    """Abstract base executor for AI delegation engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the executor."""
        pass

    @abstractmethod
    async def execute(self, request: TaskRequest) -> TaskResult:
        """Execute a task request and return the result."""
        pass

    @abstractmethod
    async def execute_stream(
        self, request: TaskRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream output chunks from the executor engine."""
        pass


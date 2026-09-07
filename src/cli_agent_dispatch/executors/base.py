from abc import ABC, abstractmethod

from cli_agent_dispatch.core.models import TaskRequest, TaskResult


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

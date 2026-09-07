import logging

from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest, TaskResult
from cli_agent_dispatch.executors.agy import AgyExecutor
from cli_agent_dispatch.executors.base import BaseExecutor
from cli_agent_dispatch.executors.opencode import OpenCodeExecutor

logger = logging.getLogger("CLIAgentDispatch.Engine")


class DispatchEngine:
    """Central orchestration engine managing executors and routing task requests."""

    def __init__(self) -> None:
        self._executors: dict[str, BaseExecutor] = {
            ExecutorType.OPENCODE.value: OpenCodeExecutor(),
            ExecutorType.AGY.value: AgyExecutor(),
        }

    def register_executor(self, executor: BaseExecutor) -> None:
        self._executors[executor.name] = executor

    def get_executor(self, name: str) -> BaseExecutor:
        executor = self._executors.get(name.lower())
        if not executor:
            raise ExecutorNotFoundError(
                f"Executor '{name}' not found. Available: {list(self._executors.keys())}"
            )
        return executor

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """Route and execute a task request against the target executor."""
        executor = self.get_executor(request.executor.value)
        logger.info(
            f"Dispatching task to executor='{executor.name}', model='{request.model}', "
            f"workspace='{request.workspace}'"
        )
        result = await executor.execute(request)
        if result.success:
            logger.info(
                f"Task succeeded via '{executor.name}' in {result.duration_seconds}s"
            )
        else:
            logger.warning(
                f"Task failed via '{executor.name}' in {result.duration_seconds}s: {result.error}"
            )
        return result


# Singleton instance
engine = DispatchEngine()


async def delegate_task(
    prompt: str,
    executor: str = "opencode",
    workspace: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
) -> TaskResult:
    """Convenience functional helper for external Python callers (e.g. monitorbot)."""
    req = TaskRequest(
        prompt=prompt,
        executor=ExecutorType(executor.lower()),
        workspace=workspace,
        model=model,
        provider=provider,
        timeout=timeout,
    )
    return await engine.dispatch(req)

import asyncio
import os
import time

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import TaskRequest, TaskResult
from cli_agent_dispatch.executors.base import BaseExecutor


class AgyExecutor(BaseExecutor):
    """Executor for Antigravity (agy) CLI agent."""

    @property
    def name(self) -> str:
        return "agy"

    async def execute(self, request: TaskRequest) -> TaskResult:
        start_time = time.perf_counter()
        timeout = request.timeout or settings.default_timeout
        model = request.model or settings.agy_default_model
        workspace = request.workspace or settings.default_workspace
        agy_path = settings.agy_path

        if not os.path.exists(agy_path):
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"agy binary not found at {agy_path}",
                duration_seconds=round(duration, 3),
            )

        if workspace and not os.path.exists(workspace):
            workspace = os.getcwd()

        cmd = [
            agy_path,
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--print",
            request.prompt,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            duration = time.perf_counter() - start_time

            if process.returncode == 0:
                return TaskResult(
                    success=True,
                    executor=self.name,
                    output=stdout.decode("utf-8", errors="replace"),
                    duration_seconds=round(duration, 3),
                    metadata={"model": model, "workspace": workspace},
                )
            else:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                return TaskResult(
                    success=False,
                    executor=self.name,
                    output=stdout.decode("utf-8", errors="replace"),
                    error=f"agy exited with code {process.returncode}: {err_msg}",
                    duration_seconds=round(duration, 3),
                    metadata={"model": model, "returncode": process.returncode},
                )

        except TimeoutError:
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"agy execution timed out after {timeout} seconds",
                duration_seconds=round(duration, 3),
            )
        except Exception as ex:
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"agy execution error: {ex}",
                duration_seconds=round(duration, 3),
            )

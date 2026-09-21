import asyncio
import json
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

        # Handle system_prompt on turn 1 (session_id is None)
        if not request.session_id and request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        cmd = [
            agy_path,
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
        ]
        if request.session_id:
            cmd.extend(["--conversation", request.session_id])
        cmd.extend(["--print", prompt])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            duration = time.perf_counter() - start_time
            raw_stdout = stdout.decode("utf-8", errors="replace")

            if process.returncode == 0:
                output_text = raw_stdout
                session_id = request.session_id
                metadata = {"model": model, "workspace": workspace}

                try:
                    data = json.loads(raw_stdout.strip())
                    if isinstance(data, dict):
                        if "conversation_id" in data and data["conversation_id"]:
                            session_id = data["conversation_id"]
                        if "response" in data and data["response"] is not None:
                            output_text = str(data["response"])
                        if "num_turns" in data:
                            metadata["num_turns"] = data["num_turns"]
                except (json.JSONDecodeError, TypeError, ValueError):
                    output_text = raw_stdout

                return TaskResult(
                    success=True,
                    executor=self.name,
                    output=output_text,
                    session_id=session_id,
                    duration_seconds=round(duration, 3),
                    metadata=metadata,
                )
            else:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                return TaskResult(
                    success=False,
                    executor=self.name,
                    output=raw_stdout,
                    session_id=request.session_id,
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
                session_id=request.session_id,
                error=f"agy execution timed out after {timeout} seconds",
                duration_seconds=round(duration, 3),
            )
        except Exception as ex:
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                session_id=request.session_id,
                error=f"agy execution error: {ex}",
                duration_seconds=round(duration, 3),
            )

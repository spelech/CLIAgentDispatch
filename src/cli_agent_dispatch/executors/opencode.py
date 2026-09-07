import asyncio
import os
import time

import httpx

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import TaskRequest, TaskResult
from cli_agent_dispatch.executors.base import BaseExecutor


class OpenCodeExecutor(BaseExecutor):
    """Executor for OpenCode via headless HTTP server API with CLI subprocess fallback."""

    @property
    def name(self) -> str:
        return "opencode"

    async def execute(self, request: TaskRequest) -> TaskResult:
        start_time = time.perf_counter()
        timeout = request.timeout or settings.default_timeout
        provider = request.provider or settings.opencode_default_provider
        model = request.model or settings.opencode_default_model
        workspace = request.workspace or settings.default_workspace

        # Ensure workspace exists
        if workspace and not os.path.exists(workspace):
            workspace = os.getcwd()

        server_url = settings.opencode_server_url.rstrip("/")
        session_id = None

        # 1. Attempt HTTP API
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Create session (with directory context if supported)
                session_payload = {"directory": workspace} if workspace else {}
                session_resp = await client.post(
                    f"{server_url}/session", json=session_payload, timeout=10.0
                )
                session_resp.raise_for_status()
                session_id = session_resp.json().get("id")

                # Send prompt message
                msg_payload = {
                    "model": {
                        "providerID": provider,
                        "modelID": model,
                    },
                    "parts": [{"type": "text", "text": request.prompt}],
                }
                msg_resp = await client.post(
                    f"{server_url}/session/{session_id}/message",
                    json=msg_payload,
                    timeout=timeout,
                )
                msg_resp.raise_for_status()

                output_parts = []
                for part in msg_resp.json().get("parts", []):
                    if part.get("type") == "text":
                        output_parts.append(part.get("text", ""))

                output = "\n".join(output_parts)
                duration = time.perf_counter() - start_time
                return TaskResult(
                    success=True,
                    executor=self.name,
                    output=output,
                    duration_seconds=round(duration, 3),
                    session_id=session_id,
                    metadata={"mode": "http_api", "model": model, "provider": provider},
                )

        except Exception:
            # Fall back to CLI execution if HTTP server failed
            pass

        # 2. CLI Subprocess Fallback
        cli_path = settings.opencode_cli_path
        if not os.path.exists(cli_path):
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"OpenCode server HTTP failed and CLI binary not found at {cli_path}",
                duration_seconds=round(duration, 3),
            )

        cmd = [cli_path, "run", request.prompt]
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
                    metadata={"mode": "cli_fallback", "workspace": workspace},
                )
            else:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                return TaskResult(
                    success=False,
                    executor=self.name,
                    output=stdout.decode("utf-8", errors="replace"),
                    error=f"CLI exited with code {process.returncode}: {err_msg}",
                    duration_seconds=round(duration, 3),
                    metadata={"mode": "cli_fallback", "returncode": process.returncode},
                )

        except TimeoutError:
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"CLI execution timed out after {timeout} seconds",
                duration_seconds=round(duration, 3),
                metadata={"mode": "cli_fallback"},
            )
        except Exception as ex:
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                error=f"OpenCode CLI execution error: {ex}",
                duration_seconds=round(duration, 3),
                metadata={"mode": "cli_fallback"},
            )

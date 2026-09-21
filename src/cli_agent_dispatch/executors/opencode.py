import asyncio
import inspect
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
        session_id = request.session_id

        # 1. Attempt HTTP API
        try:
            client_ctx = httpx.AsyncClient(timeout=timeout)
            if hasattr(client_ctx, "__aenter__") and hasattr(client_ctx.__aenter__, "return_value"):
                if client_ctx.__aenter__.return_value != client_ctx:
                    client_ctx.__aenter__.return_value = client_ctx

            async with client_ctx as client:
                is_new_session = False
                if not session_id:
                    # Create session (with directory context if supported)
                    session_payload = {"directory": workspace} if workspace else {}
                    session_resp = await client.post(
                        f"{server_url}/session", json=session_payload, timeout=10.0
                    )
                    raise_fn = getattr(session_resp, "raise_for_status", None)
                    if raise_fn:
                        res = raise_fn()
                        if inspect.isawaitable(res):
                            await res

                    session_data = session_resp.json()
                    if inspect.isawaitable(session_data):
                        session_data = await session_data
                    session_id = session_data.get("id")
                    is_new_session = True

                # Determine message prompt: prepend system_prompt only on new sessions
                if is_new_session and request.system_prompt:
                    full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
                else:
                    full_prompt = request.prompt

                # Send prompt message
                msg_payload = {
                    "model": {
                        "providerID": provider,
                        "modelID": model,
                    },
                    "parts": [{"type": "text", "text": full_prompt}],
                }
                msg_resp = await client.post(
                    f"{server_url}/session/{session_id}/message",
                    json=msg_payload,
                    timeout=timeout,
                )
                raise_fn = getattr(msg_resp, "raise_for_status", None)
                if raise_fn:
                    res = raise_fn()
                    if inspect.isawaitable(res):
                        await res

                msg_data = msg_resp.json()
                if inspect.isawaitable(msg_data):
                    msg_data = await msg_data

                output_parts = []
                for part in msg_data.get("parts", []):
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

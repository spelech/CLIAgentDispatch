import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.executors.base import BaseExecutor


def normalize_agy_model(model: str | None) -> str:
    """Normalize incoming model alias to canonical agy model string."""
    if not model:
        return settings.agy_default_model
    m = model.lower().strip()
    if "opus" in m:
        return "Claude Opus 5.5 (Medium)"
    if "sonnet" in m or "claude" in m:
        return "Claude Sonnet 5.5 (Medium)"
    if "oss" in m or "gpt" in m:
        return "GPT-OSS 120B (Medium)"
    if "gemini" in m or "flash" in m:
        if "low" in m:
            return "Gemini 3.8 Flash (Low)"
        if "high" in m:
            return "Gemini 3.8 Flash (High)"
        return "Gemini 3.8 Flash (Medium)"
    return model


class AgyExecutor(BaseExecutor):
    """Executor for Antigravity (agy) CLI agent."""

    @property
    def name(self) -> str:
        return "agy"

    async def execute(self, request: TaskRequest) -> TaskResult:
        start_time = time.perf_counter()
        timeout = request.timeout or settings.default_timeout
        model = normalize_agy_model(request.model)
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

        env = os.environ.copy()
        if workspace:
            env["PWD"] = workspace

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                env=env,
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
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
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

    async def execute_stream(self, request: TaskRequest) -> AsyncGenerator[StreamChunk, None]:
        timeout = request.timeout or settings.default_timeout
        model = normalize_agy_model(request.model)
        workspace = request.workspace or settings.default_workspace
        agy_path = settings.agy_path

        if not os.path.exists(agy_path):
            yield StreamChunk(
                delta=f"agy binary not found at {agy_path}",
                is_final=True,
                finish_reason="error",
            )
            return

        if workspace and not os.path.exists(workspace):
            workspace = os.getcwd()

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
            "stream-json",
        ]
        if request.session_id:
            cmd.extend(["--conversation", request.session_id])
        cmd.extend(["--print", prompt])

        process = None
        session_id = request.session_id
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def _stream_reader():
                nonlocal session_id
                assert process.stdout is not None
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        if isinstance(data, dict):
                            event_type = data.get("type")
                            if "conversation_id" in data and data["conversation_id"]:
                                session_id = data["conversation_id"]

                            if event_type == "init":
                                if session_id:
                                    yield StreamChunk(delta="", session_id=session_id)
                            elif event_type == "step_update":
                                delta_text = data.get("text_delta") or data.get("delta") or ""
                                if delta_text:
                                    yield StreamChunk(delta=delta_text, session_id=session_id)
                            elif event_type == "result":
                                usage = data.get("usage")
                                yield StreamChunk(
                                    delta="",
                                    is_final=True,
                                    session_id=session_id,
                                    finish_reason="stop",
                                    usage=usage,
                                )
                                return
                            elif "text_delta" in data:
                                yield StreamChunk(delta=data["text_delta"], session_id=session_id)
                            elif "response" in data and data["response"]:
                                yield StreamChunk(
                                    delta=str(data["response"]), session_id=session_id
                                )
                            continue
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    yield StreamChunk(delta=f"{line_str}\n", session_id=session_id)

            async for chunk in _stream_reader():
                yield chunk

            await asyncio.wait_for(process.wait(), timeout=timeout)
            yield StreamChunk(
                delta="",
                is_final=True,
                session_id=session_id,
                finish_reason="stop",
            )

        except TimeoutError:
            if process:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            yield StreamChunk(
                delta=f"\n[agy execution timed out after {timeout}s]",
                is_final=True,
                session_id=session_id,
                finish_reason="stop",
            )
        except Exception as ex:
            if process:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            yield StreamChunk(
                delta=f"\n[agy execution error: {ex}]",
                is_final=True,
                session_id=session_id,
                finish_reason="stop",
            )

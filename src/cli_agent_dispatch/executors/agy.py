import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import AsyncGenerator

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.executors.base import BaseExecutor

DEFAULT_AGY_MODELS: dict[str, str] = {
    "gemini-4-argon-low": "Gemini 4 Argon (Low)",
    "gemini-4-argon-medium": "Gemini 4 Argon (Medium)",
    "gemini-4-argon-high": "Gemini 4 Argon (High)",
    "gemini-3.8-flash-low": "Gemini 3.8 Flash (Low)",
    "gemini-3.8-flash-medium": "Gemini 3.8 Flash (Medium)",
    "gemini-3.8-flash-high": "Gemini 3.8 Flash (High)",
    "claude-sonnet-5-5-low": "Claude Sonnet 5.5 (Low)",
    "claude-sonnet-5-5-medium": "Claude Sonnet 5.5 (Medium)",
    "claude-sonnet-5-5-high": "Claude Sonnet 5.5 (High)",
    "claude-opus-5-5-low": "Claude Opus 5.5 (Low)",
    "claude-opus-5-5-medium": "Claude Opus 5.5 (Medium)",
    "claude-opus-5-5-high": "Claude Opus 5.5 (High)",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
}

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
_MODEL_LINE_RE = re.compile(r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)\s{2,}(.+)$")
_AGY_MODELS_CACHE: dict[str, str] | None = None
_AGY_MODELS_CACHE_TS: float = 0.0
_AGY_MODELS_CACHE_TTL: float = 3600.0


def _resolve_agy_binary() -> str | None:
    if settings.agy_path and os.path.exists(settings.agy_path):
        return settings.agy_path
    search_paths = [
        os.environ.get("PATH", ""),
        "/home/steve/.local/bin",
        os.path.expanduser("~/.local/bin"),
    ]
    combined_path = os.pathsep.join(p for p in search_paths if p)
    found = shutil.which("agy", path=combined_path)
    if found:
        return found
    fallback = "/home/steve/.local/bin/agy"
    if os.path.exists(fallback):
        return fallback
    return None


def _parse_agy_models_output(raw_output: str) -> dict[str, str]:
    cleaned = _ANSI_ESCAPE_RE.sub("", raw_output or "")
    parsed: dict[str, str] = {}
    for raw_line in cleaned.splitlines():
        # Strip carriage-return spinner frames on the same line
        segments = raw_line.split("\r")
        line = segments[-1].strip() if segments else ""
        if not line or "fetching available models" in line.lower():
            continue
        match = _MODEL_LINE_RE.match(line)
        if match:
            model_id = match.group(1).strip()
            display_name = match.group(2).strip()
            if model_id and display_name:
                parsed[model_id] = display_name
    return parsed


def discover_agy_models(force_refresh: bool = False) -> dict[str, str]:
    """Discover available agy models via `agy models` with 3600s cache and fallback."""
    global _AGY_MODELS_CACHE, _AGY_MODELS_CACHE_TS

    now = time.monotonic()
    if (
        not force_refresh
        and _AGY_MODELS_CACHE is not None
        and (now - _AGY_MODELS_CACHE_TS) < _AGY_MODELS_CACHE_TTL
    ):
        return dict(_AGY_MODELS_CACHE)

    models: dict[str, str] = dict(DEFAULT_AGY_MODELS)
    agy_bin = _resolve_agy_binary()
    if agy_bin:
        env = os.environ.copy()
        local_bin = "/home/steve/.local/bin"
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}{os.pathsep}{env.get('PATH', '')}"
        try:
            proc = subprocess.run(
                [agy_bin, "models"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                discovered = _parse_agy_models_output(proc.stdout)
                if discovered:
                    models.update(discovered)
        except Exception:
            pass

    _AGY_MODELS_CACHE = dict(models)
    _AGY_MODELS_CACHE_TS = now
    return dict(models)


def normalize_agy_model(model: str | None) -> str:
    """Normalize incoming model alias or ID to canonical agy model Display Name."""
    if not model:
        return settings.agy_default_model

    raw = model.strip()
    for prefix in ("openai/", "litellm/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break

    if not raw:
        return settings.agy_default_model

    m = raw.lower()
    if m in ["agy", "default", "auto", "agy-gemini"]:
        return settings.agy_default_model

    # Check exact match (case-insensitive) against discovered model IDs or Display Names
    discovered = discover_agy_models()
    for model_id, display_name in discovered.items():
        if m == model_id.lower() or m == display_name.lower():
            return display_name

    # Argon family matching (check before generic "gemini" matching)
    if "argon" in m:
        if "high" in m:
            return "Gemini 4 Argon (High)"
        if "low" in m:
            return "Gemini 4 Argon (Low)"
        return "Gemini 4 Argon (Medium)"

    if "opus" in m:
        if m == "agy-opus" or "high" in m:
            return "Claude Opus 5.5 (High)"
        if "low" in m:
            return "Claude Opus 5.5 (Low)"
        return "Claude Opus 5.5 (Medium)"

    if "sonnet" in m or "claude" in m:
        if m == "agy-sonnet" or "high" in m:
            return "Claude Sonnet 5.5 (High)"
        if "low" in m:
            return "Claude Sonnet 5.5 (Low)"
        return "Claude Sonnet 5.5 (Medium)"

    if "oss" in m or "gpt" in m:
        return "GPT-OSS 120B (Medium)"

    if "gemini" in m or "flash" in m:
        if "low" in m:
            return "Gemini 3.8 Flash (Low)"
        if "high" in m:
            return "Gemini 3.8 Flash (High)"
        return "Gemini 3.8 Flash (Medium)"

    return raw


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
        if (
            not request.session_id
            and request.system_prompt
            and request.system_prompt not in request.prompt
        ):
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        use_stdin = len(prompt.encode("utf-8")) > 100_000
        cmd = [
            agy_path,
            "--model",
            model,
            "--dangerously-skip-permissions",
        ]
        if request.session_id:
            cmd.extend(["--conversation", request.session_id])
        if use_stdin:
            cmd.extend(
                [
                    "--input-format",
                    "stream-json",
                    "--output-format",
                    "stream-json",
                    "--print",
                    "",
                ]
            )
            stdin_bytes = (
                json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
            ).encode("utf-8")
        else:
            cmd.extend(["--output-format", "json", "--print", prompt])
            stdin_bytes = None

        env = os.environ.copy()
        if workspace:
            env["PWD"] = workspace

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE if use_stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if request.session_id:
                session_store.register_process(request.session_id, process)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_bytes), timeout=timeout
                )
            finally:
                if request.session_id:
                    session_store.unregister_process(request.session_id)
            duration = time.perf_counter() - start_time
            raw_stdout = stdout.decode("utf-8", errors="replace")

            if process.returncode == 0:
                output_text = raw_stdout
                session_id = request.session_id
                metadata = {"model": model, "workspace": workspace}

                try:
                    lines = [ln.strip() for ln in raw_stdout.splitlines() if ln.strip()]
                    data = json.loads(lines[-1] if use_stdin and lines else raw_stdout.strip())
                    if isinstance(data, dict):
                        res_obj = (
                            data.get("result") if isinstance(data.get("result"), dict) else data
                        )
                        if "conversation_id" in res_obj and res_obj["conversation_id"]:
                            session_id = res_obj["conversation_id"]
                        elif "conversation_id" in data and data["conversation_id"]:
                            session_id = data["conversation_id"]
                        if "response" in res_obj and res_obj["response"] is not None:
                            output_text = str(res_obj["response"])
                        if "num_turns" in res_obj:
                            metadata["num_turns"] = res_obj["num_turns"]
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

        if (
            not request.session_id
            and request.system_prompt
            and request.system_prompt not in request.prompt
        ):
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        use_stdin = len(prompt.encode("utf-8")) > 100_000
        cmd = [
            agy_path,
            "--model",
            model,
            "--dangerously-skip-permissions",
        ]
        if request.session_id:
            cmd.extend(["--conversation", request.session_id])
        if use_stdin:
            cmd.extend(
                [
                    "--input-format",
                    "stream-json",
                    "--output-format",
                    "stream-json",
                    "--print",
                    "",
                ]
            )
        else:
            cmd.extend(["--output-format", "stream-json", "--print", prompt])

        process = None
        session_id = request.session_id
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdin=asyncio.subprocess.PIPE if use_stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if use_stdin and process.stdin is not None:
                payload_line = (
                    json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
                ).encode("utf-8")
                process.stdin.write(payload_line)
                await process.stdin.drain()
                process.stdin.close()

            async def _stream_reader():
                nonlocal session_id
                assert process.stdout is not None
                yielded_deltas = False
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
                            event_type = data.get("event") or data.get("type")
                            if "conversation_id" in data and data["conversation_id"]:
                                session_id = data["conversation_id"]

                            if event_type == "init":
                                if session_id:
                                    yield StreamChunk(delta="", session_id=session_id)
                            elif event_type == "step_update":
                                step_obj = (
                                    data.get("step_update")
                                    if isinstance(data.get("step_update"), dict)
                                    else data
                                )
                                if "conversation_id" in step_obj and step_obj["conversation_id"]:
                                    session_id = step_obj["conversation_id"]
                                delta_text = (
                                    step_obj.get("text_delta") or step_obj.get("delta") or ""
                                )
                                if delta_text:
                                    yielded_deltas = True
                                    yield StreamChunk(delta=delta_text, session_id=session_id)
                            elif event_type == "result":
                                res_obj = (
                                    data.get("result")
                                    if isinstance(data.get("result"), dict)
                                    else data
                                )
                                if "conversation_id" in res_obj and res_obj["conversation_id"]:
                                    session_id = res_obj["conversation_id"]
                                usage = res_obj.get("usage") or data.get("usage")
                                if res_obj.get("status") == "ERROR":
                                    err_detail = res_obj.get("error") or "agy stream error"
                                    yield StreamChunk(
                                        delta=f"[agy error: {err_detail}]",
                                        is_final=True,
                                        session_id=session_id,
                                        finish_reason="error",
                                        usage=usage,
                                    )
                                    return
                                if not yielded_deltas and res_obj.get("response"):
                                    yield StreamChunk(
                                        delta=str(res_obj["response"]),
                                        session_id=session_id,
                                    )
                                yield StreamChunk(
                                    delta="",
                                    is_final=True,
                                    session_id=session_id,
                                    finish_reason="stop",
                                    usage=usage,
                                )
                                return
                            elif "text_delta" in data:
                                yielded_deltas = True
                                yield StreamChunk(delta=data["text_delta"], session_id=session_id)
                            elif "response" in data and data["response"]:
                                yielded_deltas = True
                                yield StreamChunk(
                                    delta=str(data["response"]), session_id=session_id
                                )
                            continue
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    if line_str.startswith("AGY_ERROR:") or line_str.startswith("error:"):
                        continue
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

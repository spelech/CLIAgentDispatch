import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.executors.base import BaseExecutor

logger = logging.getLogger("CLIAgentDispatch.AgyExecutor")

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
_SESSION_KEY_RE = re.compile(r"\bsession=([^\s|]+)")
_SLACK_CHANNEL_RE = re.compile(r"\b(?:channel_id|slack_channel)=([CDG][A-Z0-9]{8,})\b")
_AGY_MODELS_CACHE: dict[str, str] | None = None
_AGY_MODELS_CACHE_TS: float = 0.0
_AGY_MODELS_CACHE_TTL: float = 3600.0

DEFAULT_OPENCLAW_LEASES_PATH = "/containers/ai/openclaw/data/mode_leases.json"
DEFAULT_OPENCLAW_ENV_PATH = "/containers/ai/openclaw/.env"
DEFAULT_SLACK_OWNER_DM = "D0C56ABQY2W"
DEFAULT_OPENCLAW_UI_URL = "https://engineer.wileyriley.com"

TOOL_ICONS: dict[str, str] = {
    "run_command": "🛠️",
    "view_file": "📖",
    "replace_file_content": "✏️",
    "multi_replace_file_content": "✏️",
    "write_to_file": "📝",
    "call_mcp_tool": "🔌",
    "grep_search": "🔍",
    "find_by_name": "📂",
    "search_web": "🌐",
    "read_url_content": "🌐",
    "invoke_subagent": "🤖",
    "manage_task": "⚙️",
}


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

    discovered = discover_agy_models()
    for model_id, display_name in discovered.items():
        if m == model_id.lower() or m == display_name.lower():
            return display_name

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


def is_openclaw_prompt(prompt: str) -> bool:
    """Return True if the prompt originates from an OpenClaw agent turn."""
    if os.environ.get("OPENCLAW_ENFORCE_POSTURE") == "1":
        return True
    return (
        "Runtime: agent=" in prompt
        or "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" in prompt
        or "openclaw:attempt" in prompt
    )


def is_interactive_slack_turn(prompt: str) -> bool:
    """Return True if the prompt is an interactive Slack turn (not a background heartbeat poll)."""
    if os.environ.get("SLACK_ACTIVITY_FORCE") == "1":
        return True
    if "[OpenClaw heartbeat poll]" in prompt or "heartbeat monitor scratch context" in prompt:
        return False
    return "channel=slack" in prompt


def resolve_openclaw_posture(
    prompt: str,
    leases_path: str | None = None,
) -> tuple[str, str, list[str], str]:
    """
    Resolve active OpenClaw execution posture and scope from mode_leases.json.
    Returns (mode, scope, cli_permission_flags, guardrail_instruction).
    """
    path = leases_path or os.environ.get("OPENCLAW_LEASES_FILE", DEFAULT_OPENCLAW_LEASES_PATH)
    mode = "normal"
    scope = "home"
    now = time.time()

    session_match = _SESSION_KEY_RE.search(prompt or "")
    session_key = session_match.group(1) if session_match else None

    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                g = data.get("global") if isinstance(data.get("global"), dict) else {}
                g_until = g.get("until")
                if g_until is not None and now >= float(g_until):
                    mode = str(g.get("revert_to_mode") or "normal")
                    scope = str(g.get("revert_to_scope") or "home")
                else:
                    mode = str(g.get("mode") or "normal")
                    scope = str(g.get("scope") or "home")

                sessions_map = (
                    data.get("sessions") if isinstance(data.get("sessions"), dict) else {}
                )
                if session_key and session_key in sessions_map:
                    s_entry = sessions_map[session_key]
                    if isinstance(s_entry, dict):
                        s_until = s_entry.get("until")
                        if s_until is None or now < float(s_until):
                            mode = str(s_entry.get("mode") or mode)
                            scope = str(s_entry.get("scope") or scope)
        except Exception:
            pass

    mode_lower = mode.lower()
    if mode_lower == "plan":
        cli_flags = ["--mode", "plan"]
        guardrail = (
            f"\n\n[ACTIVE EXECUTION POSTURE: PLAN (READ-ONLY) | SCOPE: {scope.upper()}]\n"
            "You are strictly in PLAN (Read-Only) mode. You may read files, inspect status, "
            "and query read-only MCG tools, but DO NOT modify files or execute mutating commands. "
            "Present a clear plan and ask the user before taking any mutating action."
        )
    elif mode_lower == "accept-edits":
        cli_flags = ["--mode", "accept-edits", "--dangerously-skip-permissions"]
        guardrail = (
            f"\n\n[ACTIVE EXECUTION POSTURE: ACCEPT-EDITS | SCOPE: {scope.upper()}]\n"
            f"File edits, git operations, and test/validation runners within scope '{scope}' "
            "are pre-approved. Still ask the user in Slack before restarting production Docker stacks."
        )
    elif mode_lower == "autopilot":
        cli_flags = ["--mode", "auto", "--dangerously-skip-permissions"]
        guardrail = (
            f"\n\n[ACTIVE EXECUTION POSTURE: AUTOPILOT | SCOPE: {scope.upper()}]\n"
            "Full autonomous execution and remediation are enabled across the active scope."
        )
    else:
        cli_flags = ["--mode", "accept-edits", "--dangerously-skip-permissions"]
        guardrail = (
            f"\n\n[ACTIVE EXECUTION POSTURE: NORMAL (GUARDED) | SCOPE: {scope.upper()}]\n"
            "Safe helper CLIs (openclaw-mode, openclaw-model, openclaw-updates check/canary), "
            "read-only inspection, and Smart Home (ha-mcp) / Media MCG tools are allowed automatically. "
            "BEFORE modifying code/config files or restarting/upgrading Docker containers, "
            "STOP and ask the user for confirmation in Slack."
        )

    return mode_lower, scope.lower(), cli_flags, guardrail


def format_agy_tool_step(step_obj: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """
    Format an agy `step_update` tool object into:
      1. A concise human-readable `reasoning_delta` string for SSE streaming
      2. A structured step dict for the Slack Activity Card
    Returns None if `step_obj` is not a tool step.
    """
    step_type = str(step_obj.get("step_type") or "").lower()
    tool_info = step_obj.get("tool_info") if isinstance(step_obj.get("tool_info"), dict) else {}
    tool_name = str(step_obj.get("tool_name") or tool_info.get("name") or "").strip()

    if step_type != "tool" and not tool_name:
        return None

    tool_name = tool_name or "tool"
    state = str(step_obj.get("state") or "ACTIVE").upper()
    step_index = step_obj.get("step_index", 0)
    duration = step_obj.get("duration_seconds")
    params = tool_info.get("parameters") if isinstance(tool_info.get("parameters"), dict) else {}
    icon = TOOL_ICONS.get(tool_name, "🔧")

    detail = ""
    if tool_name == "run_command":
        cmd_str = str(params.get("CommandLine") or "").strip().replace("\n", " ")
        if len(cmd_str) > 90:
            cmd_str = cmd_str[:87] + "..."
        cwd = str(params.get("Cwd") or "").strip()
        detail = f"`{cmd_str}`" if cmd_str else ""
        if cwd and cwd != "/containers":
            detail += f" (in `{cwd}`)"
    elif tool_name == "view_file":
        path = str(params.get("AbsolutePath") or "").strip()
        start_ln = params.get("StartLine")
        end_ln = params.get("EndLine")
        line_sfx = f"#L{start_ln}-{end_ln}" if start_ln or end_ln else ""
        detail = f"`{path}{line_sfx}`" if path else ""
    elif tool_name in (
        "replace_file_content",
        "multi_replace_file_content",
        "write_to_file",
    ):
        target = str(params.get("TargetFile") or "").strip()
        desc = str(params.get("Description") or params.get("Instruction") or "").strip()
        if len(desc) > 60:
            desc = desc[:57] + "..."
        detail = f"`{target}`" + (f" — {desc}" if desc else "")
    elif tool_name == "call_mcp_tool":
        srv = str(params.get("ServerName") or "").strip()
        tname = str(params.get("ToolName") or "").strip()
        args_obj = params.get("Arguments")
        sub_detail = ""
        if isinstance(args_obj, dict):
            if "name" in args_obj:
                sub_detail = f" -> `{args_obj['name']}`"
            elif "query" in args_obj:
                sub_detail = f" (`{args_obj['query']}`)"
        detail = f"`{srv}.{tname}`{sub_detail}" if srv or tname else ""
    elif tool_name in ("grep_search", "find_by_name"):
        query = str(params.get("Query") or params.get("Pattern") or "").strip()
        spath = str(params.get("SearchPath") or params.get("SearchDirectory") or "").strip()
        detail = f"`{query}`" + (f" in `{spath}`" if spath else "")
    else:
        action = str(params.get("toolAction") or params.get("toolSummary") or "").strip()
        detail = action

    if not detail:
        action_fallback = str(params.get("toolAction") or params.get("toolSummary") or "").strip()
        detail = action_fallback or tool_name

    card_step = {
        "step_index": step_index,
        "tool_name": tool_name,
        "icon": icon,
        "detail": detail,
        "state": state,
        "duration": round(float(duration), 2) if isinstance(duration, (int, float)) else None,
    }

    if state == "ACTIVE":
        reasoning_line = f"{icon} *{tool_name}*: {detail}\n"
    elif state == "DONE":
        dur_str = f" ({float(duration):.1f}s)" if isinstance(duration, (int, float)) else ""
        reasoning_line = f"  ✅ *{tool_name}* done{dur_str}\n"
    elif state in ("ERROR", "FAILED"):
        reasoning_line = f"  ❌ *{tool_name}* failed\n"
    else:
        reasoning_line = f"{icon} *{tool_name}* ({state}): {detail}\n"

    return reasoning_line, card_step


def _resolve_slack_bot_token() -> str | None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if token:
        return token
    env_file = os.environ.get("OPENCLAW_ENV_FILE", DEFAULT_OPENCLAW_ENV_PATH)
    if env_file and os.path.exists(env_file):
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("SLACK_BOT_TOKEN="):
                        val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass
    return None


def _resolve_slack_channel(prompt: str) -> str:
    env_ch = os.environ.get("OPENCLAW_SLACK_ACTIVITY_CHANNEL", "").strip()
    if env_ch:
        return env_ch
    m = _SLACK_CHANNEL_RE.search(prompt or "")
    if m:
        return m.group(1)
    return DEFAULT_SLACK_OWNER_DM


class SlackActivityTracker:
    """
    Posts and live-updates an interactive Block Kit activity card in Slack
    whenever an OpenClaw turn executes tools, complete with:
      - Rolling tool step feed (command lines, files, MCP calls, durations)
      - Active Posture & Model badge
      - [ 🛑 Stop Turn ] button (action_id='abort_agy_turn')
      - [ 🖥️ Open in OpenClaw ] button (https://engineer.wileyriley.com)
    """

    def __init__(
        self,
        run_id: str,
        model: str,
        mode: str,
        scope: str,
        prompt: str,
        enabled: bool = True,
    ):
        self.run_id = run_id
        self.model = model
        self.mode = mode
        self.scope = scope
        self.channel = _resolve_slack_channel(prompt)
        self.token = _resolve_slack_bot_token() if enabled else None
        self.enabled = bool(enabled and self.token and self.channel)
        self.message_ts: str | None = None
        self.steps: dict[Any, dict[str, Any]] = {}
        self.step_order: list[Any] = []
        self.start_time = time.monotonic()
        self.last_update_ts = 0.0

    def _build_blocks(self, status: str = "running") -> tuple[str, list[dict[str, Any]]]:
        elapsed = max(0.1, time.monotonic() - self.start_time)
        total_steps = len(self.step_order)

        if status == "running":
            header_text = (
                f"⚙️ *OpenClaw Agent Activity*  •  Posture: `{self.mode} ({self.scope})`  "
                f"•  Model: `{self.model}`"
            )
        elif status == "aborted":
            header_text = (
                f"🛑 *Turn Aborted by User* ({total_steps} step{'s' if total_steps != 1 else ''} "
                f"in {elapsed:.1f}s)  •  Posture: `{self.mode} ({self.scope})`"
            )
        elif status == "error":
            header_text = (
                f"⚠️ *Agent Turn Ended with Error* ({total_steps} step{'s' if total_steps != 1 else ''} "
                f"in {elapsed:.1f}s)  •  Posture: `{self.mode} ({self.scope})`"
            )
        else:
            header_text = (
                f"✅ *Agent Execution Trace* ({total_steps} step{'s' if total_steps != 1 else ''} "
                f"in {elapsed:.1f}s)  •  Posture: `{self.mode} ({self.scope})`"
            )

        lines: list[str] = []
        recent_keys = self.step_order[-8:]
        if len(self.step_order) > 8:
            lines.append(f"_… {len(self.step_order) - 8} earlier step(s) omitted …_")
        for key in recent_keys:
            st = self.steps[key]
            state = st.get("state", "ACTIVE")
            dur = st.get("duration")
            if state == "DONE":
                badge = "✅"
                sfx = f" _({dur:.1f}s)_" if isinstance(dur, (int, float)) else ""
            elif state in ("ERROR", "FAILED"):
                badge = "❌"
                sfx = ""
            else:
                badge = "⏳"
                sfx = " _(running…)_"
            lines.append(f"{badge} {st['icon']} *{st['tool_name']}*: {st['detail']}{sfx}")

        body_text = "\n".join(lines) if lines else "_Initializing tool execution…_"
        if len(body_text) > 2800:
            body_text = body_text[-2800:]

        blocks: list[dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{header_text}\n\n{body_text}"},
            }
        ]

        action_elements: list[dict[str, Any]] = []
        if status == "running":
            action_elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🛑 Stop Turn", "emoji": True},
                    "style": "danger",
                    "action_id": "abort_agy_turn",
                    "value": self.run_id,
                }
            )
        action_elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "🖥️ Open in OpenClaw", "emoji": True},
                "url": DEFAULT_OPENCLAW_UI_URL,
                "action_id": "open_openclaw_ui",
            }
        )
        blocks.append({"type": "actions", "elements": action_elements})
        return header_text, blocks

    def _slack_api_call_sync(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.token:
            return {"ok": False}
        req = urllib.request.Request(
            f"https://slack.com/api/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as e:
            logger.debug(f"SlackActivityTracker API error ({method}): {e}")
            return {"ok": False}

    async def record_step(self, card_step: dict[str, Any]) -> None:
        key = card_step.get("step_index") or f"{card_step.get('tool_name')}_{len(self.step_order)}"
        if key not in self.steps:
            self.step_order.append(key)
        self.steps[key] = card_step

        if not self.enabled:
            return

        now = time.monotonic()
        if self.message_ts is None:
            fallback_text, blocks = self._build_blocks(status="running")
            res = await asyncio.to_thread(
                self._slack_api_call_sync,
                "chat.postMessage",
                {
                    "channel": self.channel,
                    "text": fallback_text,
                    "blocks": blocks,
                    "unfurl_links": False,
                },
            )
            if res.get("ok") and res.get("ts"):
                self.message_ts = str(res["ts"])
                self.last_update_ts = now
        elif (now - self.last_update_ts) >= 1.0 or card_step.get("state") == "ACTIVE":
            fallback_text, blocks = self._build_blocks(status="running")
            await asyncio.to_thread(
                self._slack_api_call_sync,
                "chat.update",
                {
                    "channel": self.channel,
                    "ts": self.message_ts,
                    "text": fallback_text,
                    "blocks": blocks,
                },
            )
            self.last_update_ts = now

    async def finalize(self, status: str = "done") -> None:
        if not self.enabled or not self.message_ts:
            return
        # Mark any still-ACTIVE steps according to final status
        for st in self.steps.values():
            if st.get("state") == "ACTIVE":
                st["state"] = "DONE" if status == "done" else "ERROR"
        fallback_text, blocks = self._build_blocks(status=status)
        await asyncio.to_thread(
            self._slack_api_call_sync,
            "chat.update",
            {
                "channel": self.channel,
                "ts": self.message_ts,
                "text": fallback_text,
                "blocks": blocks,
            },
        )


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

        if (
            not request.session_id
            and request.system_prompt
            and request.system_prompt not in request.prompt
        ):
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        if is_openclaw_prompt(prompt):
            _, _, perm_flags, guardrail = resolve_openclaw_posture(prompt)
            prompt = f"{prompt}{guardrail}"
        else:
            perm_flags = ["--dangerously-skip-permissions"]

        use_stdin = len(prompt.encode("utf-8")) > 100_000
        cmd = [
            agy_path,
            "--model",
            model,
            *perm_flags,
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

        run_id = request.session_id or f"run_{uuid.uuid4().hex[:10]}"
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE if use_stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            session_store.register_process(run_id, process)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_bytes), timeout=timeout
                )
            finally:
                session_store.unregister_process(run_id)
            duration = time.perf_counter() - start_time
            raw_stdout = stdout.decode("utf-8", errors="replace")

            if session_store.was_cancelled(run_id):
                return TaskResult(
                    success=False,
                    executor=self.name,
                    output="🛑 Turn aborted by user.",
                    session_id=request.session_id,
                    error="Turn aborted by user",
                    duration_seconds=round(duration, 3),
                    metadata={"model": model, "aborted": True},
                )

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
            if process:
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

        mode = "normal"
        scope = "home"
        if is_openclaw_prompt(prompt):
            mode, scope, perm_flags, guardrail = resolve_openclaw_posture(prompt)
            prompt = f"{prompt}{guardrail}"
        else:
            perm_flags = ["--dangerously-skip-permissions"]

        use_stdin = len(prompt.encode("utf-8")) > 100_000
        cmd = [
            agy_path,
            "--model",
            model,
            *perm_flags,
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

        run_id = request.session_id or f"run_{uuid.uuid4().hex[:10]}"
        tracker = SlackActivityTracker(
            run_id=run_id,
            model=model,
            mode=mode,
            scope=scope,
            prompt=prompt,
            enabled=is_interactive_slack_turn(prompt),
        )

        process = None
        session_id = request.session_id
        stream_status = "done"
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdin=asyncio.subprocess.PIPE if use_stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            session_store.register_process(run_id, process)
            if use_stdin and process.stdin is not None:
                payload_line = (
                    json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
                ).encode("utf-8")
                process.stdin.write(payload_line)
                await process.stdin.drain()
                process.stdin.close()

            async def _stream_reader():
                nonlocal session_id, stream_status
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
                                session_store.register_process(session_id, process)

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
                                    session_store.register_process(session_id, process)

                                formatted_tool = format_agy_tool_step(step_obj)
                                if formatted_tool is not None:
                                    reasoning_line, card_step = formatted_tool
                                    await tracker.record_step(card_step)
                                    yield StreamChunk(
                                        delta="",
                                        reasoning_delta=reasoning_line,
                                        session_id=session_id,
                                    )

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
                                    stream_status = "error"
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
                if chunk.is_final:
                    return

            await asyncio.wait_for(process.wait(), timeout=timeout)
            if session_store.was_cancelled(run_id) or (
                session_id and session_store.was_cancelled(session_id)
            ):
                stream_status = "aborted"
                yield StreamChunk(
                    delta="\n\n🛑 *[Turn aborted by user]*",
                    is_final=True,
                    session_id=session_id,
                    finish_reason="stop",
                )
            else:
                yield StreamChunk(
                    delta="",
                    is_final=True,
                    session_id=session_id,
                    finish_reason="stop",
                )

        except TimeoutError:
            stream_status = "error"
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
            stream_status = "error"
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
        finally:
            session_store.unregister_process(run_id)
            if session_id:
                session_store.unregister_process(session_id)
            if session_store.was_cancelled(run_id) or (
                session_id and session_store.was_cancelled(session_id)
            ):
                stream_status = "aborted"
            await tracker.finalize(status=stream_status)

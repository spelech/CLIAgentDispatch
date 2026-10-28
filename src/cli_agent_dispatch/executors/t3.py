import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import time
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import StreamChunk, TaskRequest, TaskResult
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.executors.agy import (
    SlackActivityTracker,
    is_interactive_slack_turn,
    is_openclaw_prompt,
    resolve_agy_effort,
    resolve_openclaw_posture,
)
from cli_agent_dispatch.executors.base import BaseExecutor

logger = logging.getLogger("CLIAgentDispatch.T3Executor")

MCP_PROTOCOL_VERSION = "2025-06-18"
WAIT_SLICE_MS = 1500
TITLE_MAX_CHARS = 80

T3_STATIC_ALIASES: list[str] = [
    "t3",
    "t3-agy",
    "t3-agy-flash",
    "t3-agy-flash-low",
    "t3-agy-flash-medium",
    "t3-agy-flash-high",
    "t3-agy-argon",
    "t3-agy-argon-low",
    "t3-agy-argon-medium",
    "t3-agy-argon-high",
    "t3-agy-opus",
    "t3-agy-sonnet",
    "t3-opencode",
    "t3-opencode-flash",
    "t3-opencode-argon",
    "t3-opencode-opus",
    "t3-opencode-sonnet",
    "t3-opencode-glm",
    "t3-opencode-deepseek-pro",
    "t3-opencode-qwen-max",
]

T3_ITEM_ICONS: dict[str, str] = {
    "command_execution": "🛠️",
    "file_change": "✏️",
    "mcp_tool_call": "🔌",
    "dynamic_tool_call": "🔧",
    "collab_agent_tool_call": "🤖",
    "web_search": "🌐",
    "image_view": "🖼️",
    "plan": "📋",
    "reasoning": "💭",
}

TOOL_ACTIVITY_TYPES: set[str] = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "dynamic_tool_call",
    "collab_agent_tool_call",
    "web_search",
    "image_view",
    "plan",
}


def resolve_t3_binary() -> str | None:
    """Locate the `t3` CLI binary."""
    if settings.t3_cli_path and os.path.exists(settings.t3_cli_path):
        return settings.t3_cli_path
    search_paths = [
        os.environ.get("PATH", ""),
        "/home/steve/.local/bin",
        os.path.expanduser("~/.local/bin"),
    ]
    combined_path = os.pathsep.join(p for p in search_paths if p)
    found = shutil.which("t3", path=combined_path)
    if found:
        return found
    fallback = "/home/steve/.local/bin/t3"
    if os.path.exists(fallback):
        return fallback
    return None


def _normalize_opencode_slug(raw_slug: str) -> str:
    """Normalize a bare model name to an OpenCode provider/model slug."""
    slug = raw_slug.strip()
    if "/" in slug:
        return slug
    lower = slug.lower()
    if "opus" in lower:
        return "google-agy/claude-opus-5-5"
    if "sonnet" in lower or "claude" in lower:
        return "google-agy/claude-sonnet-5-5"
    if "argon" in lower:
        return "google-agy/gemini-4-argon"
    if "gpt-oss" in lower or "oss-120b" in lower:
        return "google-agy/gpt-oss-120b-medium"
    if lower in ("default", "opencode", "opencode-default", "flash", "gemini", "gemini-3.8-flash"):
        return "google-agy/gemini-3.8-flash"
    if lower.startswith("gemini"):
        return "google-agy/gemini-3.8-flash"
    return f"litellm/{slug}"


def _resolve_tier(
    model_lower: str,
    explicit_effort: str | None,
    default_tier: str = "high",
) -> str:
    eff = resolve_agy_effort(model_lower, explicit_effort)
    if eff in ("high", "xhigh", "max"):
        return "high"
    if eff == "low":
        return "low"
    if eff == "medium":
        return "medium"
    if model_lower.endswith("-low") or "-low-" in model_lower:
        return "low"
    if model_lower.endswith("-medium") or "-medium-" in model_lower:
        return "medium"
    if model_lower.endswith("-high") or "-high-" in model_lower:
        return "high"
    return default_tier


def normalize_t3_model_selection(
    model: str | None = None,
    provider: str | None = None,
    effort: str | None = None,
) -> tuple[str, str]:
    """
    Normalize a requested model and optional provider/effort into a T3 Code
    `(instance_id, model_slug)` tuple.

    Supported `instance_id`s:
      - `"antigravity"`: Antigravity ACP (`gemini-3.8-flash-{low,medium,high}`, `gemini-4-argon-{low,medium,high}`)
      - `"opencode"`: OpenCode v2 (`google-agy/*` and `litellm/*`)
    """
    explicit_prov = (provider or "").strip().lower()
    if explicit_prov in ("acp", "agy", "antigravity-acp"):
        explicit_prov = "antigravity"

    raw = (model or "").strip()
    for prefix in ("openai/", "litellm/"):
        if raw.lower().startswith(prefix) and not explicit_prov:
            # Keep `litellm/<model>` intact if it is not a `t3-*` alias
            remainder = raw[len(prefix) :].strip()
            if remainder.lower().startswith("t3"):
                raw = remainder
            break

    if not raw:
        if explicit_prov == "opencode":
            return ("opencode", "google-agy/gemini-3.8-flash")
        tier = _resolve_tier("", effort, default_tier="high")
        return ("antigravity", f"gemini-3.8-flash-{tier}")

    lower = raw.lower()

    # 1. Explicit path syntax: t3/<instance>/<model>
    if lower.startswith("t3/"):
        parts = raw.split("/", 2)
        if len(parts) == 3:
            inst = parts[1].strip().lower()
            sub_model = parts[2].strip()
            if inst in ("opencode",):
                return ("opencode", _normalize_opencode_slug(sub_model))
            return ("antigravity", sub_model or "gemini-3.8-flash-high")

    if lower.startswith("t3-acp/"):
        sub_model = raw.split("/", 1)[1].strip()
        return ("antigravity", sub_model or "gemini-3.8-flash-high")

    if lower.startswith("t3-opencode/"):
        sub_model = raw.split("/", 1)[1].strip()
        return ("opencode", _normalize_opencode_slug(sub_model))

    # 2. Direct OpenCode prefixes: google-agy/* or litellm/*
    if lower.startswith("google-agy/") or lower.startswith("litellm/"):
        return ("opencode", raw)

    # 3. Direct Antigravity ACP model slugs
    if lower in (
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
        "gemini-4-argon-low",
        "gemini-4-argon-medium",
        "gemini-4-argon-high",
    ):
        if explicit_prov == "opencode":
            return ("opencode", _normalize_opencode_slug(lower))
        return ("antigravity", lower)

    # 4. Explicit `t3-opencode*` aliases
    if lower == "t3-opencode" or lower == "t3-opencode-default":
        return ("opencode", "google-agy/gemini-3.8-flash")
    if lower.startswith("t3-opencode-"):
        suffix = raw[len("t3-opencode-") :].strip()
        return ("opencode", _normalize_opencode_slug(suffix))

    # 5. Claude / GPT-OSS aliases -> route via OpenCode's google-agy plugin inside T3
    if "opus" in lower:
        return ("opencode", "google-agy/claude-opus-5-5")
    if "sonnet" in lower or "claude" in lower:
        return ("opencode", "google-agy/claude-sonnet-5-5")
    if "gpt-oss" in lower or "oss-120b" in lower:
        return ("opencode", "google-agy/gpt-oss-120b-medium")

    # 6. Explicit provider="opencode"
    if explicit_prov == "opencode":
        cleaned = raw[3:].lstrip("-_") if lower.startswith("t3") else raw
        return ("opencode", _normalize_opencode_slug(cleaned or "default"))

    # 7. Argon aliases -> Antigravity ACP
    if "argon" in lower:
        tier = _resolve_tier(lower, effort, default_tier="medium")
        return ("antigravity", f"gemini-4-argon-{tier}")

    # 8. Default / Flash / Gemini aliases -> Antigravity ACP
    if lower in (
        "t3",
        "t3-default",
        "t3-agy",
        "t3-flash",
        "t3-agy-flash",
        "t3-gemini",
        "t3-agy-gemini",
    ) or ("flash" in lower or "gemini" in lower):
        tier = _resolve_tier(lower, effort, default_tier="high")
        return ("antigravity", f"gemini-3.8-flash-{tier}")

    # 9. Any other `t3-<model>` -> if not matching Gemini/Argon, route to OpenCode
    if lower.startswith("t3-"):
        suffix = raw[3:].strip()
        return ("opencode", _normalize_opencode_slug(suffix))

    return (settings.t3_default_provider, settings.t3_default_model)


def discover_t3_models() -> list[str]:
    """Return available T3 model aliases and discovered models from T3's provider caches."""
    seen: set[str] = set()
    models: list[str] = []

    def _add(m: str) -> None:
        if m and m not in seen:
            seen.add(m)
            models.append(m)

    for alias in T3_STATIC_ALIASES:
        _add(alias)

    cache_dir = os.path.expanduser("~/.t3/caches")
    for prov_file, prefix in (
        ("antigravity.json", "t3/antigravity/"),
        ("opencode.json", "t3/opencode/"),
    ):
        path = os.path.join(cache_dir, prov_file)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("models", []):
                    slug = entry.get("slug") if isinstance(entry, dict) else None
                    if isinstance(slug, str) and slug:
                        _add(f"{prefix}{slug}")
            except Exception:
                pass

    return models


SRE_REQUEST_HEADER = "=== SRE INCIDENT INVESTIGATION REQUEST ==="
DIAGNOSTIC_ENVELOPE_HEADER = "DIAGNOSTIC ENVELOPE:"
OPENCLAW_SRE_KEYWORD = "OpenClaw SRE"


def is_sre_incident_prompt(prompt: str | None) -> bool:
    """Check if the prompt represents an SRE incident investigation."""
    if not prompt or not isinstance(prompt, str):
        return False
    if SRE_REQUEST_HEADER in prompt:
        return True
    prompt_upper = prompt.upper()
    if SRE_REQUEST_HEADER.upper() in prompt_upper:
        return True
    has_diag = (
        DIAGNOSTIC_ENVELOPE_HEADER in prompt or DIAGNOSTIC_ENVELOPE_HEADER.upper() in prompt_upper
    )
    has_openclaw = OPENCLAW_SRE_KEYWORD in prompt or OPENCLAW_SRE_KEYWORD.upper() in prompt_upper
    return has_diag and has_openclaw


def extract_sre_metadata(prompt: str | None) -> dict[str, str]:
    """
    Extract target (service/container), reason, and exit_code from an SRE prompt.
    Returns dict with keys: 'target', 'reason', 'exit_code'.
    """
    p = prompt or ""
    metadata: dict[str, str] = {
        "target": "unknown",
        "reason": "failure",
        "exit_code": "unknown",
    }

    # Target extraction patterns
    target_match = re.search(
        r"(?:Target Service / Container|Target Service|Target|Container|target)\s*[:=]\s*([^\n\r,`]+)",
        p,
        re.IGNORECASE,
    )
    if target_match:
        val = target_match.group(1).strip().strip("\"'")
        if val:
            metadata["target"] = val

    # Exit code extraction patterns
    exit_match = re.search(
        r"(?:Exit Code / Status|Exit Code|Exit code|exit_code|exitcode|Exit Status)\s*[:=]\s*([^\n\r,`]+)",
        p,
        re.IGNORECASE,
    )
    status_hint = ""
    if exit_match:
        raw_exit = exit_match.group(1).strip().strip("\"'")
        paren_match = re.search(r"(\S+)\s*\(([^)]+)\)", raw_exit)
        if paren_match:
            metadata["exit_code"] = paren_match.group(1).strip()
            status_hint = paren_match.group(2).strip()
        else:
            metadata["exit_code"] = raw_exit

    # Reason extraction patterns
    reason_match = re.search(
        r"(?:Failure Reason|Reason / Failure|Reason|reason|Failure|failure)\s*[:=]\s*([^\n\r,`]+)",
        p,
        re.IGNORECASE,
    )
    if reason_match:
        val = reason_match.group(1).strip().strip("\"'")
        if val:
            metadata["reason"] = val
    elif status_hint:
        metadata["reason"] = status_hint

    return metadata


def build_sre_thread_title(metadata: dict[str, str]) -> str:
    """Format semantic thread title: 🚨 [SRE] <target> (<reason> | exit <exit_code>)"""
    target = metadata.get("target") or "unknown"
    reason = metadata.get("reason") or "failure"
    exit_code = metadata.get("exit_code") or "unknown"

    title = f"🚨 [SRE] {target} ({reason} | exit {exit_code})"
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title


def sanitize_prompt_for_t3(prompt: str | None) -> str:
    """
    Sanitize an inbound SRE prompt for T3 execution:
    - Strips OpenClaw system preamble (`System: <!-- openclaw:attempt:STABLE -->...`)
    - Strips deferred tool schemas catalog (`## Tooling\nTools policy-filtered...`)
    - Strips stacked `[OpenClaw heartbeat poll]` lines
    - Strips internal context blocks (`<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>...<<<END_OPENCLAW_INTERNAL_CONTEXT>>>`)
    Retains only the incident envelope, application logs, and root cause/remediation output contract.
    """
    if not prompt:
        return ""

    cleaned = prompt

    # 1. Strip OpenClaw system preamble
    cleaned = re.sub(
        r"(?:System:\s*)?<!--\s*openclaw:attempt:[^>]*-->.*?(?=(?:\n\s*## Tooling|\n\s*<<<BEGIN|\n\s*===|\n\s*DIAGNOSTIC|\Z))",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # 2. Strip deferred tool schemas catalog
    cleaned = re.sub(
        r"## Tooling\s*\n\s*(?:Tools policy-filtered.*?)?(?=(?:\n\s*<<<BEGIN|\n\s*===|\n\s*DIAGNOSTIC|\Z))",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"## Tooling[\s\S]*?(?=(?:\n\s*<<<BEGIN|\n\s*===|\n\s*DIAGNOSTIC|\Z))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 3. Strip stacked heartbeat poll lines
    cleaned = re.sub(r"^\s*\[OpenClaw heartbeat poll\].*$\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(
        r"^\s*Follow the heartbeat monitor scratch context.*$\n?",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 4. Strip internal context blocks
    cleaned = re.sub(
        r"<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>[\s\S]*?<<<END_OPENCLAW_INTERNAL_CONTEXT>>>\n?",
        "",
        cleaned,
    )

    # 5. Clean up excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _title_from_prompt(prompt: str) -> str:
    """Derive a clean, single-line thread title from an inbound prompt."""
    if is_sre_incident_prompt(prompt):
        return build_sre_thread_title(extract_sre_metadata(prompt))
    for raw_line in (prompt or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("System:", "Runtime:", "<<<")):
            continue
        if line.lower().startswith("user:"):
            line = line[5:].strip()
        if line:
            return (
                line[: TITLE_MAX_CHARS - 1].rstrip() + "…" if len(line) > TITLE_MAX_CHARS else line
            )
    fallback = (prompt or "Agent Task").strip().replace("\n", " ")
    return (
        fallback[: TITLE_MAX_CHARS - 1].rstrip() + "…"
        if len(fallback) > TITLE_MAX_CHARS
        else (fallback or "Agent Task")
    )


def format_t3_activity_item(
    item: dict[str, Any],
    step_index: int,
) -> tuple[str, dict[str, Any]] | None:
    """
    Convert a T3 `t3_thread_read(view='activity')` timeline item into:
      1. A concise `reasoning_delta` string for SSE streaming
      2. A structured `card_step` dict for `SlackActivityTracker`
    Returns None for non-tool items or synthetic workspace preparation items.
    """
    item_type = str(item.get("type") or "").strip()
    if item_type not in TOOL_ACTIVITY_TYPES:
        return None

    title = str(item.get("title") or "").strip()
    item_id = str(item.get("itemId") or "")
    if title == "Workspace ready" or "workspace-preparation" in item_id:
        return None

    raw_status = str(item.get("status") or "inProgress").strip().lower()
    if raw_status in ("completed", "done", "succeeded"):
        state = "DONE"
    elif raw_status in ("failed", "error", "declined"):
        state = "ERROR"
    else:
        state = "ACTIVE"

    text = str(item.get("text") or "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("$ "):
        first_line = first_line[2:].strip()
    if len(first_line) > 90:
        first_line = first_line[:87] + "..."

    tool_name = title or item_type
    detail = f"`{first_line}`" if first_line and first_line != tool_name else (title or item_type)
    icon = T3_ITEM_ICONS.get(item_type, "🔧")

    card_step = {
        "step_index": item_id or step_index,
        "tool_name": tool_name,
        "icon": icon,
        "detail": detail,
        "state": state,
        "duration": None,
    }

    if state == "ACTIVE":
        reasoning_line = f"{icon} *{tool_name}*: {detail}\n"
    elif state == "DONE":
        reasoning_line = f"  ✅ *{tool_name}*: {detail}\n"
    else:
        reasoning_line = f"  ❌ *{tool_name}* failed: {detail}\n"

    return reasoning_line, card_step


class T3McpError(RuntimeError):
    """Raised when a T3 Code MCP operation fails."""


class T3McpClient:
    """Client for T3 Code's Streamable HTTP `/mcp` orchestration server."""

    _token_lock = asyncio.Lock()
    _cached_token: str | None = None
    _cached_expires_at: float = 0.0

    def __init__(
        self,
        server_url: str | None = None,
        cli_path: str | None = None,
        token_cache_path: str | None = None,
    ):
        self.server_url = (server_url or settings.t3_server_url).rstrip("/")
        self.cli_path = cli_path or resolve_t3_binary() or settings.t3_cli_path
        self.token_cache_path = token_cache_path or settings.t3_token_cache_path
        self._mcp_session_id: str | None = None
        self._protocol_version: str = MCP_PROTOCOL_VERSION
        self._req_counter: int = 0
        self._http: httpx.AsyncClient | None = None

    @classmethod
    def _load_disk_token(cls, path: str) -> tuple[str | None, float]:
        if not path or not os.path.exists(path):
            return None, 0.0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            token = data.get("access_token")
            expires_at = float(data.get("expires_at") or 0.0)
            if isinstance(token, str) and token and expires_at > (time.time() + 3600):
                return token, expires_at
        except Exception:
            pass
        return None, 0.0

    @classmethod
    def _save_disk_token(cls, path: str, token: str, expires_at: float) -> None:
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"access_token": token, "expires_at": expires_at}, f, indent=2)
            os.chmod(path, 0o600)
        except Exception as e:
            logger.debug(f"Could not persist T3 MCP token to {path}: {e}")

    async def _mint_new_token(self) -> str:
        """Mint a 30-day full-access MCP Bearer token via `t3 auth pairing create` + `/oauth/mcp/*`."""
        if not self.cli_path or not os.path.exists(self.cli_path):
            raise T3McpError(f"t3 binary not found at {self.cli_path}")

        proc = await asyncio.create_subprocess_exec(
            self.cli_path,
            "auth",
            "pairing",
            "create",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise T3McpError(f"Failed to create T3 pairing credential: {err}")

        pair_data = json.loads(stdout.decode("utf-8", errors="replace"))
        pairing_code = pair_data.get("credential")
        if not pairing_code:
            raise T3McpError("T3 pairing output did not include a credential.")

        redirect_uri = "http://127.0.0.1:8032/callback"
        resource = f"{self.server_url}/mcp"
        verifier = secrets.token_urlsafe(32)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )

        async with httpx.AsyncClient(timeout=10.0) as client:
            reg_resp = await client.post(
                f"{self.server_url}/oauth/mcp/register",
                json={"client_name": "CLIAgentDispatch", "redirect_uris": [redirect_uri]},
            )
            reg_resp.raise_for_status()
            client_id = reg_resp.json()["client_id"]

            dec_resp = await client.post(
                f"{self.server_url}/oauth/mcp/decision",
                json={
                    "authorization": {
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": redirect_uri,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                        "state": "dispatch",
                        "resource": resource,
                    },
                    "decision": {
                        "_tag": "pairing-code",
                        "access": "full-access",
                        "code": pairing_code,
                    },
                },
            )
            dec_resp.raise_for_status()
            redirect_to = dec_resp.json()["redirectTo"]
            parsed_qs = urllib.parse.parse_qs(urllib.parse.urlsplit(redirect_to).query)
            code = (parsed_qs.get("code") or [None])[0]
            if not code:
                raise T3McpError(
                    f"T3 OAuth decision did not return an authorization code: {redirect_to}"
                )

            tok_resp = await client.post(
                f"{self.server_url}/oauth/mcp/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": verifier,
                    "resource": resource,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            tok_resp.raise_for_status()
            tok_json = tok_resp.json()
            access_token = str(tok_json["access_token"])
            expires_in = float(tok_json.get("expires_in") or 2592000)
            expires_at = time.time() + expires_in

            T3McpClient._cached_token = access_token
            T3McpClient._cached_expires_at = expires_at
            self._save_disk_token(self.token_cache_path, access_token, expires_at)
            return access_token

    async def get_token(self, force_refresh: bool = False) -> str:
        async with T3McpClient._token_lock:
            now = time.time()
            if (
                not force_refresh
                and T3McpClient._cached_token
                and T3McpClient._cached_expires_at > (now + 3600)
            ):
                return T3McpClient._cached_token

            if not force_refresh:
                disk_tok, disk_exp = self._load_disk_token(self.token_cache_path)
                if disk_tok:
                    T3McpClient._cached_token = disk_tok
                    T3McpClient._cached_expires_at = disk_exp
                    return disk_tok

            return await self._mint_new_token()

    async def __aenter__(self) -> "T3McpClient":
        self._http = httpx.AsyncClient(timeout=60.0)
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._http is not None:
            if self._mcp_session_id:
                try:
                    token = await self.get_token()
                    await self._http.delete(
                        f"{self.server_url}/mcp",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "mcp-session-id": self._mcp_session_id,
                            "mcp-protocol-version": self._protocol_version,
                        },
                    )
                except Exception:
                    pass
            await self._http.aclose()
            self._http = None

    def _headers(self, token: str) -> dict[str, str]:
        hdrs = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if self._mcp_session_id:
            hdrs["mcp-session-id"] = self._mcp_session_id
        if self._protocol_version:
            hdrs["mcp-protocol-version"] = self._protocol_version
        return hdrs

    async def _initialize_session(self) -> None:
        assert self._http is not None
        token = await self.get_token()
        for attempt in range(2):
            resp = await self._http.post(
                f"{self.server_url}/mcp",
                headers=self._headers(token),
                json={
                    "jsonrpc": "2.0",
                    "id": "init-0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "cli-agent-dispatch", "version": "0.4.0"},
                    },
                },
            )
            if resp.status_code in (401, 403) and attempt == 0:
                token = await self.get_token(force_refresh=True)
                continue
            resp.raise_for_status()
            self._mcp_session_id = (
                resp.headers.get("mcp-session-id")
                or resp.headers.get("Mcp-Session-Id")
                or self._mcp_session_id
            )
            await self._http.post(
                f"{self.server_url}/mcp",
                headers=self._headers(token),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            return
        raise T3McpError("Failed to authenticate with T3 Code /mcp endpoint.")

    @staticmethod
    def _extract_jsonrpc_envelope(resp: httpx.Response) -> dict[str, Any]:
        ctype = resp.headers.get("content-type", "").lower()
        body = resp.text.strip()
        if "text/event-stream" in ctype or body.startswith("event:") or body.startswith("data:"):
            last_data: dict[str, Any] = {}
            for line in body.splitlines():
                if line.startswith("data:"):
                    raw_json = line[5:].strip()
                    if raw_json:
                        try:
                            parsed = json.loads(raw_json)
                            if isinstance(parsed, dict):
                                last_data = parsed
                        except Exception:
                            pass
            return last_data
        return resp.json() if body else {}

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            raise T3McpError("T3McpClient session is not open.")
        self._req_counter += 1
        req_id = f"cad-{self._req_counter}"
        token = await self.get_token()

        resp = await self._http.post(
            f"{self.server_url}/mcp",
            headers=self._headers(token),
            json={
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        if resp.status_code in (401, 403):
            self._mcp_session_id = None
            await self._initialize_session()
            token = await self.get_token()
            resp = await self._http.post(
                f"{self.server_url}/mcp",
                headers=self._headers(token),
                json={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )

        resp.raise_for_status()
        envelope = self._extract_jsonrpc_envelope(resp)
        if "error" in envelope and envelope["error"]:
            raise T3McpError(f"T3 MCP {tool_name} error: {envelope['error']}")
        result = envelope.get("result") or {}
        if result.get("isError"):
            content_list = result.get("content") or []
            text_err = next(
                (c.get("text") for c in content_list if isinstance(c, dict) and c.get("text")),
                "Tool call failed",
            )
            raise T3McpError(f"T3 {tool_name} failed: {text_err}")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        return result

    async def resolve_or_create_project(self, workspace: str) -> str:
        """
        Find the most specific registered T3 project matching `workspace`,
        or auto-create a T3 project for `workspace` if none contains it.
        """
        target = os.path.realpath(os.path.abspath(workspace or settings.default_workspace))
        projects: list[dict[str, Any]] = []
        cursor: int | None = 0
        while cursor is not None:
            page = await self.call_tool("t3_project_list", {"cursor": cursor, "limit": 100})
            page_projects = page.get("projects") or []
            if isinstance(page_projects, list):
                projects.extend(page_projects)
            cursor = page.get("nextCursor")

        best_match: tuple[int, str] | None = None
        for proj in projects:
            proj_id = proj.get("id")
            root = proj.get("workspaceRoot")
            if not proj_id or not root:
                continue
            real_root = os.path.realpath(os.path.abspath(str(root)))
            if target == real_root or target.startswith(real_root.rstrip(os.sep) + os.sep):
                if best_match is None or len(real_root) > best_match[0]:
                    best_match = (len(real_root), str(proj_id))

        if best_match is not None:
            return best_match[1]

        title = os.path.basename(target.rstrip(os.sep)) or "workspace"
        created = await self.call_tool(
            "t3_project_create",
            {
                "title": title,
                "workspaceRoot": target,
                "createWorkspaceRootIfMissing": False,
            },
        )
        proj_obj = created.get("project") if isinstance(created.get("project"), dict) else created
        proj_id = proj_obj.get("id") or proj_obj.get("projectId")
        if not proj_id:
            raise T3McpError(f"Could not resolve or create T3 project for {target}")
        return str(proj_id)

    async def launch_thread(
        self,
        project_id: str,
        title: str,
        message: str,
        instance_id: str,
        model_slug: str,
        runtime_mode: str = "full-access",
        interaction_mode: str = "default",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "t3_thread_launch",
            {
                "projectId": project_id,
                "title": title,
                "message": message,
                "modelSelection": {"instanceId": instance_id, "model": model_slug},
                "runtimeMode": runtime_mode,
                "interactionMode": interaction_mode,
            },
        )

    async def configure_thread(
        self,
        thread_id: str,
        instance_id: str,
        model_slug: str,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "t3_thread_configure",
            {
                "threadId": thread_id,
                "modelSelection": {"instanceId": instance_id, "model": model_slug},
            },
        )

    async def send_message(
        self,
        thread_id: str,
        message: str,
        mode: str = "queue",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "t3_thread_send",
            {
                "threadId": thread_id,
                "message": message,
                "mode": mode,
            },
        )

    async def wait_run(
        self,
        thread_id: str,
        run_id: str,
        timeout_ms: int = WAIT_SLICE_MS,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "t3_thread_wait",
            {
                "threadId": thread_id,
                "runId": run_id,
                "timeoutMs": timeout_ms,
            },
        )

    async def read_thread(
        self,
        thread_id: str,
        view: str = "activity",
        after_position: int | None = None,
        item_id: str | None = None,
        text_offset: int | None = None,
        limit: int = 100,
        run_limit: int = 5,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "threadId": thread_id,
            "view": view,
            "limit": limit,
            "runLimit": run_limit,
        }
        if after_position is not None:
            args["afterPosition"] = after_position
        if item_id is not None:
            args["itemId"] = item_id
        if text_offset is not None:
            args["textOffset"] = text_offset
        return await self.call_tool("t3_thread_read", args)

    async def read_full_item_text(self, thread_id: str, item: dict[str, Any]) -> str:
        text = str(item.get("text") or "")
        next_offset = item.get("nextTextOffset") if item.get("textTruncated") else None
        item_id = item.get("itemId")
        while next_offset is not None and item_id:
            chunk_page = await self.read_thread(
                thread_id=thread_id,
                item_id=str(item_id),
                text_offset=int(next_offset),
                run_limit=1,
            )
            items = chunk_page.get("items") or []
            matched = next(
                (c for c in items if isinstance(c, dict) and c.get("itemId") == item_id),
                None,
            )
            if not matched:
                break
            text += str(matched.get("text") or "")
            next_offset = matched.get("nextTextOffset") if matched.get("textTruncated") else None
        return text

    async def read_run_result_text(
        self,
        thread_id: str,
        run_id: str,
        item_type: str = "assistant_message",
    ) -> str:
        view = "messages" if item_type == "assistant_message" else "activity"
        all_items: list[dict[str, Any]] = []
        after_pos: int | None = None
        while True:
            page = await self.read_thread(
                thread_id=thread_id,
                view=view,
                after_position=after_pos,
                limit=100,
                run_limit=5,
            )
            page_items = page.get("items") or []
            if isinstance(page_items, list):
                all_items.extend(c for c in page_items if isinstance(c, dict))
            if not page.get("hasMore") or page.get("nextPosition") is None:
                break
            after_pos = int(page["nextPosition"])

        matching = [
            it for it in all_items if it.get("runId") == run_id and it.get("type") == item_type
        ]
        if not matching:
            return ""
        texts: list[str] = []
        for it in matching:
            full_t = await self.read_full_item_text(thread_id, it)
            if full_t:
                texts.append(full_t)
        return "\n\n".join(texts)

    async def interrupt_thread(
        self,
        thread_id: str,
        run_id: str | None = None,
        reason: str = "Aborted by user",
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"threadId": thread_id, "reason": reason}
        if run_id:
            args["runId"] = run_id
        return await self.call_tool("t3_thread_interrupt", args)


def _normalize_t3_thread_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    s = session_id.strip()
    if not s:
        return None
    if s.startswith("mcp:"):
        return s
    return None


class T3Executor(BaseExecutor):
    """
    Executor that orchestrates agent sessions through T3 Code (`https://ttt.wileyriley.com`)
    supporting both the `antigravity` ACP provider and `opencode` provider, dynamic mid-thread
    model switching, OpenClaw posture leases, and live Slack Activity Cards.
    """

    def __init__(self, client_factory: type[T3McpClient] = T3McpClient):
        self._client_factory = client_factory

    @property
    def name(self) -> str:
        return "t3"

    def _prepare_prompt_and_posture(self, request: TaskRequest) -> tuple[str, str, str, str]:
        if (
            not request.session_id
            and request.system_prompt
            and request.system_prompt not in request.prompt
        ):
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        if is_sre_incident_prompt(prompt):
            prompt = sanitize_prompt_for_t3(prompt)

        mode = "normal"
        scope = "home"
        interaction_mode = "default"
        if is_openclaw_prompt(prompt):
            mode, scope, _, guardrail = resolve_openclaw_posture(prompt)
            prompt = f"{prompt}{guardrail}"
            if mode == "plan":
                interaction_mode = "plan"

        return prompt, mode, scope, interaction_mode

    async def launch_thread(
        self,
        client: T3McpClient,
        project_id: str,
        title: str,
        message: str,
        instance_id: str,
        model_slug: str,
        runtime_mode: str = "full-access",
        interaction_mode: str = "default",
    ) -> dict[str, Any]:
        """Launch a new T3 Code thread, applying SRE title and prompt sanitization if applicable."""
        if is_sre_incident_prompt(message):
            metadata = extract_sre_metadata(message)
            title = build_sre_thread_title(metadata)
            message = sanitize_prompt_for_t3(message)

        return await client.launch_thread(
            project_id=project_id,
            title=title,
            message=message,
            instance_id=instance_id,
            model_slug=model_slug,
            runtime_mode=runtime_mode,
            interaction_mode=interaction_mode,
        )

    async def _start_or_resume_run(
        self,
        client: T3McpClient,
        request: TaskRequest,
        prompt: str,
        instance_id: str,
        model_slug: str,
        interaction_mode: str,
    ) -> tuple[str, str]:
        workspace = request.workspace or settings.default_workspace
        if workspace and not os.path.exists(workspace):
            workspace = os.getcwd()

        thread_id = _normalize_t3_thread_id(request.session_id)
        if thread_id:
            try:
                snap = await client.read_thread(thread_id, view="messages", limit=1, run_limit=1)
                thread_info = snap.get("thread") if isinstance(snap.get("thread"), dict) else {}
                cur_instance = thread_info.get("providerInstanceId")
                cur_model = thread_info.get("model")
                if (cur_instance and cur_instance != instance_id) or (
                    cur_model and cur_model != model_slug
                ):
                    logger.info(
                        f"Reconfiguring T3 thread {thread_id} from "
                        f"{cur_instance}/{cur_model} -> {instance_id}/{model_slug}"
                    )
                    await client.configure_thread(thread_id, instance_id, model_slug)
                sent = await client.send_message(thread_id, prompt, mode="queue")
                run_id = str(sent.get("runId") or "")
                return thread_id, run_id
            except T3McpError as e:
                logger.warning(f"Could not resume T3 thread {thread_id}, starting new thread: {e}")

        project_id = await client.resolve_or_create_project(workspace)
        if is_sre_incident_prompt(request.prompt) or is_sre_incident_prompt(prompt):
            metadata = extract_sre_metadata(request.prompt)
            if not metadata.get("target") or metadata.get("target") == "unknown":
                metadata = extract_sre_metadata(prompt)
            title = build_sre_thread_title(metadata)
            prompt = sanitize_prompt_for_t3(prompt)
        else:
            title = _title_from_prompt(request.prompt)

        launched = await self.launch_thread(
            client=client,
            project_id=project_id,
            title=title,
            message=prompt,
            instance_id=instance_id,
            model_slug=model_slug,
            runtime_mode="full-access",
            interaction_mode=interaction_mode,
        )
        new_thread_id = str(launched["threadId"])
        new_run_id = str(launched.get("runId") or "")
        return new_thread_id, new_run_id

    async def execute(self, request: TaskRequest) -> TaskResult:
        start_time = time.perf_counter()
        timeout = request.timeout or settings.default_timeout
        instance_id, model_slug = normalize_t3_model_selection(
            request.model, request.provider, request.effort
        )
        model_label = f"{instance_id}/{model_slug}"
        prompt, mode, scope, interaction_mode = self._prepare_prompt_and_posture(request)

        run_key = request.session_id or f"run_{uuid.uuid4().hex[:10]}"
        tracker = SlackActivityTracker(
            run_id=run_key,
            model=model_label,
            mode=mode,
            scope=scope,
            prompt=prompt,
            enabled=is_interactive_slack_turn(prompt),
            t3_ui_url=settings.t3_ui_url,
        )

        thread_id: str | None = _normalize_t3_thread_id(request.session_id)
        run_id: str | None = None
        seen_item_states: dict[str, str] = {}
        step_counter = 0

        try:
            async with self._client_factory() as client:
                thread_id, run_id = await self._start_or_resume_run(
                    client=client,
                    request=request,
                    prompt=prompt,
                    instance_id=instance_id,
                    model_slug=model_slug,
                    interaction_mode=interaction_mode,
                )
                tracker.run_id = thread_id

                async def _cancel_cb() -> bool:
                    try:
                        async with self._client_factory() as cancel_client:
                            await cancel_client.interrupt_thread(thread_id, run_id)
                        return True
                    except Exception:
                        return False

                session_store.register_cancel_callback(run_key, _cancel_cb)
                session_store.register_cancel_callback(thread_id, _cancel_cb)

                deadline = time.monotonic() + timeout
                final_status = "completed"
                while True:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        await client.interrupt_thread(thread_id, run_id, reason="Timed out")
                        await tracker.finalize(status="error")
                        duration = time.perf_counter() - start_time
                        return TaskResult(
                            success=False,
                            executor=self.name,
                            output="",
                            session_id=thread_id,
                            error=f"t3 execution timed out after {timeout} seconds",
                            duration_seconds=round(duration, 3),
                            metadata={
                                "provider": instance_id,
                                "model": model_slug,
                                "thread_id": thread_id,
                                "run_id": run_id,
                            },
                        )

                    slice_ms = min(WAIT_SLICE_MS, max(200, int(remaining_s * 1000)))
                    waited = await client.wait_run(thread_id, run_id, timeout_ms=slice_ms)

                    # Poll activity items for Slack Activity Card updates
                    act_page = await client.read_thread(
                        thread_id, view="activity", limit=100, run_limit=2
                    )
                    for item in act_page.get("items") or []:
                        if not isinstance(item, dict) or item.get("runId") != run_id:
                            continue
                        item_id = str(item.get("itemId") or "")
                        item_status = str(item.get("status") or "")
                        if seen_item_states.get(item_id) == item_status:
                            continue
                        seen_item_states[item_id] = item_status
                        step_counter += 1
                        formatted = format_t3_activity_item(item, step_counter)
                        if formatted is not None:
                            _, card_step = formatted
                            await tracker.record_step(card_step)

                    if session_store.was_cancelled(run_key) or session_store.was_cancelled(
                        thread_id
                    ):
                        await tracker.finalize(status="aborted")
                        duration = time.perf_counter() - start_time
                        return TaskResult(
                            success=False,
                            executor=self.name,
                            output="🛑 Turn aborted by user.",
                            session_id=thread_id,
                            error="Turn aborted by user",
                            duration_seconds=round(duration, 3),
                            metadata={
                                "provider": instance_id,
                                "model": model_slug,
                                "thread_id": thread_id,
                                "aborted": True,
                            },
                        )

                    if not waited.get("timedOut", False):
                        final_status = str(waited.get("status") or "completed")
                        break

                duration = time.perf_counter() - start_time
                if final_status == "completed":
                    reply = await client.read_run_result_text(
                        thread_id, run_id, "assistant_message"
                    )
                    await tracker.finalize(status="done")
                    return TaskResult(
                        success=True,
                        executor=self.name,
                        output=reply,
                        session_id=thread_id,
                        duration_seconds=round(duration, 3),
                        metadata={
                            "provider": instance_id,
                            "model": model_slug,
                            "thread_id": thread_id,
                            "run_id": run_id,
                        },
                    )

                err_text = await client.read_run_result_text(thread_id, run_id, "error")
                await tracker.finalize(
                    status="aborted" if final_status == "interrupted" else "error"
                )
                return TaskResult(
                    success=False,
                    executor=self.name,
                    output="",
                    session_id=thread_id,
                    error=err_text or f"T3 thread {thread_id} ended with status '{final_status}'",
                    duration_seconds=round(duration, 3),
                    metadata={
                        "provider": instance_id,
                        "model": model_slug,
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "status": final_status,
                    },
                )
        except Exception as ex:
            await tracker.finalize(status="error")
            duration = time.perf_counter() - start_time
            return TaskResult(
                success=False,
                executor=self.name,
                output="",
                session_id=thread_id,
                error=f"t3 execution error: {ex}",
                duration_seconds=round(duration, 3),
            )
        finally:
            session_store.unregister_cancel_callback(run_key)
            if thread_id:
                session_store.unregister_cancel_callback(thread_id)

    async def execute_stream(self, request: TaskRequest) -> AsyncGenerator[StreamChunk, None]:
        timeout = request.timeout or settings.default_timeout
        instance_id, model_slug = normalize_t3_model_selection(
            request.model, request.provider, request.effort
        )
        model_label = f"{instance_id}/{model_slug}"
        prompt, mode, scope, interaction_mode = self._prepare_prompt_and_posture(request)

        run_key = request.session_id or f"run_{uuid.uuid4().hex[:10]}"
        tracker = SlackActivityTracker(
            run_id=run_key,
            model=model_label,
            mode=mode,
            scope=scope,
            prompt=prompt,
            enabled=is_interactive_slack_turn(prompt),
            t3_ui_url=settings.t3_ui_url,
        )

        thread_id: str | None = _normalize_t3_thread_id(request.session_id)
        run_id: str | None = None
        seen_item_states: dict[str, str] = {}
        step_counter = 0
        stream_status = "done"

        try:
            async with self._client_factory() as client:
                thread_id, run_id = await self._start_or_resume_run(
                    client=client,
                    request=request,
                    prompt=prompt,
                    instance_id=instance_id,
                    model_slug=model_slug,
                    interaction_mode=interaction_mode,
                )
                tracker.run_id = thread_id

                async def _cancel_cb() -> bool:
                    try:
                        async with self._client_factory() as cancel_client:
                            await cancel_client.interrupt_thread(thread_id, run_id)
                        return True
                    except Exception:
                        return False

                session_store.register_cancel_callback(run_key, _cancel_cb)
                session_store.register_cancel_callback(thread_id, _cancel_cb)

                # Emit initial session_id chunk
                yield StreamChunk(delta="", session_id=thread_id)

                deadline = time.monotonic() + timeout
                final_status = "completed"
                while True:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        stream_status = "error"
                        await client.interrupt_thread(thread_id, run_id, reason="Timed out")
                        yield StreamChunk(
                            delta=f"\n[t3 execution timed out after {timeout}s]",
                            is_final=True,
                            session_id=thread_id,
                            finish_reason="stop",
                        )
                        return

                    slice_ms = min(WAIT_SLICE_MS, max(200, int(remaining_s * 1000)))
                    waited = await client.wait_run(thread_id, run_id, timeout_ms=slice_ms)

                    act_page = await client.read_thread(
                        thread_id, view="activity", limit=100, run_limit=2
                    )
                    for item in act_page.get("items") or []:
                        if not isinstance(item, dict) or item.get("runId") != run_id:
                            continue
                        item_id = str(item.get("itemId") or "")
                        item_status = str(item.get("status") or "")
                        if seen_item_states.get(item_id) == item_status:
                            continue
                        seen_item_states[item_id] = item_status
                        step_counter += 1
                        formatted = format_t3_activity_item(item, step_counter)
                        if formatted is not None:
                            reasoning_line, card_step = formatted
                            await tracker.record_step(card_step)
                            yield StreamChunk(
                                delta="",
                                reasoning_delta=reasoning_line,
                                session_id=thread_id,
                            )

                    if session_store.was_cancelled(run_key) or session_store.was_cancelled(
                        thread_id
                    ):
                        stream_status = "aborted"
                        yield StreamChunk(
                            delta="\n\n🛑 *[Turn aborted by user]*",
                            is_final=True,
                            session_id=thread_id,
                            finish_reason="stop",
                        )
                        return

                    if not waited.get("timedOut", False):
                        final_status = str(waited.get("status") or "completed")
                        break

                if final_status == "completed":
                    reply = await client.read_run_result_text(
                        thread_id, run_id, "assistant_message"
                    )
                    if reply:
                        yield StreamChunk(delta=reply, session_id=thread_id)
                    yield StreamChunk(
                        delta="",
                        is_final=True,
                        session_id=thread_id,
                        finish_reason="stop",
                    )
                elif final_status == "interrupted":
                    stream_status = "aborted"
                    yield StreamChunk(
                        delta="\n\n🛑 *[Turn aborted by user]*",
                        is_final=True,
                        session_id=thread_id,
                        finish_reason="stop",
                    )
                else:
                    stream_status = "error"
                    err_text = await client.read_run_result_text(thread_id, run_id, "error")
                    detail = err_text or f"T3 thread ended with status '{final_status}'"
                    yield StreamChunk(
                        delta=f"[t3 error: {detail}]",
                        is_final=True,
                        session_id=thread_id,
                        finish_reason="error",
                    )
        except Exception as ex:
            stream_status = "error"
            yield StreamChunk(
                delta=f"\n[t3 execution error: {ex}]",
                is_final=True,
                session_id=thread_id,
                finish_reason="error",
            )
        finally:
            session_store.unregister_cancel_callback(run_key)
            if thread_id:
                session_store.unregister_cancel_callback(thread_id)
            if session_store.was_cancelled(run_key) or (
                thread_id and session_store.was_cancelled(thread_id)
            ):
                stream_status = "aborted"
            await tracker.finalize(status=stream_status)

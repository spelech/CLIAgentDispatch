import asyncio
import json
import logging
import os
import shutil

import httpx

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server import MCPServer

from cli_agent_dispatch.api.t3_pair_routes import create_t3_pairing_link, send_t3_pairing_ntfy
from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.models import ExecutorType, InvestigationRequest, TaskRequest
from cli_agent_dispatch.core.session_store import session_store

logger = logging.getLogger("CLIAgentDispatch.MCP")

mcp_server = MCPServer(
    name="cli-agent-dispatch",
    instructions="Universal CLI and Headless AI Agent Delegation Gateway (Antigravity 'agy' and OpenCode)",
    version="0.5.0",
)


@mcp_server.tool(name="delegate_task")
async def delegate_task_tool(
    prompt: str,
    executor: str = "agy",
    workspace: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout: int | None = None,
    session_id: str | None = None,
) -> str:
    """Delegate a coding, analysis, or troubleshooting task to a local AI agent engine.

    Args:
        prompt: Task instructions, prompt, or problem description for the agent.
        executor: Target engine: 'agy' (default, Antigravity CLI) or 'opencode' (OpenCode v2).
        workspace: Absolute path to target repository/directory (defaults to /containers).
        model: Model override (e.g. 'gemini-4-argon', 'gemini-3.8-flash', 'claude-sonnet-5.5', 'gpt-oss-120b', or 'opencode-default').
        effort: Optional reasoning effort ('low', 'medium', 'high', 'xhigh', 'max').
        timeout: Execution timeout in seconds (default: 180).
        session_id: Optional session identifier for continuous multi-turn thread persistence.

    Returns:
        The raw agent execution output, diffs, or error diagnostics.
    """
    exec_lower = executor.lower()
    if exec_lower == "t3":
        exec_type = ExecutorType.T3
    elif exec_lower == "opencode":
        exec_type = ExecutorType.OPENCODE
    else:
        exec_type = ExecutorType.AGY
    req = TaskRequest(
        prompt=prompt,
        executor=exec_type,
        workspace=workspace,
        model=model,
        effort=effort,
        timeout=timeout,
        session_id=session_id,
    )
    result = await engine.dispatch(req)
    if result.success:
        return result.output
    return f"Execution failed via {result.executor}: {result.error}\n\nOutput:\n{result.output}"


@mcp_server.tool(name="investigate_incident")
async def investigate_incident_tool(
    target: str,
    error_logs: str = "",
    exit_code: str | None = None,
    workspace: str | None = None,
    executor: str = "agy",
    max_turns: int = 5,
    model: str | None = None,
    timeout: int | None = None,
    session_id: str | None = None,
) -> str:
    """Run an automated multi-turn SRE diagnostic investigation on a failing service or container.

    Args:
        target: Failing service or container name (e.g. 'caddy', 'homeassistant', 'radarr').
        error_logs: Relevant error logs or stack traces.
        exit_code: Exit code of the failing process/container.
        workspace: Working repository directory (defaults to /containers).
        executor: Target agent engine: 'agy' (default) or 'opencode'.
        max_turns: Maximum diagnostic turns allowed (default: 5).
        model: Model override (e.g. 'gemini-3.8-flash', 'claude-sonnet-5.5').
        timeout: Total investigation timeout in seconds.
        session_id: Optional existing session ID for thread continuity.

    Returns:
        Structured JSON resolution including root cause, proposed fix, and turns used.
    """
    exec_lower = executor.lower()
    if exec_lower == "t3":
        exec_type = ExecutorType.T3
    elif exec_lower == "opencode":
        exec_type = ExecutorType.OPENCODE
    else:
        exec_type = ExecutorType.AGY
    req = InvestigationRequest(
        target=target,
        error_logs=error_logs,
        exit_code=exit_code,
        workspace=workspace,
        executor=exec_type,
        max_turns=max_turns,
        model=model,
        timeout=timeout,
        session_id=session_id,
    )
    result = await engine.investigate(req)
    output_dict = {
        "success": result.success,
        "target": result.target,
        "root_cause": result.root_cause,
        "proposed_fix": result.proposed_fix,
        "category": result.category,
        "turns_used": result.turns_used,
        "duration_seconds": result.duration_seconds,
        "session_id": result.session_id,
        "error": result.error,
        "raw_output": result.raw_output,
    }
    return json.dumps(output_dict, indent=2)


@mcp_server.tool(name="list_active_sessions")
async def list_active_sessions_tool() -> str:
    """List all currently active multi-turn agent sessions and their metadata."""
    sessions = await session_store.list_all()
    records = [
        {
            "client_id": s.client_id,
            "executor_type": s.executor_type.value,
            "native_session_id": s.native_session_id,
            "turn_count": s.turn_count,
            "created_at": s.created_at,
            "last_active_at": s.last_active_at,
        }
        for s in sessions
    ]
    return json.dumps({"active_sessions": records, "count": len(records)}, indent=2)


@mcp_server.tool(name="close_session")
async def close_session_tool(client_id: str) -> str:
    """Close and evict an active multi-turn agent session from the registry.

    Args:
        client_id: The client or conversation identifier of the session to close.
    """
    deleted = await session_store.delete(client_id)
    if deleted:
        return f"Session '{client_id}' successfully closed and evicted."
    return f"Session '{client_id}' not found or already expired."


@mcp_server.tool(name="force_kill_session")
async def force_kill_session_tool(client_id: str) -> str:
    """Forcefully terminate any running background subprocess and evict the session.

    Args:
        client_id: The client or conversation identifier to terminate.
    """
    killed_proc = await session_store.kill_active_process(client_id)
    deleted_session = await session_store.delete(client_id)
    result = {
        "client_id": client_id,
        "active_process_killed": killed_proc,
        "session_evicted": deleted_session,
        "status": "terminated" if (killed_proc or deleted_session) else "not_found",
    }
    return json.dumps(result, indent=2)


@mcp_server.tool(name="list_available_models")
async def list_available_models_tool() -> str:
    """List all available models, executors, and token limits across the gateway."""
    models = [
        {
            "id": "agy",
            "executor": "agy",
            "name": "Antigravity Default (Gemini 3.8 Flash Low)",
            "context_window": 1048576,
            "pricing": "free",
        },
        {
            "id": "gemini-3.8-flash",
            "executor": "agy",
            "name": "Gemini 3.8 Flash (Autonomous Tool Calling)",
            "context_window": 1048576,
            "pricing": "free",
        },
        {
            "id": "claude-sonnet-5.5",
            "executor": "agy",
            "name": "Claude Sonnet 5.5",
            "context_window": 200000,
            "pricing": "free",
        },
        {
            "id": "claude-opus-5.5",
            "executor": "agy",
            "name": "Claude Opus 5.5",
            "context_window": 200000,
            "pricing": "free",
        },
        {
            "id": "gpt-oss-120b",
            "executor": "agy",
            "name": "GPT-OSS 120B",
            "context_window": 131072,
            "pricing": "free",
        },
        {
            "id": "opencode-default",
            "executor": "opencode",
            "name": "OpenCode Primary Model",
            "context_window": 1048576,
            "pricing": "free",
        },
        {
            "id": "opencode-small",
            "executor": "opencode",
            "name": "OpenCode Fast/Small Model",
            "context_window": 262144,
            "pricing": "free",
        },
        {
            "id": "t3-agy-flash",
            "executor": "t3",
            "name": "T3 Code — Antigravity ACP (Gemini 3.8 Flash High)",
            "context_window": 1048576,
            "pricing": "free",
        },
        {
            "id": "t3-agy-argon",
            "executor": "t3",
            "name": "T3 Code — Antigravity ACP (Gemini 4 Argon)",
            "context_window": 1048576,
            "pricing": "free",
        },
        {
            "id": "t3-opencode-opus",
            "executor": "t3",
            "name": "T3 Code — OpenCode (Claude Opus 5.5 via google-agy)",
            "context_window": 200000,
            "pricing": "free",
        },
        {
            "id": "t3-opencode-sonnet",
            "executor": "t3",
            "name": "T3 Code — OpenCode (Claude Sonnet 5.5 via google-agy)",
            "context_window": 200000,
            "pricing": "free",
        },
    ]
    return json.dumps({"models": models, "count": len(models)}, indent=2)


@mcp_server.tool(name="check_executor_health")
async def check_executor_health_tool() -> str:
    """Check the operational health and binary availability of agent executors."""
    # Check agy
    agy_exists = os.path.exists(settings.agy_path)
    agy_exec = os.access(settings.agy_path, os.X_OK) if agy_exists else False

    # Check opencode CLI & background HTTP service
    opencode_exists = os.path.exists(settings.opencode_cli_path)
    opencode_exec = os.access(settings.opencode_cli_path, os.X_OK) if opencode_exists else False

    opencode_server_alive = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.opencode_server_url.rstrip('/')}/api/info")
            opencode_server_alive = resp.status_code in [200, 401, 403]
    except Exception:
        opencode_server_alive = False

    # Check t3 CLI & background HTTP service
    t3_exists = os.path.exists(settings.t3_cli_path)
    t3_exec = os.access(settings.t3_cli_path, os.X_OK) if t3_exists else False
    t3_server_alive = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                f"{settings.t3_server_url.rstrip('/')}/.well-known/oauth-protected-resource"
            )
            t3_server_alive = resp.status_code in [200, 401, 403]
    except Exception:
        t3_server_alive = False

    status = {
        "status": "healthy" if (agy_exec or opencode_exec or t3_exec) else "degraded",
        "executors": {
            "agy": {
                "available": agy_exec,
                "binary_path": settings.agy_path,
                "executable": agy_exec,
            },
            "opencode": {
                "available": opencode_exec,
                "binary_path": settings.opencode_cli_path,
                "executable": opencode_exec,
                "server_url": settings.opencode_server_url,
                "server_alive": opencode_server_alive,
            },
            "t3": {
                "available": t3_exec,
                "binary_path": settings.t3_cli_path,
                "executable": t3_exec,
                "server_url": settings.t3_server_url,
                "ui_url": settings.t3_ui_url,
                "server_alive": t3_server_alive,
            },
        },
    }
    return json.dumps(status, indent=2)


@mcp_server.tool(name="inspect_workspace_diff")
async def inspect_workspace_diff_tool(workspace: str = "/containers") -> str:
    """Inspect git changes (status and diff stat) in a target workspace after agent execution.

    Args:
        workspace: Path to the git workspace directory (defaults to /containers).
    """
    if not os.path.exists(workspace):
        return json.dumps({"error": f"Workspace directory not found: {workspace}"}, indent=2)

    git_bin = shutil.which("git")
    if not git_bin:
        return json.dumps({"error": "git executable not found in PATH"}, indent=2)

    try:
        # Check status
        proc_status = await asyncio.create_subprocess_exec(
            git_bin,
            "-C",
            workspace,
            "status",
            "--short",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_s, stderr_s = await asyncio.wait_for(proc_status.communicate(), timeout=10.0)

        # Check diff stat
        proc_diff = await asyncio.create_subprocess_exec(
            git_bin,
            "-C",
            workspace,
            "diff",
            "--stat",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_d, _ = await asyncio.wait_for(proc_diff.communicate(), timeout=10.0)

        status_text = stdout_s.decode("utf-8", errors="replace").strip()
        diff_text = stdout_d.decode("utf-8", errors="replace").strip()

        modified_files = [line.strip() for line in status_text.splitlines() if line.strip()]
        return json.dumps(
            {
                "workspace": workspace,
                "clean": len(modified_files) == 0,
                "file_count": len(modified_files),
                "modified_files": modified_files,
                "diff_stat": diff_text,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to inspect workspace: {e}"}, indent=2)


@mcp_server.tool(name="generate_t3_pairing_link")
async def generate_t3_pairing_link_tool(
    ttl: str = "15m",
    label: str = "Mobile / Remote Device",
    send_ntfy: bool = True,
) -> str:
    """Generate a T3 Code client pairing link (`https://ttt.wileyriley.com/pair#token=...`) and push it to ntfy.

    Args:
        ttl: Credential expiration TTL (e.g. '15m', '1h', '24h').
        label: Human-readable label for the device being paired.
        send_ntfy: Whether to also push a 1-tap notification to ntfy ('wileyriley-sso').
    """
    pair_data = await create_t3_pairing_link(ttl=ttl, label=label)
    ntfy_sent = False
    if send_ntfy:
        ntfy_sent = await send_t3_pairing_ntfy(pair_data)
    return json.dumps({**pair_data, "ntfy_sent": ntfy_sent}, indent=2)

import json
import logging

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server import MCPServer

from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.models import ExecutorType, InvestigationRequest, TaskRequest
from cli_agent_dispatch.core.session_store import session_store

logger = logging.getLogger("CLIAgentDispatch.MCP")

mcp_server = MCPServer(
    name="cli-agent-dispatch",
    instructions="Universal CLI and Headless AI Agent Delegation Gateway (Antigravity 'agy' and OpenCode)",
    version="0.1.0",
)


@mcp_server.tool(name="delegate_task")
async def delegate_task_tool(
    prompt: str,
    executor: str = "agy",
    workspace: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    session_id: str | None = None,
) -> str:
    """Delegate a coding, analysis, or troubleshooting task to a local AI agent engine.

    Args:
        prompt: Task instructions, prompt, or problem description for the agent.
        executor: Target engine: 'agy' (default, Antigravity CLI) or 'opencode' (OpenCode v2).
        workspace: Absolute path to target repository/directory (defaults to /containers).
        model: Model override (e.g. 'gemini-3.8-flash', 'claude-sonnet-5.5', 'gpt-oss-120b', or 'opencode-default').
        timeout: Execution timeout in seconds (default: 180).
        session_id: Optional session identifier for continuous multi-turn thread persistence.

    Returns:
        The raw agent execution output, diffs, or error diagnostics.
    """
    exec_type = ExecutorType.OPENCODE if executor.lower() == "opencode" else ExecutorType.AGY
    req = TaskRequest(
        prompt=prompt,
        executor=exec_type,
        workspace=workspace,
        model=model,
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
    exec_type = ExecutorType.OPENCODE if executor.lower() == "opencode" else ExecutorType.AGY
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
    """Close and evict an active multi-turn agent session.

    Args:
        client_id: The client or conversation identifier of the session to close.
    """
    deleted = await session_store.delete(client_id)
    if deleted:
        return f"Session '{client_id}' successfully closed and evicted."
    return f"Session '{client_id}' not found or already expired."

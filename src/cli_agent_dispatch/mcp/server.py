import logging

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server import MCPServer
from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest

logger = logging.getLogger("CLIAgentDispatch.MCP")

mcp_server = MCPServer(
    name="cli-agent-dispatch",
    instructions="Universal CLI and Headless AI Agent Delegation Gateway (OpenCode and agy)",
    version="0.1.0",
)


@mcp_server.tool(name="delegate_task")
async def delegate_task_tool(
    prompt: str,
    executor: str = "opencode",
    workspace: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Delegate a coding, analysis, or troubleshooting task to a local AI agent engine.

    Args:
        prompt: Task instructions, prompt, or problem description for the agent.
        executor: Target engine: 'opencode' (default) or 'agy' (Antigravity CLI).
        workspace: Absolute path to target repository/directory (defaults to /containers).
        model: Model override (e.g. 'qwen3.7-flash' or 'gemini-2.5-flash').
        timeout: Execution timeout in seconds (default: 180).

    Returns:
        The raw agent execution output, diffs, or error diagnostics.
    """
    exec_type = ExecutorType.AGY if executor.lower() == "agy" else ExecutorType.OPENCODE
    req = TaskRequest(
        prompt=prompt,
        executor=exec_type,
        workspace=workspace,
        model=model,
        timeout=timeout,
    )
    result = await engine.dispatch(req)
    if result.success:
        return result.output
    return f"Execution failed via {result.executor}: {result.error}\n\nOutput:\n{result.output}"

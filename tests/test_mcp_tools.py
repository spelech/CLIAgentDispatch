import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_dispatch.core.models import ExecutorType, InvestigationResult, TaskResult
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.mcp.server import (
    check_executor_health_tool,
    close_session_tool,
    delegate_task_tool,
    force_kill_session_tool,
    inspect_workspace_diff_tool,
    investigate_incident_tool,
    list_active_sessions_tool,
    list_available_models_tool,
)


@pytest.mark.asyncio
async def test_delegate_task_tool_success():
    mock_res = TaskResult(
        success=True,
        executor="opencode",
        output="Refactored 3 files successfully",
        duration_seconds=1.2,
    )
    with patch(
        "cli_agent_dispatch.mcp.server.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_res
        output = await delegate_task_tool(
            prompt="Refactor auth",
            executor="opencode",
            workspace="/containers",
            session_id="session-123",
        )
        assert "Refactored 3 files successfully" in output
        req = mock_dispatch.call_args[0][0]
        assert req.session_id == "session-123"


@pytest.mark.asyncio
async def test_delegate_task_tool_failure():
    mock_res = TaskResult(
        success=False,
        executor="agy",
        output="Partial output before crash",
        error="Permission denied",
        duration_seconds=0.4,
    )
    with patch(
        "cli_agent_dispatch.mcp.server.engine.dispatch", new_callable=AsyncMock
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_res
        output = await delegate_task_tool(
            prompt="Delete root",
            executor="agy",
        )
        assert "Execution failed via agy" in output
        assert "Permission denied" in output
        assert "Partial output before crash" in output


@pytest.mark.asyncio
async def test_investigate_incident_tool():
    mock_res = InvestigationResult(
        target="caddy",
        success=True,
        root_cause="Stale lock file in data dir",
        proposed_fix="rm -f /data/lock && restart",
        category="filesystem",
        turns_used=2,
        duration_seconds=3.5,
        raw_output="Root cause determined",
    )
    with patch(
        "cli_agent_dispatch.mcp.server.engine.investigate", new_callable=AsyncMock
    ) as mock_investigate:
        mock_investigate.return_value = mock_res
        output = await investigate_incident_tool(
            target="caddy",
            error_logs="bind: address already in use",
            exit_code="1",
        )
        data = json.loads(output)
        assert data["success"] is True
        assert data["target"] == "caddy"
        assert "Stale lock file" in data["root_cause"]
        assert data["category"] == "filesystem"
        assert data["turns_used"] == 2


@pytest.mark.asyncio
async def test_session_management_tools():
    # Setup test session
    client_id = "test-mcp-session-1"
    await session_store.get_or_create(client_id=client_id, executor_type=ExecutorType.AGY)

    # Test list_active_sessions
    list_output = await list_active_sessions_tool()
    data = json.loads(list_output)
    assert data["count"] >= 1
    session_ids = [s["client_id"] for s in data["active_sessions"]]
    assert client_id in session_ids

    # Test close_session
    close_output = await close_session_tool(client_id)
    assert f"Session '{client_id}' successfully closed" in close_output

    # Test close_session when not found
    close_again = await close_session_tool(client_id)
    assert "not found" in close_again


@pytest.mark.asyncio
async def test_force_kill_session_tool():
    client_id = "test-mcp-session-kill"
    await session_store.get_or_create(client_id=client_id, executor_type=ExecutorType.AGY)

    # Mock active process
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    session_store.register_process(client_id, mock_proc)

    # Call force kill tool
    kill_output = await force_kill_session_tool(client_id)
    data = json.loads(kill_output)
    assert data["client_id"] == client_id
    assert data["active_process_killed"] is True
    assert data["session_evicted"] is True
    assert data["status"] == "terminated"

    # Verify process terminated
    mock_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_list_available_models_tool():
    output = await list_available_models_tool()
    data = json.loads(output)
    assert "models" in data
    assert data["count"] >= 5
    model_ids = [m["id"] for m in data["models"]]
    assert "agy" in model_ids
    assert "gemini-3.8-flash" in model_ids
    assert "claude-sonnet-5.5" in model_ids
    assert "opencode-default" in model_ids


@pytest.mark.asyncio
async def test_check_executor_health_tool():
    output = await check_executor_health_tool()
    data = json.loads(output)
    assert "status" in data
    assert "executors" in data
    assert "agy" in data["executors"]
    assert "opencode" in data["executors"]


@pytest.mark.asyncio
async def test_inspect_workspace_diff_tool(tmp_path):
    # Non-existent directory
    res_err = await inspect_workspace_diff_tool("/non/existent/path/xyz")
    assert "Workspace directory not found" in res_err

    # Valid directory
    res = await inspect_workspace_diff_tool(str(tmp_path))
    data = json.loads(res)
    assert data["workspace"] == str(tmp_path)
    assert "clean" in data

from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import TaskResult
from cli_agent_dispatch.mcp.server import delegate_task_tool


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
        )
        assert "Refactored 3 files successfully" in output


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

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from cli_agent_dispatch.cli import app
from cli_agent_dispatch.core.models import TaskResult

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Universal CLI & Headless AI Agent Delegation Engine" in result.output


def test_cli_run_success():
    mock_result = TaskResult(
        success=True,
        executor="opencode",
        output="Result from agent",
        duration_seconds=1.2,
    )
    with patch("cli_agent_dispatch.cli.delegate_task", new_callable=AsyncMock) as mock_delegate:
        mock_delegate.return_value = mock_result
        res = runner.invoke(app, ["run", "test prompt", "--executor", "opencode"])
        assert res.exit_code == 0
        assert "Dispatching task to opencode..." in res.output
        assert "Result from agent" in res.output
        mock_delegate.assert_called_once()


def test_cli_run_failure():
    mock_result = TaskResult(
        success=False,
        executor="agy",
        output="Partial output",
        error="Execution timed out",
        duration_seconds=5.0,
    )
    with patch("cli_agent_dispatch.cli.delegate_task", new_callable=AsyncMock) as mock_delegate:
        mock_delegate.return_value = mock_result
        res = runner.invoke(app, ["run", "failing prompt", "--executor", "agy"])
        assert res.exit_code == 1
        assert "Execution Failed: Execution timed out" in res.output
        assert "Partial output" in res.output

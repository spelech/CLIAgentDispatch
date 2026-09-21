from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_dispatch.core.engine import DispatchEngine
from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest
from cli_agent_dispatch.executors.agy import AgyExecutor
from cli_agent_dispatch.executors.opencode import OpenCodeExecutor


@pytest.mark.asyncio
async def test_opencode_http_success():
    executor = OpenCodeExecutor()
    req = TaskRequest(prompt="test prompt", executor=ExecutorType.OPENCODE)

    mock_session_resp = MagicMock()
    mock_session_resp.json.return_value = {"id": "ses-123"}
    mock_session_resp.raise_for_status = MagicMock()

    mock_msg_resp = MagicMock()
    mock_msg_resp.json.return_value = {"parts": [{"type": "text", "text": "Hello from OpenCode!"}]}
    mock_msg_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", side_effect=[mock_session_resp, mock_msg_resp]):
        result = await executor.execute(req)

    assert result.success is True
    assert result.output == "Hello from OpenCode!"
    assert result.session_id == "ses-123"
    assert result.executor == "opencode"


@pytest.mark.asyncio
async def test_opencode_http_fail_cli_fallback_success(tmp_path):
    executor = OpenCodeExecutor()
    req = TaskRequest(prompt="fix bugs", executor=ExecutorType.OPENCODE, workspace=str(tmp_path))

    fake_cli = tmp_path / "opencode"
    fake_cli.write_text("#!/bin/sh\necho 'CLI output'")
    fake_cli.chmod(0o755)

    with (
        patch("cli_agent_dispatch.executors.opencode.settings.opencode_cli_path", str(fake_cli)),
        patch("httpx.AsyncClient.post", side_effect=Exception("HTTP connection refused")),
    ):
        result = await executor.execute(req)

    assert result.success is True
    assert "CLI output" in result.output
    assert result.metadata.get("mode") == "cli_fallback"


@pytest.mark.asyncio
async def test_opencode_cli_binary_not_found():
    executor = OpenCodeExecutor()
    req = TaskRequest(prompt="fix bugs", executor=ExecutorType.OPENCODE)

    with (
        patch(
            "cli_agent_dispatch.executors.opencode.settings.opencode_cli_path",
            "/nonexistent/path/opencode",
        ),
        patch("httpx.AsyncClient.post", side_effect=Exception("HTTP failed")),
    ):
        result = await executor.execute(req)

    assert result.success is False
    assert "CLI binary not found" in result.error


@pytest.mark.asyncio
async def test_opencode_cli_timeout(tmp_path):
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="slow task",
        executor=ExecutorType.OPENCODE,
        timeout=1,
        workspace=str(tmp_path),
    )

    fake_cli = tmp_path / "opencode"
    fake_cli.write_text("#!/bin/sh\nsleep 10")
    fake_cli.chmod(0o755)

    with (
        patch("cli_agent_dispatch.executors.opencode.settings.opencode_cli_path", str(fake_cli)),
        patch("httpx.AsyncClient.post", side_effect=Exception("HTTP failed")),
    ):
        result = await executor.execute(req)

    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_agy_success(tmp_path):
    executor = AgyExecutor()
    req = TaskRequest(prompt="inspect repo", executor=ExecutorType.AGY, workspace=str(tmp_path))

    fake_agy = tmp_path / "agy"
    fake_agy.write_text("#!/bin/sh\necho 'Agy report'")
    fake_agy.chmod(0o755)

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", str(fake_agy)):
        result = await executor.execute(req)

    assert result.success is True
    assert "Agy report" in result.output
    assert result.executor == "agy"


@pytest.mark.asyncio
async def test_agy_binary_not_found():
    executor = AgyExecutor()
    req = TaskRequest(prompt="test", executor=ExecutorType.AGY)

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", "/nonexistent/agy"):
        result = await executor.execute(req)

    assert result.success is False
    assert "binary not found" in result.error


@pytest.mark.asyncio
async def test_agy_execution_failure(tmp_path):
    executor = AgyExecutor()
    req = TaskRequest(prompt="fail", executor=ExecutorType.AGY, workspace=str(tmp_path))

    fake_agy = tmp_path / "agy"
    fake_agy.write_text("#!/bin/sh\necho 'fatal error' >&2\nexit 2")
    fake_agy.chmod(0o755)

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", str(fake_agy)):
        result = await executor.execute(req)

    assert result.success is False
    assert "exited with code 2" in result.error


@pytest.mark.asyncio
async def test_engine_dispatch_and_convenience_function():
    engine = DispatchEngine()
    mock_executor = MagicMock()
    mock_executor.name = "custom"
    mock_executor.execute = AsyncMock(
        return_value=MagicMock(success=True, duration_seconds=0.1, output="done")
    )

    engine.register_executor(mock_executor)
    res = await engine.dispatch(TaskRequest(prompt="test", executor=ExecutorType.OPENCODE))
    assert res is not None

    with pytest.raises(ExecutorNotFoundError):
        engine.get_executor("nonexistent")

import json
from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import TaskRequest
from cli_agent_dispatch.executors.agy import AgyExecutor


@pytest.mark.asyncio
async def test_agy_uses_conversation_flag_when_session_id_provided():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Second turn message",
        session_id="conv_xyz_123",
        executor="agy",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    json_output = json.dumps({
        "conversation_id": "conv_xyz_123",
        "status": "SUCCESS",
        "response": "Agy turn 2 reply",
        "num_turns": 2,
    }).encode("utf-8")
    mock_process.communicate.return_value = (json_output, b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await executor.execute(req)

    # Must include --conversation conv_xyz_123
    # Must include --output-format json
    args = mock_exec.call_args[0]
    assert "--conversation" in args
    assert "conv_xyz_123" in args
    assert "--output-format" in args
    assert "json" in args
    assert result.session_id == "conv_xyz_123"
    assert result.output == "Agy turn 2 reply"
    assert result.success is True
    assert result.metadata.get("num_turns") == 2


@pytest.mark.asyncio
async def test_agy_turn_1_creates_session_and_parses_json():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="First turn message",
        session_id=None,
        executor="agy",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    json_output = json.dumps({
        "conversation_id": "conv_new_456",
        "status": "SUCCESS",
        "response": "First turn reply",
        "num_turns": 1,
    }).encode("utf-8")
    mock_process.communicate.return_value = (json_output, b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await executor.execute(req)

    args = mock_exec.call_args[0]
    assert "--conversation" not in args
    assert "--output-format" in args
    assert "json" in args
    assert result.session_id == "conv_new_456"
    assert result.output == "First turn reply"
    assert result.success is True


@pytest.mark.asyncio
async def test_agy_prepends_system_prompt_on_turn_1():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Analyze logs",
        session_id=None,
        system_prompt="You are an SRE agent.",
        executor="agy",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    json_output = json.dumps({
        "conversation_id": "conv_sre_789",
        "response": "Starting investigation",
    }).encode("utf-8")
    mock_process.communicate.return_value = (json_output, b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await executor.execute(req)

    args = mock_exec.call_args[0]
    assert "You are an SRE agent.\n\nAnalyze logs" in args
    assert result.session_id == "conv_sre_789"


@pytest.mark.asyncio
async def test_agy_does_not_reinject_system_prompt_on_subsequent_turns():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Next diagnostic step",
        session_id="conv_existing_111",
        system_prompt="You are an SRE agent.",
        executor="agy",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    json_output = json.dumps({
        "conversation_id": "conv_existing_111",
        "response": "Step output",
    }).encode("utf-8")
    mock_process.communicate.return_value = (json_output, b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await executor.execute(req)

    args = mock_exec.call_args[0]
    assert "--conversation" in args
    assert "conv_existing_111" in args
    assert "Next diagnostic step" in args
    assert "You are an SRE agent.\n\nNext diagnostic step" not in args


@pytest.mark.asyncio
async def test_agy_falls_back_gracefully_on_plain_text_output():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Legacy plain text command",
        session_id="conv_fallback_222",
        executor="agy",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    plain_output = b"Plain text output that is not valid JSON\n"
    mock_process.communicate.return_value = (plain_output, b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await executor.execute(req)

    assert result.success is True
    assert result.output == "Plain text output that is not valid JSON\n"
    assert result.session_id == "conv_fallback_222"


@pytest.mark.asyncio
async def test_agy_timeout():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Hanging task",
        session_id="conv_timeout_1",
        executor="agy",
        timeout=1,
    )

    with patch("asyncio.create_subprocess_exec", side_effect=TimeoutError()):
        result = await executor.execute(req)

    assert result.success is False
    assert "timed out" in result.error
    assert result.session_id == "conv_timeout_1"


@pytest.mark.asyncio
async def test_agy_general_exception():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Crash task",
        session_id="conv_err_1",
        executor="agy",
    )

    with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("Subprocess failed to spawn")):
        result = await executor.execute(req)

    assert result.success is False
    assert "Subprocess failed to spawn" in result.error
    assert result.session_id == "conv_err_1"


@pytest.mark.asyncio
async def test_agy_nonexistent_workspace_fallback():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Test prompt",
        executor="agy",
        workspace="/path/that/does/not/exist/for/sure",
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b'{"conversation_id": "c1", "response": "ok"}', b"")

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await executor.execute(req)

    assert result.success is True
    assert mock_exec.call_args[1]["cwd"] != "/path/that/does/not/exist/for/sure"


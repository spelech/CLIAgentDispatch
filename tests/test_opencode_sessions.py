from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.models import TaskRequest
from cli_agent_dispatch.executors.opencode import OpenCodeExecutor


@pytest.mark.asyncio
async def test_opencode_reuses_existing_session_id():
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="Second turn message",
        session_id="existing_session_999",
        executor="opencode",
    )

    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"parts": [{"type": "text", "text": "Turn 2 reply"}]}
    mock_client.post.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await executor.execute(req)

    # Must NOT call POST /session
    # Must call POST /session/existing_session_999/message
    assert result.success is True
    assert result.session_id == "existing_session_999"
    assert result.output == "Turn 2 reply"
    assert mock_client.post.call_count == 1
    assert "existing_session_999/message" in mock_client.post.call_args[0][0]


@pytest.mark.asyncio
async def test_opencode_creates_new_session_when_session_id_none():
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="First turn message",
        session_id=None,
        executor="opencode",
    )

    mock_client = AsyncMock()
    mock_session_resp = AsyncMock()
    mock_session_resp.status_code = 200
    mock_session_resp.json.return_value = {"id": "ses_new_123"}

    mock_msg_resp = AsyncMock()
    mock_msg_resp.status_code = 200
    mock_msg_resp.json.return_value = {"parts": [{"type": "text", "text": "New session reply"}]}

    mock_client.post.side_effect = [mock_session_resp, mock_msg_resp]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await executor.execute(req)

    assert result.success is True
    assert result.session_id == "ses_new_123"
    assert result.output == "New session reply"
    assert mock_client.post.call_count == 2
    assert mock_client.post.call_args_list[0][0][0].endswith("/session")
    assert "ses_new_123/message" in mock_client.post.call_args_list[1][0][0]


@pytest.mark.asyncio
async def test_opencode_injects_system_prompt_on_new_session():
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="Investigate outage",
        session_id=None,
        system_prompt="You are an SRE agent.",
        executor="opencode",
    )

    mock_client = AsyncMock()
    mock_session_resp = AsyncMock()
    mock_session_resp.status_code = 200
    mock_session_resp.json.return_value = {"id": "ses_sre_456"}

    mock_msg_resp = AsyncMock()
    mock_msg_resp.status_code = 200
    mock_msg_resp.json.return_value = {"parts": [{"type": "text", "text": "Investigation started"}]}

    mock_client.post.side_effect = [mock_session_resp, mock_msg_resp]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await executor.execute(req)

    assert result.success is True
    assert result.session_id == "ses_sre_456"
    assert mock_client.post.call_count == 2
    msg_call_kwargs = mock_client.post.call_args_list[1][1]
    sent_text = msg_call_kwargs["json"]["parts"][0]["text"]
    assert "You are an SRE agent." in sent_text
    assert "Investigate outage" in sent_text


@pytest.mark.asyncio
async def test_opencode_does_not_reinject_system_prompt_on_existing_session():
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="Next step",
        session_id="existing_ses_789",
        system_prompt="You are an SRE agent.",
        executor="opencode",
    )

    mock_client = AsyncMock()
    mock_msg_resp = AsyncMock()
    mock_msg_resp.status_code = 200
    mock_msg_resp.json.return_value = {"parts": [{"type": "text", "text": "Next step reply"}]}

    mock_client.post.return_value = mock_msg_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await executor.execute(req)

    assert result.success is True
    assert result.session_id == "existing_ses_789"
    assert mock_client.post.call_count == 1
    msg_call_kwargs = mock_client.post.call_args[1]
    sent_text = msg_call_kwargs["json"]["parts"][0]["text"]
    # Existing session must reuse session without prepending system prompt again
    assert sent_text == "Next step"


@pytest.mark.asyncio
async def test_opencode_session_http_fail_cli_fallback(tmp_path):
    executor = OpenCodeExecutor()
    fake_cli = tmp_path / "opencode"
    fake_cli.write_text("#!/bin/sh\necho 'CLI fallback reply'")
    fake_cli.chmod(0o755)

    req = TaskRequest(
        prompt="Execute command",
        session_id="ses_fallback_1",
        executor="opencode",
        workspace=str(tmp_path),
    )

    with (
        patch("cli_agent_dispatch.executors.opencode.settings.opencode_cli_path", str(fake_cli)),
        patch("httpx.AsyncClient.post", side_effect=Exception("HTTP error")),
    ):
        result = await executor.execute(req)

    assert result.success is True
    assert "CLI fallback reply" in result.output
    assert result.metadata.get("mode") == "cli_fallback"

from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationResult,
    TaskResult,
)


@pytest.mark.asyncio
async def test_investigate_endpoint_success(async_client):
    mock_inv_result = InvestigationResult(
        success=True,
        session_id="ses_inv_123",
        target="matter-hub",
        executor="opencode",
        root_cause="Port conflict on 8123",
        proposed_fix="docker compose restart matter-hub",
        category="port_conflict",
        turns_used=2,
        duration_seconds=1.5,
        transcript=[
            {"role": "user", "content": "Investigate matter-hub"},
            {"role": "assistant", "content": "Found port conflict"},
        ],
        raw_output="Found port conflict",
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.investigate",
        new_callable=AsyncMock,
    ) as mock_inv:
        mock_inv.return_value = mock_inv_result
        payload = {
            "target": "matter-hub",
            "exit_code": 1,
            "error_logs": "bind: address already in use",
            "max_turns": 3,
            "executor": "opencode",
        }
        resp = await async_client.post("/v1/investigate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["target"] == "matter-hub"
        assert data["session_id"] == "ses_inv_123"
        assert data["root_cause"] == "Port conflict on 8123"
        assert data["proposed_fix"] == "docker compose restart matter-hub"
        assert data["category"] == "port_conflict"
        assert data["turns_used"] == 2
        mock_inv.assert_awaited_once()


@pytest.mark.asyncio
async def test_investigate_endpoint_validation_error(async_client):
    # Missing required 'target' field
    resp = await async_client.post("/v1/investigate", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_investigate_endpoint_error_handling(async_client):
    with patch(
        "cli_agent_dispatch.api.session_routes.engine.investigate",
        new_callable=AsyncMock,
    ) as mock_inv:
        mock_inv.side_effect = RuntimeError("Investigate engine failed")
        resp = await async_client.post("/v1/investigate", json={"target": "radarr4k"})
        assert resp.status_code == 500
        assert "Investigate engine failed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_session_endpoint(async_client):
    mock_task_result = TaskResult(
        success=True,
        executor="opencode",
        output="Session initialized successfully",
        session_id="sess_123",
        duration_seconds=0.4,
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        payload = {
            "executor": "opencode",
            "prompt": "Initialize debug session",
            "system_prompt": "You are an SRE debugger",
        }
        resp = await async_client.post("/v1/sessions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_123"
        assert data["executor"] == "opencode"
        assert data["status"] == "ready"
        assert data["output"] == "Session initialized successfully"
        mock_dispatch.assert_awaited_once()
        call_req = mock_dispatch.call_args[0][0]
        assert call_req.prompt == "Initialize debug session"
        assert call_req.system_prompt == "You are an SRE debugger"


@pytest.mark.asyncio
async def test_create_session_default_prompt(async_client):
    mock_task_result = TaskResult(
        success=True,
        executor="agy",
        output="Ready",
        session_id="conv_456",
        duration_seconds=0.2,
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        resp = await async_client.post("/v1/sessions", json={"executor": "agy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "conv_456"
        assert data["status"] == "ready"
        call_req = mock_dispatch.call_args[0][0]
        assert call_req.prompt == "Initialize session"
        assert call_req.executor == ExecutorType.AGY


@pytest.mark.asyncio
async def test_create_session_dispatch_failure(async_client):
    mock_task_result = TaskResult(
        success=False,
        executor="opencode",
        output="",
        error="Container daemon offline",
        duration_seconds=0.1,
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        resp = await async_client.post("/v1/sessions", json={"executor": "opencode"})
        assert resp.status_code == 500
        assert "Container daemon offline" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_session_exception_handling(async_client):
    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.side_effect = RuntimeError("Fatal engine crash")
        resp = await async_client.post("/v1/sessions", json={})
        assert resp.status_code == 500
        assert "Fatal engine crash" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_send_session_message_endpoint(async_client):
    mock_task_result = TaskResult(
        success=True,
        executor="opencode",
        output="Turn 2 reply: inspected logs",
        session_id="sess_123",
        duration_seconds=0.8,
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        payload = {
            "prompt": "Check container logs",
            "executor": "opencode",
        }
        resp = await async_client.post("/v1/sessions/sess_123/message", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["session_id"] == "sess_123"
        assert data["output"] == "Turn 2 reply: inspected logs"
        call_req = mock_dispatch.call_args[0][0]
        assert call_req.session_id == "sess_123"
        assert call_req.prompt == "Check container logs"
        assert call_req.executor == ExecutorType.OPENCODE


@pytest.mark.asyncio
async def test_send_session_message_validation_error(async_client):
    # Both prompt and message missing
    resp = await async_client.post("/v1/sessions/sess_123/message", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_send_session_message_with_alternative_field(async_client):
    mock_task_result = TaskResult(
        success=True,
        executor="opencode",
        output="Got message",
        session_id="sess_123",
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        resp = await async_client.post(
            "/v1/sessions/sess_123/message", json={"message": "Check status"}
        )
        assert resp.status_code == 200
        call_req = mock_dispatch.call_args[0][0]
        assert call_req.prompt == "Check status"


@pytest.mark.asyncio
async def test_send_session_message_infer_executor_agy(async_client):
    mock_task_result = TaskResult(
        success=True,
        executor="agy",
        output="Agy response",
        session_id="conv_xyz_789",
    )

    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.return_value = mock_task_result
        resp = await async_client.post(
            "/v1/sessions/conv_xyz_789/message", json={"prompt": "Next step"}
        )
        assert resp.status_code == 200
        call_req = mock_dispatch.call_args[0][0]
        assert call_req.executor == ExecutorType.AGY


@pytest.mark.asyncio
async def test_send_session_message_not_found_executor(async_client):
    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.side_effect = ExecutorNotFoundError("Executor not found")
        resp = await async_client.post("/v1/sessions/sess_123/message", json={"prompt": "Hello"})
        assert resp.status_code == 404
        assert "Executor not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_send_session_message_exception_handling(async_client):
    with patch(
        "cli_agent_dispatch.api.session_routes.engine.dispatch",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        mock_dispatch.side_effect = RuntimeError("Unexpected dispatch failure")
        resp = await async_client.post("/v1/sessions/sess_123/message", json={"prompt": "Hello"})
        assert resp.status_code == 500
        assert "Unexpected dispatch failure" in resp.json()["detail"]

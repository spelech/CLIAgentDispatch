import pytest
from pydantic import ValidationError

from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationRequest,
    InvestigationResult,
    SessionCreateRequest,
    SessionMessageRequest,
    TaskRequest,
)


def test_task_request_accepts_session_id():
    req = TaskRequest(prompt="Inspect logs", session_id="ses_12345", system_prompt="Be an SRE")
    assert req.session_id == "ses_12345"
    assert req.system_prompt == "Be an SRE"

    # Verify defaults
    req_default = TaskRequest(prompt="Run simple task")
    assert req_default.session_id is None
    assert req_default.system_prompt is None


def test_investigation_request_and_result():
    inv_req = InvestigationRequest(
        target="radarr4k",
        exit_code=1,
        error_logs="FATAL database locked",
        workspace="/containers/media_content",
    )
    assert inv_req.target == "radarr4k"
    assert inv_req.max_turns == 5
    assert inv_req.executor == ExecutorType.OPENCODE

    res = InvestigationResult(
        success=True,
        session_id="ses_abc",
        target="radarr4k",
        root_cause="Database locked by zombie thread",
        proposed_fix="docker compose restart radarr4k",
        category="database_error",
        turns_used=2,
        duration_seconds=3.5,
        transcript=[],
    )
    assert res.success is True
    assert res.session_id == "ses_abc"
    assert res.category == "database_error"


def test_investigation_request_str_exit_code():
    inv_req = InvestigationRequest(
        target="matter-hub",
        exit_code="1",
        error_logs="FATAL websocket connection refused",
    )
    assert inv_req.target == "matter-hub"
    assert inv_req.exit_code == "1"


def test_session_create_request():
    req = SessionCreateRequest(
        executor=ExecutorType.AGY,
        prompt="Initial prompt",
        system_prompt="SRE Persona",
        workspace="/containers/dev",
    )
    assert req.executor == ExecutorType.AGY
    assert req.prompt == "Initial prompt"
    assert req.system_prompt == "SRE Persona"
    assert req.workspace == "/containers/dev"

    # Test initial_prompt fallback
    req2 = SessionCreateRequest(initial_prompt="Fallback prompt")
    assert req2.prompt == "Fallback prompt"


def test_session_message_request():
    req1 = SessionMessageRequest(prompt="Follow up turn")
    assert req1.prompt == "Follow up turn"

    # Test message field alias/fallback
    req2 = SessionMessageRequest(message="Alternative message text")
    assert req2.prompt == "Alternative message text"
    assert req2.message == "Alternative message text"

    # Test validation error when neither prompt nor message provided
    with pytest.raises(ValidationError):
        SessionMessageRequest()

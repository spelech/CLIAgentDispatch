import json
from unittest.mock import AsyncMock, patch

import pytest

from cli_agent_dispatch.core.engine import DispatchEngine
from cli_agent_dispatch.core.models import (
    InvestigationRequest,
    InvestigationResult,
    TaskRequest,
    TaskResult,
)
from cli_agent_dispatch.core.sre_prompts import (
    MCG_READ_ONLY_DIRECTIVE,
    build_investigation_prompt,
)


def test_sre_prompt_contains_strict_mcg_read_only_directive():
    assert (
        "=== CRITICAL MODEL CONTEXT GATEWAY (MCG) SAFETY DIRECTIVE ===" in MCG_READ_ONLY_DIRECTIVE
    )
    assert "STRICTLY CONFINED to READ-ONLY inspection" in MCG_READ_ONLY_DIRECTIVE
    assert "search_tools" in MCG_READ_ONLY_DIRECTIVE
    assert "execute_tool" in MCG_READ_ONLY_DIRECTIVE
    assert (
        "DO NOT execute mutating, editing, restarting, creating, or deleting"
        in MCG_READ_ONLY_DIRECTIVE
    )
    assert "All proposed fixes must be returned as clean bash commands" in MCG_READ_ONLY_DIRECTIVE

    req = InvestigationRequest(
        target="my-service",
        exit_code=137,
        error_logs="OOMKilled process 123",
    )
    prompt = build_investigation_prompt(req)
    assert MCG_READ_ONLY_DIRECTIVE in prompt


def test_sre_prompt_contains_incident_details():
    req = InvestigationRequest(
        target="database-pg",
        exit_code=1,
        error_logs="FATAL: password authentication failed for user 'app'",
        workspace="/containers/db",
    )
    prompt = build_investigation_prompt(req)

    assert "database-pg" in prompt
    assert "FATAL: password authentication failed for user 'app'" in prompt
    assert "1" in prompt
    assert "/containers/db" in prompt
    assert "root_cause" in prompt
    assert "proposed_fix" in prompt
    assert "category" in prompt


@pytest.mark.asyncio
async def test_investigate_single_turn_complete():
    engine = DispatchEngine()
    req = InvestigationRequest(
        target="caddy",
        exit_code=1,
        error_logs="bind: address already in use :80",
    )

    expected_json = {
        "root_cause": "Port collision on 80",
        "proposed_fix": "fuser -k 80/tcp && docker compose restart caddy",
        "category": "configuration",
        "notes": "Port 80 occupied by stray process.",
    }

    mock_dispatch = AsyncMock(
        return_value=TaskResult(
            success=True,
            executor="opencode",
            output=json.dumps(expected_json),
            session_id="ses_caddy_single_123",
            duration_seconds=1.5,
        )
    )

    with patch.object(engine, "dispatch", mock_dispatch):
        result = await engine.investigate(req)

    assert isinstance(result, InvestigationResult)
    assert result.success is True
    assert result.target == "caddy"
    assert result.session_id == "ses_caddy_single_123"
    assert result.root_cause == "Port collision on 80"
    assert result.proposed_fix == "fuser -k 80/tcp && docker compose restart caddy"
    assert result.category == "configuration"
    assert result.turns_used == 1
    assert len(result.transcript) >= 1
    assert mock_dispatch.call_count == 1

    first_call_req: TaskRequest = mock_dispatch.call_args_list[0][0][0]
    assert first_call_req.system_prompt == MCG_READ_ONLY_DIRECTIVE
    assert "caddy" in first_call_req.prompt


@pytest.mark.asyncio
async def test_investigate_multi_turn_flow():
    engine = DispatchEngine()
    req = InvestigationRequest(
        target="influxdb",
        exit_code=1,
        error_logs="No space left on device",
        max_turns=3,
    )

    turn1_result = TaskResult(
        success=True,
        executor="opencode",
        output="I am examining disk usage and volume mounts across stacks.",
        session_id="ses_influx_multi_456",
        duration_seconds=1.0,
    )

    turn2_json = {
        "root_cause": "Docker overlay2 storage exhausted root filesystem",
        "proposed_fix": "docker system prune -a --volumes -f",
        "category": "resource",
        "notes": "Pruned 24GB unreferenced layers.",
    }
    turn2_result = TaskResult(
        success=True,
        executor="opencode",
        output=f"Here is the final diagnosis:\n```json\n{json.dumps(turn2_json)}\n```",
        session_id="ses_influx_multi_456",
        duration_seconds=1.2,
    )

    mock_dispatch = AsyncMock(side_effect=[turn1_result, turn2_result])

    with patch.object(engine, "dispatch", mock_dispatch):
        result = await engine.investigate(req)

    assert result.success is True
    assert result.turns_used == 2
    assert result.session_id == "ses_influx_multi_456"
    assert result.root_cause == "Docker overlay2 storage exhausted root filesystem"
    assert result.proposed_fix == "docker system prune -a --volumes -f"
    assert result.category == "resource"
    assert mock_dispatch.call_count == 2

    # Second turn must reuse session_id
    second_call_req: TaskRequest = mock_dispatch.call_args_list[1][0][0]
    assert second_call_req.session_id == "ses_influx_multi_456"


@pytest.mark.asyncio
async def test_investigate_fallback_on_unstructured_output():
    engine = DispatchEngine()
    req = InvestigationRequest(
        target="homeassistant",
        exit_code=2,
        error_logs="Database disk image is malformed",
        max_turns=1,
    )

    unstructured_text = (
        "Investigation complete:\n"
        "Root cause: SQLite home-assistant_v2.db corrupted after sudden power loss.\n"
        "Proposed fix: rm -f /config/home-assistant_v2.db* && docker compose restart homeassistant\n"
        "Category: storage\n"
    )

    mock_dispatch = AsyncMock(
        return_value=TaskResult(
            success=True,
            executor="opencode",
            output=unstructured_text,
            session_id="ses_ha_unstruct_789",
            duration_seconds=0.8,
        )
    )

    with patch.object(engine, "dispatch", mock_dispatch):
        result = await engine.investigate(req)

    assert result.success is True
    assert "corrupted" in result.root_cause or "home-assistant_v2.db" in result.root_cause
    assert (
        "docker compose restart homeassistant" in result.proposed_fix
        or "rm -f" in result.proposed_fix
    )
    assert result.category in ["storage", "unknown"]
    assert result.raw_output == unstructured_text


@pytest.mark.asyncio
async def test_investigate_handles_dispatch_failure():
    engine = DispatchEngine()
    req = InvestigationRequest(
        target="plex",
        exit_code=137,
        error_logs="Killed",
        max_turns=2,
    )

    mock_dispatch = AsyncMock(
        return_value=TaskResult(
            success=False,
            executor="opencode",
            output="",
            error="Executor process timed out after 120s",
        )
    )

    with patch.object(engine, "dispatch", mock_dispatch):
        result = await engine.investigate(req)

    assert result.success is False
    assert result.error == "Executor process timed out after 120s"
    assert result.turns_used == 1
    assert mock_dispatch.call_count == 1


@pytest.mark.asyncio
async def test_investigate_target_convenience_function():
    from cli_agent_dispatch.core.engine import investigate_target

    expected_json = {
        "root_cause": "Configuration syntax error in compose file",
        "proposed_fix": "docker compose config",
        "category": "configuration",
    }

    mock_dispatch = AsyncMock(
        return_value=TaskResult(
            success=True,
            executor="opencode",
            output=json.dumps(expected_json),
            session_id="ses_helper_123",
            duration_seconds=0.5,
        )
    )

    with patch("cli_agent_dispatch.core.engine.engine.dispatch", mock_dispatch):
        result = await investigate_target(
            target="nginx",
            exit_code=1,
            error_logs="syntax error",
        )

    assert result.success is True
    assert result.target == "nginx"
    assert result.root_cause == "Configuration syntax error in compose file"
    assert result.proposed_fix == "docker compose config"
    assert result.category == "configuration"

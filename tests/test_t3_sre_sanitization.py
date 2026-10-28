"""Tests for SRE prompt detection, metadata extraction, semantic thread titling, and sanitization."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent_dispatch.core.models import TaskRequest
from cli_agent_dispatch.core.sre_prompts import build_investigation_prompt
from cli_agent_dispatch.core.models import InvestigationRequest
from cli_agent_dispatch.executors.t3 import (
    T3Executor,
    build_sre_thread_title,
    extract_sre_metadata,
    is_sre_incident_prompt,
    sanitize_prompt_for_t3,
)


def test_is_sre_incident_prompt():
    req = InvestigationRequest(
        target="homebox",
        exit_code=137,
        error_logs="Out of memory",
    )
    sre_prompt = build_investigation_prompt(req)
    assert is_sre_incident_prompt(sre_prompt) is True

    openclaw_diag_prompt = """
    System: <!-- openclaw:attempt:STABLE -->
    DIAGNOSTIC ENVELOPE:
    Target: postgres-cluster
    Reason: CrashLoopBackOff
    Exit Code: 1
    OpenClaw SRE automated incident report
    """
    assert is_sre_incident_prompt(openclaw_diag_prompt) is True

    normal_prompt = "Please explain the Docker compose networks configuration."
    assert is_sre_incident_prompt(normal_prompt) is False
    assert is_sre_incident_prompt("") is False
    assert is_sre_incident_prompt(None) is False


def test_extract_sre_metadata_standard_format():
    prompt = """
=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: caddy
Exit Code / Status: 1
Workspace: /containers/webservices

=== ERROR LOGS & DIAGNOSTIC CONTEXT ===
bind: address already in use :80
    """
    meta = extract_sre_metadata(prompt)
    assert meta["target"] == "caddy"
    assert meta["exit_code"] == "1"
    assert meta["reason"] == "failure"


def test_extract_sre_metadata_with_status_hint_and_openclaw_envelope():
    prompt = """
DIAGNOSTIC ENVELOPE:
Target: homebox-app
Exit Code / Status: 137 (OOMKilled)
Reason: OutOfMemory
OpenClaw SRE
    """
    meta = extract_sre_metadata(prompt)
    assert meta["target"] == "homebox-app"
    assert meta["exit_code"] == "137"
    assert meta["reason"] == "OutOfMemory"

    prompt_paren_only = """
=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: redis-broker
Exit Code / Status: 143 (SIGTERM)
    """
    meta2 = extract_sre_metadata(prompt_paren_only)
    assert meta2["target"] == "redis-broker"
    assert meta2["exit_code"] == "143"
    assert meta2["reason"] == "SIGTERM"


def test_extract_sre_metadata_fallback_defaults():
    meta = extract_sre_metadata("Some random prompt text")
    assert meta["target"] == "unknown"
    assert meta["reason"] == "failure"
    assert meta["exit_code"] == "unknown"


def test_build_sre_thread_title():
    meta = {"target": "caddy", "reason": "port_conflict", "exit_code": "98"}
    title = build_sre_thread_title(meta)
    assert title == "🚨 [SRE] caddy (port_conflict | exit 98)"

    long_meta = {
        "target": "very-long-container-name-exceeding-standard-length-for-testing",
        "reason": "extremely-detailed-reason-why-the-application-failed-to-start-properly",
        "exit_code": "137",
    }
    long_title = build_sre_thread_title(long_meta)
    assert len(long_title) <= 80
    assert long_title.endswith("…")


def test_sanitize_prompt_for_t3():
    raw_prompt = """System: <!-- openclaw:attempt:STABLE --> You are OpenClaw agent runner.

## Tooling
Tools policy-filtered: 65 tools deferred to reduce context size:
- docker_restart
- docker_ps
- read_logs

<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>
{"agent_state": "recovering", "step": 3}
<<<END_OPENCLAW_INTERNAL_CONTEXT>>>

[OpenClaw heartbeat poll] 10:00:00
[OpenClaw heartbeat poll] 10:00:30
Follow the heartbeat monitor scratch context

=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: authelia
Exit Code / Status: 1
Workspace: /containers/webservices

=== ERROR LOGS & DIAGNOSTIC CONTEXT ===
configuration error: secret_key missing

=== REQUIRED OUTPUT FORMAT ===
```json
{
  "root_cause": "missing secret_key",
  "proposed_fix": "check secrets file"
}
```"""

    cleaned = sanitize_prompt_for_t3(raw_prompt)

    assert "System: <!-- openclaw:attempt:STABLE -->" not in cleaned
    assert "## Tooling" not in cleaned
    assert "Tools policy-filtered" not in cleaned
    assert "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" not in cleaned
    assert "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>" not in cleaned
    assert "[OpenClaw heartbeat poll]" not in cleaned
    assert "Follow the heartbeat monitor scratch context" not in cleaned

    assert "=== SRE INCIDENT INVESTIGATION REQUEST ===" in cleaned
    assert "Target Service / Container: authelia" in cleaned
    assert "=== ERROR LOGS & DIAGNOSTIC CONTEXT ===" in cleaned
    assert "configuration error: secret_key missing" in cleaned
    assert "=== REQUIRED OUTPUT FORMAT ===" in cleaned


@pytest.mark.asyncio
async def test_t3_executor_launches_thread_with_sre_title_and_sanitized_prompt():
    raw_prompt = """System: <!-- openclaw:attempt:STABLE --> Agent init
## Tooling
Tools policy-filtered: 65 tools deferred
[OpenClaw heartbeat poll]
<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>
{"id": 1}
<<<END_OPENCLAW_INTERNAL_CONTEXT>>>

=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: caddy
Exit Code / Status: 1
Reason: port_bind_error

=== ERROR LOGS & DIAGNOSTIC CONTEXT ===
bind: address already in use :80
"""

    mock_client = AsyncMock()
    mock_client.resolve_or_create_project.return_value = "proj_123"
    mock_client.launch_thread.return_value = {"threadId": "mcp:thread_sre_1", "runId": "run_1"}
    mock_client.wait_run.return_value = {"status": "completed"}
    mock_client.read_thread.return_value = {"items": [], "hasMore": False}
    mock_client.read_run_result_text.return_value = '{"root_cause": "Port in use"}'

    client_factory = MagicMock()
    client_factory.return_value.__aenter__.return_value = mock_client
    client_factory.return_value.__aexit__.return_value = None

    executor = T3Executor(client_factory=client_factory)

    task_req = TaskRequest(
        prompt=raw_prompt,
        workspace="/containers/webservices",
        model="t3-agy-flash",
    )

    result = await executor.execute(task_req)
    assert result.success is True
    assert result.session_id == "mcp:thread_sre_1"

    # Verify launch_thread args
    mock_client.launch_thread.assert_called_once()
    launch_kwargs = mock_client.launch_thread.call_args[1]
    assert launch_kwargs["title"] == "🚨 [SRE] caddy (port_bind_error | exit 1)"
    launched_msg = launch_kwargs["message"]
    assert "System: <!-- openclaw:attempt:STABLE -->" not in launched_msg
    assert "## Tooling" not in launched_msg
    assert "[OpenClaw heartbeat poll]" not in launched_msg
    assert "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" not in launched_msg
    assert "=== SRE INCIDENT INVESTIGATION REQUEST ===" in launched_msg


@pytest.mark.asyncio
async def test_t3_executor_launch_thread_method():
    mock_client = AsyncMock()
    mock_client.launch_thread.return_value = {"threadId": "mcp:t1", "runId": "r1"}
    executor = T3Executor()

    sre_msg = """=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: authelia
Exit Code / Status: 2
Reason: config_invalid

=== ERROR LOGS & DIAGNOSTIC CONTEXT ===
syntax error in configuration.yml
"""
    await executor.launch_thread(
        client=mock_client,
        project_id="p1",
        title="default title",
        message=sre_msg,
        instance_id="antigravity",
        model_slug="gemini-3.8-flash",
    )
    mock_client.launch_thread.assert_called_once()
    kwargs = mock_client.launch_thread.call_args[1]
    assert kwargs["title"] == "🚨 [SRE] authelia (config_invalid | exit 2)"
    assert "=== SRE INCIDENT INVESTIGATION REQUEST ===" in kwargs["message"]


from unittest.mock import MagicMock, patch

import pytest

from cli_agent_dispatch.core.models import ExecutorType, TaskRequest
from cli_agent_dispatch.executors.agy import (
    DEFAULT_AGY_MODELS,
    AgyExecutor,
    discover_agy_models,
    normalize_agy_model,
)


def test_normalize_agy_model():
    from cli_agent_dispatch.config import settings

    assert normalize_agy_model(None) == settings.agy_default_model
    assert normalize_agy_model("claude-3-5-sonnet") == "Claude Sonnet 5.5 (Medium)"
    assert normalize_agy_model("claude-opus") == "Claude Opus 5.5 (Medium)"
    assert normalize_agy_model("gpt-oss-120b") == "GPT-OSS 120B (Medium)"
    assert normalize_agy_model("gemini-flash-low") == "Gemini 3.8 Flash (Low)"
    assert normalize_agy_model("gemini-flash-high") == "Gemini 3.8 Flash (High)"
    assert normalize_agy_model("custom-exact-name") == "custom-exact-name"

    # Exact IDs & Argon models
    assert normalize_agy_model("gemini-4-argon-medium") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("gemini-4-argon-high") == "Gemini 4 Argon (High)"
    assert normalize_agy_model("gemini-4-argon-low") == "Gemini 4 Argon (Low)"
    assert normalize_agy_model("agy-argon") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("agy-argon-high") == "Gemini 4 Argon (High)"
    assert normalize_agy_model("agy-argon-low") == "Gemini 4 Argon (Low)"
    assert normalize_agy_model("argon") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("agy-opus") == "Claude Opus 5.5 (High)"
    assert normalize_agy_model("agy-sonnet") == "Claude Sonnet 5.5 (High)"
    assert normalize_agy_model("openai/gemini-4-argon-medium") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("litellm/agy-argon-high") == "Gemini 4 Argon (High)"


def test_discover_agy_models_parses_spinner_and_ansi():
    fake_stdout = (
        "\r\x1b[36m⠋\x1b[0m Fetching available models..."
        "\r\x1b[36m⠙\x1b[0m Fetching available models...\r"
        "gemini-4-argon-medium     Gemini 4 Argon (Medium)\n"
        "gemini-4-argon-high       Gemini 4 Argon (High)\n"
        "custom-agy-exp-high       Custom Experimental (High)\n"
    )
    mock_proc = MagicMock(returncode=0, stdout=fake_stdout, stderr="")
    with (
        patch("cli_agent_dispatch.executors.agy._resolve_agy_binary", return_value="/usr/bin/agy"),
        patch("subprocess.run", return_value=mock_proc),
    ):
        models = discover_agy_models(force_refresh=True)

    for k, v in DEFAULT_AGY_MODELS.items():
        assert models[k] == v
    assert models["custom-agy-exp-high"] == "Custom Experimental (High)"
    assert normalize_agy_model("custom-agy-exp-high") == "Custom Experimental (High)"


@pytest.mark.asyncio
async def test_agy_execute_stream_ndjson(tmp_path):
    fake_agy = tmp_path / "agy"
    script = (
        "#!/bin/sh\n"
        'echo \'{"type":"init","conversation_id":"conv-test-999"}\'\n'
        'echo \'{"type":"step_update","text_delta":"First token "}\'\n'
        'echo \'{"type":"step_update","text_delta":"second token"}\'\n'
        'echo \'{"type":"result","response":"full response","usage":{"total_tokens":42}}\'\n'
    )
    fake_agy.write_text(script)
    fake_agy.chmod(0o755)

    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Streaming task",
        executor=ExecutorType.AGY,
        workspace=str(tmp_path),
    )

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", str(fake_agy)):
        chunks = []
        async for chunk in executor.execute_stream(req):
            chunks.append(chunk)

    assert len(chunks) >= 3
    # Check conversation_id propagation
    assert any(c.session_id == "conv-test-999" for c in chunks)
    deltas = "".join(c.delta for c in chunks)
    assert "First token second token" in deltas
    # Final chunk verification
    final_chunk = next(c for c in chunks if c.is_final)
    assert final_chunk.finish_reason == "stop"


@pytest.mark.asyncio
async def test_agy_execute_stream_binary_not_found():
    executor = AgyExecutor()
    req = TaskRequest(prompt="task", executor=ExecutorType.AGY)

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", "/nonexistent/agy"):
        chunks = []
        async for chunk in executor.execute_stream(req):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].is_final is True
    assert "not found" in chunks[0].delta


def test_format_agy_tool_step():
    from cli_agent_dispatch.executors.agy import format_agy_tool_step

    active_cmd = {
        "step_index": 3,
        "state": "ACTIVE",
        "step_type": "tool",
        "tool_name": "run_command",
        "tool_info": {
            "name": "run_command",
            "parameters": {
                "CommandLine": "openclaw-updates check",
                "Cwd": "/containers/ai/openclaw",
            },
        },
    }
    res = format_agy_tool_step(active_cmd)
    assert res is not None
    reasoning_line, card_step = res
    assert "run_command" in reasoning_line
    assert "openclaw-updates check" in reasoning_line
    assert card_step["state"] == "ACTIVE"
    assert card_step["icon"] == "🛠️"

    done_cmd = {
        "step_index": 3,
        "state": "DONE",
        "step_type": "tool",
        "tool_name": "run_command",
        "duration_seconds": 1.25,
        "tool_info": {"name": "run_command", "parameters": {}},
    }
    res_done = format_agy_tool_step(done_cmd)
    assert res_done is not None
    r_done, c_done = res_done
    assert "1.2s" in r_done or "1.3s" in r_done
    assert c_done["state"] == "DONE"
    assert c_done["duration"] == 1.25


def test_resolve_openclaw_posture(tmp_path):
    import json

    from cli_agent_dispatch.executors.agy import resolve_openclaw_posture

    leases_file = tmp_path / "mode_leases.json"
    leases_file.write_text(
        json.dumps(
            {
                "global": {
                    "mode": "plan",
                    "scope": "infra",
                    "until": None,
                    "revert_to_mode": "normal",
                    "revert_to_scope": "home",
                },
                "sessions": {
                    "agent:main:thread1": {
                        "mode": "autopilot",
                        "scope": "all",
                        "until": None,
                    }
                },
            }
        )
    )

    mode, scope, flags, guardrail = resolve_openclaw_posture(
        "Runtime: agent=main | session=agent:main:main | channel=slack",
        leases_path=str(leases_file),
    )
    assert mode == "plan"
    assert scope == "infra"
    assert flags == ["--mode", "plan"]
    assert "PLAN (READ-ONLY)" in guardrail

    mode_s, scope_s, flags_s, guardrail_s = resolve_openclaw_posture(
        "Runtime: agent=main | session=agent:main:thread1 | channel=slack",
        leases_path=str(leases_file),
    )
    assert mode_s == "autopilot"
    assert scope_s == "all"
    assert "--dangerously-skip-permissions" in flags_s
    assert "AUTOPILOT" in guardrail_s


@pytest.mark.asyncio
async def test_slack_activity_tracker_posts_and_finalizes():
    from cli_agent_dispatch.executors.agy import SlackActivityTracker, format_agy_tool_step

    tracker = SlackActivityTracker(
        run_id="run_test_123",
        model="Gemini 4 Argon (Medium)",
        mode="normal",
        scope="home",
        prompt="Runtime: agent=main | channel=slack",
        enabled=True,
    )
    tracker.token = "xoxb-fake-token"
    tracker.enabled = True

    calls = []

    def fake_api_call(method, payload):
        calls.append((method, payload))
        return {"ok": True, "ts": "1700000000.123456"}

    tracker._slack_api_call_sync = fake_api_call

    _, step1 = format_agy_tool_step(
        {
            "step_index": 1,
            "state": "ACTIVE",
            "step_type": "tool",
            "tool_name": "view_file",
            "tool_info": {"parameters": {"AbsolutePath": "/containers/README.md"}},
        }
    )
    await tracker.record_step(step1)
    assert tracker.message_ts == "1700000000.123456"
    assert calls[0][0] == "chat.postMessage"
    assert any(
        b.get("type") == "actions"
        and any(el.get("action_id") == "abort_agy_turn" for el in b.get("elements", []))
        for b in calls[0][1]["blocks"]
    )

    await tracker.finalize(status="aborted")
    assert calls[-1][0] == "chat.update"
    assert "Turn Aborted by User" in calls[-1][1]["text"]


@pytest.mark.asyncio
async def test_agy_execute_stream_emits_reasoning_delta_for_tools(tmp_path):
    fake_agy = tmp_path / "agy"
    script = (
        "#!/bin/sh\n"
        'echo \'{"event":"init","conversation_id":"conv-tools-1"}\'\n'
        'echo \'{"event":"step_update","step_update":{"conversation_id":"conv-tools-1","step_index":1,"state":"ACTIVE","step_type":"tool","tool_name":"view_file","tool_info":{"name":"view_file","parameters":{"AbsolutePath":"/containers/ai/openclaw/VERSION"}}}}\'\n'
        'echo \'{"event":"step_update","step_update":{"conversation_id":"conv-tools-1","step_index":1,"state":"DONE","step_type":"tool","tool_name":"view_file","duration_seconds":0.05,"tool_info":{"name":"view_file","parameters":{"AbsolutePath":"/containers/ai/openclaw/VERSION"}}}}\'\n'
        'echo \'{"event":"step_update","step_update":{"conversation_id":"conv-tools-1","text_delta":"Version is 0.1.4"}}\'\n'
        'echo \'{"event":"result","result":{"conversation_id":"conv-tools-1","status":"DONE","response":"Version is 0.1.4"}}\'\n'
    )
    fake_agy.write_text(script)
    fake_agy.chmod(0o755)

    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Check version",
        executor=ExecutorType.AGY,
        workspace=str(tmp_path),
    )

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", str(fake_agy)):
        chunks = [c async for c in executor.execute_stream(req)]

    reasoning_chunks = [c.reasoning_delta for c in chunks if c.reasoning_delta]
    assert len(reasoning_chunks) == 2
    assert "view_file" in reasoning_chunks[0]
    assert "/containers/ai/openclaw/VERSION" in reasoning_chunks[0]
    assert "done" in reasoning_chunks[1]
    text_deltas = "".join(c.delta for c in chunks)
    assert "Version is 0.1.4" in text_deltas

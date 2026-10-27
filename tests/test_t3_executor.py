import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cli_agent_dispatch.api.anthropic_routes import (
    _resolve_executor_and_model as resolve_anthropic_model,
)
from cli_agent_dispatch.api.openai_routes import (
    _resolve_executor_and_model as resolve_openai_model,
)
from cli_agent_dispatch.api.session_routes import _resolve_session_executor
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.executors.agy import SlackActivityTracker
from cli_agent_dispatch.executors.t3 import (
    T3Executor,
    _title_from_prompt,
    discover_t3_models,
    format_t3_activity_item,
    normalize_t3_model_selection,
)
from cli_agent_dispatch.main import app
from cli_agent_dispatch.mcp.server import check_executor_health_tool, list_available_models_tool

# ---------------------------------------------------------------------------
# 1. Unit Tests: Model & Provider Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "provider", "effort", "expected"),
    [
        (None, None, None, ("antigravity", "gemini-3.8-flash-high")),
        ("t3", None, None, ("antigravity", "gemini-3.8-flash-high")),
        ("t3-agy", None, "low", ("antigravity", "gemini-3.8-flash-low")),
        ("t3-agy-flash-medium", None, None, ("antigravity", "gemini-3.8-flash-medium")),
        ("t3-agy-argon", None, None, ("antigravity", "gemini-4-argon-medium")),
        ("t3-agy-argon-high", None, None, ("antigravity", "gemini-4-argon-high")),
        ("t3-agy-argon", None, "max", ("antigravity", "gemini-4-argon-high")),
        ("t3-agy-argon-low", None, None, ("antigravity", "gemini-4-argon-low")),
        ("t3-opus", None, None, ("opencode", "google-agy/claude-opus-5-5")),
        ("t3-agy-opus", None, None, ("opencode", "google-agy/claude-opus-5-5")),
        ("t3-sonnet", None, None, ("opencode", "google-agy/claude-sonnet-5-5")),
        ("t3-agy-sonnet", None, None, ("opencode", "google-agy/claude-sonnet-5-5")),
        ("t3-gpt-oss", None, None, ("opencode", "google-agy/gpt-oss-120b-medium")),
        ("t3-opencode", None, None, ("opencode", "google-agy/gemini-3.8-flash")),
        ("t3-opencode-argon", None, None, ("opencode", "google-agy/gemini-4-argon")),
        ("t3-opencode-glm", None, None, ("opencode", "litellm/glm")),
        ("t3-opencode-deepseek-pro", None, None, ("opencode", "litellm/deepseek-pro")),
        ("t3/antigravity/gemini-4-argon-high", None, None, ("antigravity", "gemini-4-argon-high")),
        ("t3/opencode/litellm/qwen-max", None, None, ("opencode", "litellm/qwen-max")),
        ("t3/opencode/glm-flash", None, None, ("opencode", "litellm/glm-flash")),
        ("t3-acp/gemini-3.8-flash-low", None, None, ("antigravity", "gemini-3.8-flash-low")),
        ("t3-opencode/claude-opus-5-5", None, None, ("opencode", "google-agy/claude-opus-5-5")),
        ("openai/t3-agy-argon-high", None, None, ("antigravity", "gemini-4-argon-high")),
        ("glm", "opencode", None, ("opencode", "litellm/glm")),
        ("gemini-3.8-flash-low", "opencode", None, ("opencode", "google-agy/gemini-3.8-flash")),
    ],
)
def test_normalize_t3_model_selection(
    model: str | None,
    provider: str | None,
    effort: str | None,
    expected: tuple[str, str],
) -> None:
    assert normalize_t3_model_selection(model, provider, effort) == expected


def test_discover_t3_models_and_route_resolvers() -> None:
    models = discover_t3_models()
    assert "t3" in models
    assert "t3-agy-flash" in models
    assert "t3-agy-argon" in models
    assert "t3-opencode-sonnet" in models

    assert resolve_openai_model("t3-agy-flash") == (ExecutorType.T3, "t3-agy-flash")
    assert resolve_openai_model("openai/t3-opencode-glm") == (ExecutorType.T3, "t3-opencode-glm")
    assert resolve_anthropic_model("t3-agy-opus") == (ExecutorType.T3, "t3-agy-opus")
    assert _resolve_session_executor("mcp:1234-5678", None) == ExecutorType.T3


def test_format_t3_activity_item_and_title() -> None:
    # Skip workspace preparation
    ws_item = {
        "itemId": "turn-item:workspace-preparation:1",
        "type": "command_execution",
        "status": "completed",
        "title": "Workspace ready",
        "text": "$ Preparing workspace",
    }
    assert format_t3_activity_item(ws_item, 1) is None

    # Non-tool item skipped
    msg_item = {
        "itemId": "turn-item:msg:1",
        "type": "assistant_message",
        "status": "completed",
        "text": "Hello",
    }
    assert format_t3_activity_item(msg_item, 1) is None

    # Active command execution
    cmd_item = {
        "itemId": "turn-item:cmd:1",
        "type": "command_execution",
        "status": "inProgress",
        "title": "Run command",
        "text": "$ pytest -q\nrunning...",
    }
    res = format_t3_activity_item(cmd_item, 2)
    assert res is not None
    reasoning_line, card_step = res
    assert "🛠️ *Run command*: `pytest -q`" in reasoning_line
    assert card_step["state"] == "ACTIVE"
    assert card_step["icon"] == "🛠️"

    # Completed MCP tool call
    mcp_item = {
        "itemId": "turn-item:mcp:1",
        "type": "mcp_tool_call",
        "status": "completed",
        "title": "mcg.search_tools",
        "text": "query=docker",
    }
    res_done = format_t3_activity_item(mcp_item, 3)
    assert res_done is not None
    assert "✅ *mcg.search_tools*" in res_done[0]
    assert res_done[1]["state"] == "DONE"

    # Title helper skips system/runtime headers
    prompt = "System: You are an SRE agent\nRuntime: agent=main\nUser: Investigate why caddy failed\nMore details"
    assert _title_from_prompt(prompt) == "Investigate why caddy failed"


def test_slack_activity_tracker_includes_t3_hijack_button() -> None:
    tracker = SlackActivityTracker(
        run_id="mcp:abc-123",
        model="antigravity/gemini-3.8-flash-high",
        mode="normal",
        scope="home",
        prompt="channel=slack",
        enabled=False,
        t3_ui_url="https://ttt.wileyriley.com",
    )
    _, blocks = tracker._build_blocks(status="running")
    actions = blocks[1]["elements"]
    action_ids = [el["action_id"] for el in actions]
    assert "abort_agy_turn" in action_ids
    assert "open_t3_ui" in action_ids
    assert "open_openclaw_ui" in action_ids
    t3_btn = next(el for el in actions if el["action_id"] == "open_t3_ui")
    assert t3_btn["url"] == "https://ttt.wileyriley.com"


# ---------------------------------------------------------------------------
# 2. Unit Tests: T3Executor with FakeT3McpClient
# ---------------------------------------------------------------------------


class FakeT3McpClient:
    """In-memory mock of T3McpClient for deterministic unit testing."""

    instances: list["FakeT3McpClient"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.configured_calls: list[tuple[str, str, str]] = []
        self.launched_calls: list[dict[str, Any]] = []
        self.sent_calls: list[tuple[str, str]] = []
        self.interrupted_calls: list[tuple[str, str | None]] = []
        self.thread_provider = "antigravity"
        self.thread_model = "gemini-3.8-flash-high"
        self.wait_responses: list[dict[str, Any]] = [
            {"timedOut": False, "status": "completed", "runId": "run:1"}
        ]
        self.reply_text = "MOCK_T3_REPLY"
        FakeT3McpClient.instances.append(self)

    async def __aenter__(self) -> "FakeT3McpClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        pass

    async def resolve_or_create_project(self, workspace: str) -> str:
        return "proj-123"

    async def launch_thread(self, **kwargs: Any) -> dict[str, Any]:
        self.launched_calls.append(kwargs)
        return {"threadId": "mcp:unit-thread-1", "projectId": "proj-123", "runId": "run:1"}

    async def configure_thread(
        self, thread_id: str, instance_id: str, model_slug: str
    ) -> dict[str, Any]:
        self.configured_calls.append((thread_id, instance_id, model_slug))
        self.thread_provider = instance_id
        self.thread_model = model_slug
        return {"sequence": 1}

    async def send_message(
        self, thread_id: str, message: str, mode: str = "queue"
    ) -> dict[str, Any]:
        self.sent_calls.append((thread_id, message))
        return {"runId": "run:2", "delivery": "sent"}

    async def wait_run(self, thread_id: str, run_id: str, timeout_ms: int = 1500) -> dict[str, Any]:
        if self.wait_responses:
            return self.wait_responses.pop(0)
        return {"timedOut": False, "status": "completed", "runId": run_id}

    async def read_thread(
        self, thread_id: str, view: str = "activity", **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "thread": {
                "threadId": thread_id,
                "providerInstanceId": self.thread_provider,
                "model": self.thread_model,
            },
            "items": [
                {
                    "position": 1,
                    "itemId": "item-cmd-1",
                    "runId": "run:1",
                    "type": "command_execution",
                    "status": "completed",
                    "title": "Run command",
                    "text": "$ git status",
                }
            ],
            "hasMore": False,
            "nextPosition": None,
        }

    async def read_run_result_text(
        self, thread_id: str, run_id: str, item_type: str = "assistant_message"
    ) -> str:
        if item_type == "assistant_message":
            return self.reply_text
        return "mock error"

    async def interrupt_thread(
        self, thread_id: str, run_id: str | None = None, reason: str = ""
    ) -> dict[str, Any]:
        self.interrupted_calls.append((thread_id, run_id))
        return {"interrupted": True}


@pytest.mark.asyncio
async def test_t3_executor_launch_and_mid_thread_model_switch() -> None:
    FakeT3McpClient.instances.clear()
    executor = T3Executor(client_factory=FakeT3McpClient)  # type: ignore[arg-type]

    # Turn 1: Launch new thread on Antigravity ACP
    req1 = TaskRequest(
        prompt="Runtime: agent=main\nInspect repo",
        executor=ExecutorType.T3,
        model="t3-agy-flash",
        workspace="/containers/dev/CLIAgentDispatch",
    )
    with patch(
        "cli_agent_dispatch.executors.t3.resolve_openclaw_posture",
        return_value=("plan", "home", ["--mode", "plan"], "\n[GUARDRAIL]"),
    ):
        res1 = await executor.execute(req1)

    assert res1.success is True
    assert res1.output == "MOCK_T3_REPLY"
    assert res1.session_id == "mcp:unit-thread-1"
    client1 = FakeT3McpClient.instances[0]
    assert len(client1.launched_calls) == 1
    assert client1.launched_calls[0]["interaction_mode"] == "plan"
    assert "[GUARDRAIL]" in client1.launched_calls[0]["message"]

    # Turn 2: Resume thread and switch provider/model to OpenCode Claude Sonnet 5.5
    req2 = TaskRequest(
        prompt="Now switch to Sonnet and summarize",
        session_id="mcp:unit-thread-1",
        executor=ExecutorType.T3,
        model="t3-opencode-sonnet",
    )
    res2 = await executor.execute(req2)
    assert res2.success is True
    client2 = FakeT3McpClient.instances[1]
    assert client2.configured_calls == [
        ("mcp:unit-thread-1", "opencode", "google-agy/claude-sonnet-5-5")
    ]
    assert client2.sent_calls == [("mcp:unit-thread-1", "Now switch to Sonnet and summarize")]


@pytest.mark.asyncio
async def test_t3_executor_stream_and_cancel() -> None:
    FakeT3McpClient.instances.clear()
    executor = T3Executor(client_factory=FakeT3McpClient)  # type: ignore[arg-type]

    req = TaskRequest(
        prompt="Stream test",
        executor=ExecutorType.T3,
        model="t3-agy-argon-high",
    )
    chunks = [c async for c in executor.execute_stream(req)]
    assert any(c.session_id == "mcp:unit-thread-1" for c in chunks)
    assert any("Run command" in c.reasoning_delta for c in chunks)
    assert any(c.delta == "MOCK_T3_REPLY" for c in chunks)
    assert chunks[-1].is_final is True

    # Test cancellation callback in session_store
    cancelled_flag = False

    async def _dummy_cancel() -> bool:
        nonlocal cancelled_flag
        cancelled_flag = True
        return True

    session_store.register_cancel_callback("mcp:cancel-test", _dummy_cancel)
    assert "mcp:cancel-test" in session_store.active_process_ids()
    killed = await session_store.kill_active_process("mcp:cancel-test")
    assert killed is True
    assert cancelled_flag is True
    assert session_store.was_cancelled("mcp:cancel-test") is True


@pytest.mark.asyncio
async def test_mcp_server_tools_include_t3() -> None:
    models_json = json.loads(await list_available_models_tool())
    model_ids = [m["id"] for m in models_json["models"]]
    assert "t3-agy-flash" in model_ids
    assert "t3-opencode-sonnet" in model_ids

    health_json = json.loads(await check_executor_health_tool())
    assert "t3" in health_json["executors"]
    assert health_json["executors"]["t3"]["available"] is True


# ---------------------------------------------------------------------------
# 3. Live End-to-End (E2E) Tests against Running T3 Daemon (127.0.0.1:3773)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_e2e_t3_multi_turn_and_model_switch() -> None:
    """
    End-to-end test against the live local T3 Code server:
      1. Turn 1 launches a new T3 thread using `antigravity` (`gemini-3.8-flash-high`).
      2. Turn 2 reconfigures the same thread to `opencode` (`google-agy/gemini-3.8-flash`)
         and verifies multi-turn continuity and response.
    """
    executor = T3Executor()

    # Turn 1: Antigravity ACP provider
    turn1 = await executor.execute(
        TaskRequest(
            prompt="Remember the secret code word: PINEAPPLE_42. Reply with the single word: ACK_1",
            executor=ExecutorType.T3,
            model="t3-agy-flash-high",
            workspace="/containers/dev/CLIAgentDispatch",
            timeout=90,
        )
    )
    assert turn1.success is True, f"Turn 1 failed: {turn1.error}"
    assert turn1.session_id is not None and turn1.session_id.startswith("mcp:")
    assert "ACK_1" in turn1.output

    # Turn 2: Switch model on the same thread to gemini-3.8-flash-low and verify
    turn2 = await executor.execute(
        TaskRequest(
            prompt="What was the secret code word I told you in the previous message? Reply with just the code word.",
            session_id=turn1.session_id,
            executor=ExecutorType.T3,
            model="t3-agy-flash-low",
            workspace="/containers/dev/CLIAgentDispatch",
            timeout=90,
        )
    )
    assert turn2.success is True, f"Turn 2 failed: {turn2.error}"
    assert turn2.session_id == turn1.session_id
    assert "PINEAPPLE_42" in turn2.output


def test_live_e2e_openai_route_with_t3() -> None:
    """End-to-end test of `/v1/chat/completions` with `model='t3-opencode-flash'`."""
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "t3-opencode-flash",
            "messages": [{"role": "user", "content": "Reply with the exact single word: E2E_OK"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert "E2E_OK" in content
    assert resp.headers.get("x-native-session-id", "").startswith("mcp:")

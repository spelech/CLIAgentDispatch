import json
import os
import shutil
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cli_agent_dispatch.api.anthropic_routes import (
    _resolve_executor_and_model as resolve_anthropic_model,
)
from cli_agent_dispatch.api.openai_routes import (
    _resolve_executor_and_model as resolve_openai_model,
)
from cli_agent_dispatch.api.session_routes import _resolve_session_executor
from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.executors.agy import SlackActivityTracker
from cli_agent_dispatch.executors.t3 import (
    T3Executor,
    T3McpClient,
    T3McpError,
    _title_from_prompt,
    discover_t3_models,
    format_t3_activity_item,
    normalize_t3_model_selection,
)
from cli_agent_dispatch.main import app
from cli_agent_dispatch.mcp.server import (
    check_executor_health_tool,
    generate_t3_pairing_link_tool,
    list_available_models_tool,
)

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

    with (
        patch("cli_agent_dispatch.mcp.server.os.path.exists", return_value=True),
        patch("cli_agent_dispatch.mcp.server.os.access", return_value=True),
    ):
        health_json = json.loads(await check_executor_health_tool())
    assert "t3" in health_json["executors"]
    assert health_json["executors"]["t3"]["available"] is True


@pytest.mark.asyncio
async def test_t3_mcp_client_protocol_and_oauth_flow(tmp_path: Any) -> None:
    """Unit test T3McpClient OAuth token minting, disk caching, JSON-RPC / SSE parsing, and project resolution."""
    tok_file = str(tmp_path / "t3_token.json")
    fake_cli = str(tmp_path / "t3")
    with open(fake_cli, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")

    T3McpClient._cached_token = None
    T3McpClient._cached_expires_at = 0.0

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(
        return_value=(json.dumps({"credential": "PAIR123"}).encode("utf-8"), b"")
    )

    def _make_resp(
        json_data: dict[str, Any] | None = None,
        text_data: str = "",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> MagicMock:
        r = MagicMock()
        r.status_code = status_code
        r.headers = headers or {"content-type": "application/json"}
        r.text = text_data or (json.dumps(json_data) if json_data is not None else "")
        r.json = MagicMock(return_value=json_data or {})
        r.raise_for_status = MagicMock()
        return r

    oauth_client = AsyncMock()
    oauth_client.__aenter__.return_value = oauth_client
    oauth_client.post = AsyncMock(
        side_effect=[
            _make_resp({"client_id": "cid-1"}),
            _make_resp({"redirectTo": "http://127.0.0.1:8032/callback?code=authcode-1"}),
            _make_resp({"access_token": "tok-minted-1", "expires_in": 2592000}),
        ]
    )

    mcp_http = AsyncMock()
    mcp_http.post = AsyncMock(
        side_effect=[
            # initialize
            _make_resp(
                {"jsonrpc": "2.0", "id": "init-0", "result": {}},
                headers={"mcp-session-id": "sess-xyz", "content-type": "application/json"},
            ),
            # notifications/initialized
            _make_resp({}),
            # t3_project_list (empty -> triggers t3_project_create)
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-1",
                    "result": {"structuredContent": {"projects": [], "nextCursor": None}},
                }
            ),
            # t3_project_create via SSE envelope
            _make_resp(
                text_data='event: message\ndata: {"jsonrpc":"2.0","id":"cad-2","result":{"structuredContent":{"id":"proj-created-1"}}}\n\n',
                headers={"content-type": "text/event-stream"},
            ),
            # launch_thread
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-3",
                    "result": {"structuredContent": {"threadId": "mcp:new-1"}},
                }
            ),
            # configure_thread
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-4",
                    "result": {"structuredContent": {"ok": True}},
                }
            ),
            # send_message
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-5",
                    "result": {"structuredContent": {"ok": True}},
                }
            ),
            # wait_run
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-6",
                    "result": {"structuredContent": {"status": "completed", "timedOut": False}},
                }
            ),
            # read_thread (page 1 for read_run_result_text, with truncated text)
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-7",
                    "result": {
                        "structuredContent": {
                            "items": [
                                {
                                    "itemId": "it-1",
                                    "runId": "run-1",
                                    "type": "assistant_message",
                                    "text": "HELLO_",
                                    "textTruncated": True,
                                    "nextTextOffset": 6,
                                }
                            ],
                            "hasMore": False,
                            "nextPosition": None,
                        }
                    },
                }
            ),
            # read_thread (continuation chunk for read_full_item_text)
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-8",
                    "result": {
                        "structuredContent": {
                            "items": [
                                {
                                    "itemId": "it-1",
                                    "runId": "run-1",
                                    "type": "assistant_message",
                                    "text": "WORLD",
                                    "textTruncated": False,
                                    "nextTextOffset": None,
                                }
                            ]
                        }
                    },
                }
            ),
            # interrupt_thread
            _make_resp(
                {
                    "jsonrpc": "2.0",
                    "id": "cad-9",
                    "result": {"structuredContent": {"ok": True}},
                }
            ),
        ]
    )
    mcp_http.delete = AsyncMock(return_value=_make_resp({}))
    mcp_http.aclose = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        patch("httpx.AsyncClient", side_effect=[mcp_http, oauth_client]),
    ):
        client = T3McpClient(
            server_url="http://127.0.0.1:3773",
            cli_path=fake_cli,
            token_cache_path=tok_file,
        )
        async with client as c:
            proj_id = await c.resolve_or_create_project(str(tmp_path))
            assert proj_id == "proj-created-1"
            launched = await c.launch_thread(
                proj_id, "Title", "Hi", "antigravity", "gemini-3.8-flash-high"
            )
            assert launched["threadId"] == "mcp:new-1"
            await c.configure_thread("mcp:new-1", "opencode", "google-agy/claude-sonnet-5-5")
            await c.send_message("mcp:new-1", "Followup")
            await c.wait_run("mcp:new-1", "run-1", timeout_ms=5000)
            full_text = await c.read_run_result_text("mcp:new-1", "run-1", "assistant_message")
            assert full_text == "HELLO_WORLD"
            await c.interrupt_thread("mcp:new-1", run_id="run-1")

    # Verify disk cache reload path
    T3McpClient._cached_token = None
    T3McpClient._cached_expires_at = 0.0
    client2 = T3McpClient(
        server_url="http://127.0.0.1:3773",
        cli_path=fake_cli,
        token_cache_path=tok_file,
    )
    loaded_tok = await client2.get_token()
    assert loaded_tok == "tok-minted-1"
    assert T3McpError("boom").args[0] == "boom"


@pytest.mark.asyncio
async def test_sso_pair_routes_and_ntfy_push(tmp_path: Any) -> None:
    """Test `/sso-pair` and `/v1/t3/pair` redirect, card, JSON, ntfy push, and MCP tool."""
    fake_cli = str(tmp_path / "t3")
    with open(fake_cli, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(
        return_value=(
            json.dumps(
                {
                    "id": "pair-id-1",
                    "credential": "PAIRCODE99",
                    "label": "Steve (15m)",
                    "expiresAt": "2026-10-27T05:00:00Z",
                    "pairUrl": "https://ttt.wileyriley.com/pair#token=PAIRCODE99",
                }
            ).encode("utf-8"),
            b"",
        )
    )

    ntfy_http = AsyncMock()
    ntfy_http.__aenter__.return_value = ntfy_http
    ntfy_resp = MagicMock()
    ntfy_resp.status_code = 200
    ntfy_resp.raise_for_status = MagicMock()
    ntfy_http.post = AsyncMock(return_value=ntfy_resp)

    with (
        patch.object(settings, "t3_cli_path", fake_cli),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        patch("cli_agent_dispatch.api.t3_pair_routes.httpx.AsyncClient", return_value=ntfy_http),
    ):
        client = TestClient(app)

        # 1. Redirect mode (?mode=redirect)
        resp_redir = client.get(
            "/sso-pair?mode=redirect",
            headers={"Remote-Name": "Steven Pelech"},
            follow_redirects=False,
        )
        assert resp_redir.status_code == 302
        assert resp_redir.headers["location"] == "https://ttt.wileyriley.com/pair#token=PAIRCODE99"

        # 2. Default Card HTML mode (with Copy Pairing Code button & ntfy push)
        resp_card = client.get(
            "/sso-pair?notify=1",
            headers={"Remote-User": "steve"},
        )
        assert resp_card.status_code == 200
        assert "PAIRCODE99" in resp_card.text
        assert "Copy Pairing Code" in resp_card.text
        assert "Sent to ntfy" in resp_card.text

        # 3. JSON mode
        resp_json = client.get("/v1/t3/pair?mode=json&notify=1")
        assert resp_json.status_code == 200
        jdata = resp_json.json()
        assert jdata["credential"] == "PAIRCODE99"
        assert jdata["pair_url"] == "https://ttt.wileyriley.com/pair#token=PAIRCODE99"
        assert jdata["ntfy_sent"] is True

        # 4. Copy-only helper route (/sso-pair/copy)
        resp_copy = client.get("/sso-pair/copy?code=PAIRCODE99&expires=15m")
        assert resp_copy.status_code == 200
        assert "Copy Code: PAIRCODE99" in resp_copy.text

        # Verify ntfy payload omits "click" (so clicking opens ntfy notification)
        # and includes "📋 Copy Code" + "🚀 Pair This Browser" actions
        ntfy_payload = ntfy_http.post.call_args.kwargs["json"]
        assert "click" not in ntfy_payload
        assert ntfy_payload["markdown"] is True
        action_labels = [a["label"] for a in ntfy_payload["actions"]]
        assert "📋 Copy Code" in action_labels
        assert "🚀 Pair This Browser" in action_labels

        # 5. MCP tool generate_t3_pairing_link
        mcp_out = json.loads(
            await generate_t3_pairing_link_tool(ttl="1h", label="Tablet", send_ntfy=True)
        )
        assert mcp_out["credential"] == "PAIRCODE99"
        assert mcp_out["ntfy_sent"] is True


# ---------------------------------------------------------------------------
# 3. Live End-to-End (E2E) Tests against Running T3 Daemon (127.0.0.1:3773)
# ---------------------------------------------------------------------------

_LIVE_T3_AVAILABLE = os.environ.get("RUN_LIVE_T3_E2E") == "1" and bool(
    os.path.exists(settings.t3_cli_path) or shutil.which(settings.t3_cli_path)
)


@pytest.mark.skipif(not _LIVE_T3_AVAILABLE, reason="Live T3 Code daemon not installed on CI runner")
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


@pytest.mark.skipif(not _LIVE_T3_AVAILABLE, reason="Live T3 Code daemon not installed on CI runner")
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

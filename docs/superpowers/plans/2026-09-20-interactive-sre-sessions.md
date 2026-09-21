# Interactive SRE Sessions & Multi-Turn Dispatch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable true multi-turn, back-and-forth investigation sessions in CLIAgentDispatch across both OpenCode and Antigravity (agy), with access to Model Context Gateway (MCG) tools governed by a strict read-only system prompt directive.

**Architecture:**
1. Extend core models with persistent `session_id`, `InvestigationRequest`, and `InvestigationResult`.
2. Update `OpenCodeExecutor` to reuse existing HTTP server sessions (`POST /session/{id}/message`) across turns.
3. Update `AgyExecutor` to use `--output-format json` and `--conversation <id>` for seamless conversation resumption.
4. Implement an SRE investigation module with strict system prompt enforcement restricting MCG tools to read-only diagnostic inspection (forbidding out-of-band mutations).
5. Expose `/v1/sessions` and `/v1/investigate` endpoints on the FastAPI gateway.
6. Verify 100% test pass rate and $\ge$ 80% code coverage.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, HTTPX, Pytest.

## Global Constraints
- Target repository: `/containers/dev/CLIAgentDispatch` on branch `feature/interactive-sre-sessions`.
- Python virtual environment: `/containers/dev/CLIAgentDispatch/.venv/bin/python` and `.venv/bin/pytest`.
- Maintain code coverage $\ge$ 80%.
- Strict backwards compatibility for existing OpenAI bridge (`/v1/chat/completions`) and FastMCP endpoints.
- MCG Tool Safety: All investigation prompts MUST enforce read-only tool usage for MCG tools.

---

### Task 1: Core Models & Session Abstraction

**Files:**
- Modify: `src/cli_agent_dispatch/core/models.py`
- Test: `tests/test_session_models.py`

**Interfaces:**
- `TaskRequest(prompt: str, session_id: str | None = None, system_prompt: str | None = None, ...)`
- `InvestigationRequest(target: str, exit_code: str | int | None, error_logs: str, ...)`
- `InvestigationResult(success: bool, session_id: str | None, root_cause: str, proposed_fix: str, ...)`
- `SessionCreateRequest`, `SessionMessageRequest`

- [ ] **Step 1: Write failing test for session models**

```python
# tests/test_session_models.py
from cli_agent_dispatch.core.models import (
    TaskRequest,
    InvestigationRequest,
    InvestigationResult,
    SessionCreateRequest,
    SessionMessageRequest,
    ExecutorType
)

def test_task_request_accepts_session_id():
    req = TaskRequest(prompt="Inspect logs", session_id="ses_12345", system_prompt="Be an SRE")
    assert req.session_id == "ses_12345"
    assert req.system_prompt == "Be an SRE"

def test_investigation_request_and_result():
    inv_req = InvestigationRequest(
        target="radarr4k",
        exit_code=1,
        error_logs="FATAL database locked",
        workspace="/containers/media_content"
    )
    assert inv_req.target == "radarr4k"
    assert inv_req.max_turns == 5

    res = InvestigationResult(
        success=True,
        session_id="ses_abc",
        target="radarr4k",
        root_cause="Database locked by zombie thread",
        proposed_fix="docker compose restart radarr4k",
        category="database_error",
        turns_used=2,
        duration_seconds=3.5,
        transcript=[]
    )
    assert res.success is True
```

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_session_models.py -v`
Expected: FAIL (ValidationError / missing fields)

- [ ] **Step 3: Update `src/cli_agent_dispatch/core/models.py`**
Implement the new fields on `TaskRequest` and the new Pydantic models.

- [ ] **Step 4: Run test to verify it passes**
Run: `.venv/bin/pytest tests/test_session_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**
```bash
git add src/cli_agent_dispatch/core/models.py tests/test_session_models.py
git commit -m "feat(models): add session_id, InvestigationRequest, and InvestigationResult schemas"
```

---

### Task 2: Multi-Turn Session Support in OpenCode Executor

**Files:**
- Modify: `src/cli_agent_dispatch/executors/opencode.py`
- Test: `tests/test_opencode_sessions.py`

**Interfaces:**
- `OpenCodeExecutor.execute(request: TaskRequest) -> TaskResult`

- [ ] **Step 1: Write failing test for OpenCode session resumption**

```python
# tests/test_opencode_sessions.py
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from cli_agent_dispatch.core.models import TaskRequest
from cli_agent_dispatch.executors.opencode import OpenCodeExecutor

@pytest.mark.asyncio
async def test_opencode_reuses_existing_session_id():
    executor = OpenCodeExecutor()
    req = TaskRequest(
        prompt="Second turn message",
        session_id="existing_session_999",
        executor="opencode"
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
```

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_opencode_sessions.py -v`
Expected: FAIL

- [ ] **Step 3: Update `src/cli_agent_dispatch/executors/opencode.py`**
1. Check `request.session_id`. If present, skip session creation and use `session_id = request.session_id`.
2. If absent, create new session and record its `session_id`.
3. If `request.system_prompt` is provided on new session, prepend or inject initial context.
4. Return `session_id` in `TaskResult`.

- [ ] **Step 4: Run test to verify it passes**
Run: `.venv/bin/pytest tests/test_opencode_sessions.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**
```bash
git add src/cli_agent_dispatch/executors/opencode.py tests/test_opencode_sessions.py
git commit -m "feat(opencode): support session resumption and multi-turn message dispatch"
```

---

### Task 3: Multi-Turn Conversation Resumption in Agy Executor

**Files:**
- Modify: `src/cli_agent_dispatch/executors/agy.py`
- Test: `tests/test_agy_sessions.py`

**Interfaces:**
- `AgyExecutor.execute(request: TaskRequest) -> TaskResult`

- [ ] **Step 1: Write failing test for Agy conversation resumption**

```python
# tests/test_agy_sessions.py
import pytest
import json
from unittest.mock import AsyncMock, patch
from cli_agent_dispatch.core.models import TaskRequest
from cli_agent_dispatch.executors.agy.py import AgyExecutor

@pytest.mark.asyncio
async def test_agy_uses_conversation_flag_when_session_id_provided():
    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Second turn message",
        session_id="conv_xyz_123",
        executor="agy"
    )

    mock_process = AsyncMock()
    mock_process.returncode = 0
    json_output = json.dumps({
        "conversation_id": "conv_xyz_123",
        "status": "SUCCESS",
        "response": "Agy turn 2 reply",
        "num_turns": 2
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
```

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_agy_sessions.py -v`
Expected: FAIL

- [ ] **Step 3: Update `src/cli_agent_dispatch/executors/agy.py`**
1. Always add `--output-format json` to command arguments.
2. If `request.session_id` is provided, add `--conversation {request.session_id}`.
3. Parse JSON output from `stdout`. Extract `conversation_id` -> `session_id` and `response` -> `output`.
4. Fallback gracefully if output is plain text.

- [ ] **Step 4: Run test to verify it passes**
Run: `.venv/bin/pytest tests/test_agy_sessions.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**
```bash
git add src/cli_agent_dispatch/executors/agy.py tests/test_agy_sessions.py
git commit -m "feat(agy): parse JSON output and support --conversation resumption"
```

---

### Task 4: SRE Prompt with Strict MCG Read-Only Directive & Investigation Engine

**Files:**
- Create: `src/cli_agent_dispatch/core/sre_prompts.py`
- Modify: `src/cli_agent_dispatch/core/engine.py`
- Test: `tests/test_sre_investigation.py`

**Interfaces:**
- `build_investigation_prompt(request: InvestigationRequest) -> str`
- `DispatchEngine.investigate(request: InvestigationRequest) -> InvestigationResult`

- [ ] **Step 1: Write failing test for SRE prompt and MCG read-only directive**

```python
# tests/test_sre_investigation.py
from cli_agent_dispatch.core.models import InvestigationRequest
from cli_agent_dispatch.core.sre_prompts import build_investigation_prompt, MCG_READ_ONLY_DIRECTIVE

def test_sre_prompt_contains_strict_mcg_read_only_directive():
    req = InvestigationRequest(
        target="matter-hub",
        exit_code="1",
        error_logs="FATAL websocket connection refused"
    )
    prompt = build_investigation_prompt(req)
    assert MCG_READ_ONLY_DIRECTIVE in prompt
    assert "READ-ONLY" in prompt
    assert "DO NOT execute mutating" in prompt
```

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_sre_investigation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `sre_prompts.py` and `DispatchEngine.investigate`**
In `src/cli_agent_dispatch/core/sre_prompts.py`:
Define the prompt template with:
```
=== CRITICAL MODEL CONTEXT GATEWAY (MCG) SAFETY DIRECTIVE ===
You have access to Model Context Gateway tools (via `mcg`).
You are STRICTLY CONFINED to READ-ONLY inspection when using MCG tools.
ALLOWED MCG ACTIONS:
- search_tools
- execute_tool for queries, read-only status inspections, reading notes, and diagnostic queries.
FORBIDDEN MCG ACTIONS:
- DO NOT execute mutating, editing, restarting, creating, or deleting tool calls via MCG.
- All proposed fixes must be returned as clean bash commands in your structured JSON response.
```
In `src/cli_agent_dispatch/core/engine.py`:
Implement `investigate(request: InvestigationRequest) -> InvestigationResult`:
- Builds initial prompt with the MCG safety directive.
- Dispatches first turn to the selected executor (OpenCode or agy).
- Extracts session_id.
- If the agent needs follow-up or returns diagnostics, continues up to `max_turns`.
- Parses final structured JSON (`root_cause`, `proposed_fix`, `category`).
- Returns complete `InvestigationResult` with transcript.

- [ ] **Step 4: Run test to verify it passes**
Run: `.venv/bin/pytest tests/test_sre_investigation.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**
```bash
git add src/cli_agent_dispatch/core/sre_prompts.py src/cli_agent_dispatch/core/engine.py tests/test_sre_investigation.py
git commit -m "feat(investigate): add SRE investigation engine with strict MCG read-only prompt directive"
```

---

### Task 5: Interactive Session & Investigation API Endpoints

**Files:**
- Create: `src/cli_agent_dispatch/api/session_routes.py`
- Modify: `src/cli_agent_dispatch/main.py`
- Test: `tests/test_session_routes.py`

**Interfaces:**
- `POST /v1/investigate`
- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/message`

- [ ] **Step 1: Write failing test for session API endpoints**

```python
# tests/test_session_routes.py
import pytest
from httpx import AsyncClient, ASGITransport
from cli_agent_dispatch.main import app

@pytest.mark.asyncio
async def test_api_investigate_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/investigate", json={
            "target": "frigate",
            "exit_code": 1,
            "error_logs": "FFmpeg exited with code 1"
        })
    assert resp.status_code in [200, 500]
```

- [ ] **Step 2: Run test to verify failure**
Run: `.venv/bin/pytest tests/test_session_routes.py -v`
Expected: FAIL (404 Not Found)

- [ ] **Step 3: Implement `session_routes.py` and register in `main.py`**
1. Add `POST /v1/investigate` calling `engine.investigate(request)`.
2. Add `POST /v1/sessions` calling `engine.dispatch(TaskRequest(...))` to initialize a session and return `{ "session_id": "...", "executor": "..." }`.
3. Add `POST /v1/sessions/{session_id}/message` calling `engine.dispatch(TaskRequest(prompt=..., session_id=session_id))`.
4. Register router in `main.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `.venv/bin/pytest tests/test_session_routes.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**
```bash
git add src/cli_agent_dispatch/api/session_routes.py src/cli_agent_dispatch/main.py tests/test_session_routes.py
git commit -m "feat(api): expose /v1/investigate and /v1/sessions interactive endpoints"
```

---

### Task 6: Full Regression Verification & Coverage Gate

**Files:**
- Test: All tests in `tests/`

- [ ] **Step 1: Run full test suite with coverage**
Run: `.venv/bin/pytest --cov=src/cli_agent_dispatch --cov-report=term-missing`
Expected: All tests pass, coverage $\ge$ 80%

- [ ] **Step 2: Verify live health endpoint**
Verify `main.py` and CLI entry points load cleanly.

- [ ] **Step 3: Final commit and summary**
Commit any remaining docs or cleanup.

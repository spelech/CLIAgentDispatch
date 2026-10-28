# CLIAgentDispatch (v0.5.0)

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](pyproject.toml)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-orange.svg)](https://modelcontextprotocol.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Universal Headless & CLI AI Agent Delegation Gateway. Bridges local AI execution engines (**T3 Code**, **Antigravity `agy`**, and **OpenCode**) to **Model Context Gateway (MCG)**, **LiteLLM**, and **OpenClaw**.

---

## 🌟 Capabilities

1. **Triple Execution Engines**:
   - **T3 Code (`t3` / `t3-code`)**: Headless JSON-RPC MCP executor with automated tool permissions, session pairing, mid-thread model switching, and real-time execution feedback.
   - **Antigravity (`agy`)**: Native subprocess CLI execution with `--model`, `--effort` (low/medium/high) reasoning mapping, OSC 7501 suppression, streaming token parsing, and non-blocking timeout handling.
   - **OpenCode**: Primary HTTP REST API (`http://localhost:4096/session`) with automatic fallback to headless CLI subprocess execution (`opencode run`).
2. **OpenAI Compatibility Bridge (`/v1`)**: Exposes standard `/v1/models` and `/v1/chat/completions` (real-time SSE streaming, `reasoning_delta`, and multi-turn thread continuity), allowing LiteLLM and OpenClaw to delegate to local agents directly as conversational models.
3. **Anthropic Messages Bridge (`/v1/messages`)**: Full Anthropic API compatibility with streaming event protocol (`message_start`, `content_block_delta`, `thinking_delta`, `message_stop`) for native OpenClaw and extended reasoning integration.
4. **FastMCP Server (`/mcp` / `/sse`)**: Exposes 9 comprehensive agent management and SRE tools directly to MCG and any Model Context Protocol client:
   - `delegate_task`: Universal task execution across `t3`, `agy`, or `opencode`.
   - `investigate_incident`: Automated multi-turn diagnostic investigation and root-cause analysis.
   - `generate_t3_pairing_link`: On-demand minting of SSO-protected pairing tokens.
   - `list_active_sessions`: Audit active client and native engine sessions.
   - `close_session` & `force_kill_session`: Session lifecycle and process management.
   - `list_available_models`: Dynamic discovery of underlying models across all engines.
   - `check_executor_health`: Deep health probes across HTTP, CLI, and socket listeners.
   - `inspect_workspace_diff`: Uncommitted workspace git diff inspection.
5. **Interactive Operator & Observability Suite**:
   - **SSO Pairing Portal (`/sso-pair`)**: Interactive web card with one-click "Copy Pairing Code" button for zero-friction terminal pairing behind TinyAuth/Authentik.
   - **ntfy Push Alerts**: Automated push notifications dispatched when pairing is requested, including quick-copy actions.
   - **Slack Live Activity Cards**: Live interactive Slack notifications showing tool calls, streaming status, and operator actions ("Stop Turn" and "Hijack Thread").
   - **Visual Debugging (`/v1/screenshot`)**: On-demand visual inspection and multimodal content normalization.
6. **Direct Python Library**: Fully importable by other local services (e.g. `monitorbot`) via `from cli_agent_dispatch import delegate_task`.

---

## 🚀 Quickstart

### Installation
```bash
git clone git@github.com:spelech/CLIAgentDispatch.git
cd CLIAgentDispatch
uv sync
```

### CLI Usage
```bash
# Run task with T3 Code (default/recommended for coding)
uv run cli-agent-dispatch run --executor t3 "Refactor authentication layer"

# Run task with Antigravity (agy)
uv run cli-agent-dispatch run --executor agy "Analyze disk mount health"

# Run task with OpenCode
uv run cli-agent-dispatch run --executor opencode "Audit security configurations in /containers"
```

### Starting the Gateway Server
```bash
# Start FastAPI + FastMCP on port 8032
uv run cli-agent-dispatch serve --port 8032
```

---

## 🧪 Testing & Verification
```bash
# Run unit tests and coverage report (enforces >= 80% coverage)
uv run pytest

# Verify release consistency and link integrity
python3 scripts/verify_release.py

# Linting and formatting with ruff
uv run ruff check .
uv run ruff format --check .
```

---

## 📜 Architecture & Contracts
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full Mermaid topologies, sequence diagrams, and endpoint contracts.

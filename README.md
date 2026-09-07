# CLIAgentDispatch

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-orange.svg)](https://modelcontextprotocol.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Universal Headless & CLI AI Agent Delegation Gateway. Bridges local AI execution engines (**OpenCode** daemon and **Antigravity `agy`** CLI) to **Model Context Gateway (MCG)**, **LiteLLM**, and **LibreChat**.

---

## 🌟 Capabilities

1. **FastMCP Server (`/mcp` / `/sse`)**: Exposes the `delegate_task` tool directly to MCG and any Model Context Protocol client.
2. **OpenAI Compatibility Bridge (`/v1`)**: Exposes standard `/v1/models` and `/v1/chat/completions` (streaming & non-streaming), allowing LibreChat and LiteLLM to use OpenCode and agy directly as conversational models.
3. **Dual Execution Engines**:
   - **OpenCode**: Primary HTTP REST API (`http://localhost:4096/session`) with automatic fallback to headless CLI subprocess execution (`opencode run`).
   - **Antigravity (`agy`)**: Native subprocess CLI execution with `--model`, `--dangerously-skip-permissions`, and non-blocking timeout handling.
4. **Direct Python Library**: Fully importable by other local services (e.g. `monitorbot`) via `from cli_agent_dispatch import delegate_task`.

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
# Run task with OpenCode (default)
uv run cli-agent-dispatch run "Audit security configurations in /containers"

# Run task with Antigravity (agy)
uv run cli-agent-dispatch run --executor agy "Analyze disk mount health"
```

### Starting the Server
```bash
# Start FastAPI + FastMCP on port 8028
uv run cli-agent-dispatch serve --port 8028
```

---

## 🧪 Testing & Verification
```bash
# Run unit tests and coverage report (enforces >= 80% coverage)
uv run pytest

# Linting with ruff
uv run ruff check .
```

---

## 📜 Architecture & Contracts
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full Mermaid topologies, sequence diagrams, and endpoint contracts.

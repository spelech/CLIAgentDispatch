# AGENTS.md

Instructions and architectural guidelines for AI coding agents working on **CLIAgentDispatch**.

## 🏛️ Archetype & Core Philosophy
This project strictly follows the **`python-fastapi-mcp`** archetype from Steven T. Pelech's **AgenticEngineeringToolbelt**:
- **Runtime**: Python 3.12+ managed via `uv`.
- **Packaging**: `pyproject.toml` with `hatchling`.
- **Web & API**: FastAPI + Uvicorn with Pydantic v2 schemas and `pydantic-settings`.
- **MCP Server**: FastMCP / MCP SDK mounted on FastAPI at `/mcp` and `/sse`.
- **Code Discipline**: Modular single-responsibility services, zero junk-drawer files (`utils.py`, `helpers.py`), semantic naming, rich debug payloads on errors.
- **Coverage**: Maintain $\ge 80\%$ test coverage across unit tests, FastMCP tool tests, and OpenAI bridge tests. Zero linter warnings via `ruff`.

## 🛑 Guardrails & Rules
1. **Host Isolation**: `CLIAgentDispatch` runs on host `10.0.0.10` to interface natively with `opencode` (`http://localhost:4096`) and `agy` (`/home/steve/.local/bin/agy`). Never attempt to containerize or hot-patch without explicit user instruction.
2. **Subprocess Sanitization**: All CLI invocations must pass arguments as sanitized arrays (`subprocess.run(cmd, ...)`), never with `shell=True`. Always enforce a non-blocking timeout.
3. **No Unhandled Errors**: Never return raw stack traces or unhandled 500s. Wrap errors in structured `ExecutionResult` or typed OpenAI error envelopes.
4. **Mandatory Test Verification**: Before committing or pushing changes, verify with `uv run pytest` and `uv run ruff check .`.

## 🛠️ Verification Commands
```bash
# Linting & Formatting
uv run ruff check .
uv run ruff format --check .

# Automated Tests & Coverage
uv run pytest

# Run Service Locally
uv run cli-agent-dispatch serve --port 8028
```

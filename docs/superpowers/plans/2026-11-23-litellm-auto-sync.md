# Automatic CLIAgentDispatch → LiteLLM Base-Model Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically sync `CLIAgentDispatch`'s discovered Antigravity (`agy`) base models—with exact per-model `reasoning_effort_levels` in `model_info`—directly into LiteLLM via LiteLLM's live Admin API, and migrate all downstream callers off legacy tier-suffixed model IDs with zero backward-compatibility shims.

**Architecture:** `executors/agy.py` parses `agy models` into a typed base-model catalog (`build_agy_model_catalog`) and resolves `model + reasoning_effort` dynamically to exact `agy` display names for both known and future models. A single-responsibility `LiteLLMSyncService` (`core/litellm_sync.py`) reconciles `managed_by="cli-agent-dispatch"` DB deployments against `GET /v1/model/info` using `POST /model/new`, `PATCH /model/{id}/update`, and `POST /model/delete` on startup, on a periodic schedule, and via `POST /api/litellm/sync`. LiteLLM YAML and OpenClaw configs are cleaned up directly to drop all suffixed `agy-*-<tier>` entries and keep T3 strictly out of LiteLLM.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, `pydantic-settings`, `httpx`, `pytest`, `pytest-asyncio`, `ruff`, `uv`.

## Global Constraints

- Follow `python-fastapi-mcp` archetype: single-responsibility modules, zero junk-drawer files, Pydantic v2 schemas, zero unhandled 500s.
- All CLI invocations must pass sanitized argument arrays (`subprocess.run(cmd, ...)`), never `shell=True`, always with timeout.
- Maintain $\ge 80\%$ test coverage across `uv run pytest`, zero `uv run ruff check .` warnings, and clean `uv run ruff format --check .`.
- Never log `LITELLM_MASTER_KEY` or other secrets.
- Never host T3 models in LiteLLM (OpenClaw contract: `/containers/ai/openclaw/tests/test_config_validation.py:39`).
- Never prune LiteLLM deployments unless live `agy models` succeeded (`from_live_cli=True`) and returned non-empty results.
- Never modify or delete LiteLLM entries that do not have `model_info.managed_by == "cli-agent-dispatch"` and `model_info.db_model is True`.

---

## File Structure

- **Modify:** `src/cli_agent_dispatch/core/models.py` — Add `AgyBaseModelSpec`, `LiteLLMSyncResult`, and optional reasoning metadata fields on `ModelCard`.
- **Modify:** `src/cli_agent_dispatch/executors/agy.py` — Add `discover_agy_models_with_status()`, `build_agy_model_catalog()`, effort-aware dynamic `normalize_agy_model(model, explicit_effort=None)`, and pass `request.effort` in `AgyExecutor.execute()` and `execute_stream()`.
- **Modify:** `src/cli_agent_dispatch/api/openai_routes.py` — Remove suffixed `agy-*-<tier>` entries from `STATIC_MODEL_ALIASES`; populate `/v1/models` from `build_agy_model_catalog()` with reasoning metadata; route all discovered base slugs to `ExecutorType.AGY`.
- **Modify:** `src/cli_agent_dispatch/executors/t3.py` — Remove tier-suffixed `t3-agy-*-low/medium/high/xhigh` entries from `T3_STATIC_ALIASES`.
- **Modify:** `src/cli_agent_dispatch/config.py` — Add `litellm_sync_enabled`, `litellm_base_url`, `litellm_master_key`, `litellm_env_file`, `litellm_sync_interval_seconds`, `cad_public_api_base`.
- **Create:** `src/cli_agent_dispatch/core/litellm_sync.py` — `LiteLLMSyncService` + singleton `litellm_sync_service` + master-key `.env` resolver + payload builder + reconciliation loop.
- **Modify:** `src/cli_agent_dispatch/main.py` — Start/stop `litellm_sync_loop()` in `lifespan`, add `POST /api/litellm/sync`, include `litellm_sync` in `GET /api/diagnostics`.
- **Create:** `tests/test_agy_catalog_and_normalization.py` — Unit tests for `discover_agy_models_with_status`, `build_agy_model_catalog`, and effort-aware `normalize_agy_model`.
- **Create:** `tests/test_litellm_sync.py` — Unit + API tests for `LiteLLMSyncService`, `POST /api/litellm/sync`, and `GET /api/diagnostics`.
- **Modify:** `tests/test_openai_bridge.py` — Update `/v1/models` assertions to verify clean base models + `reasoning_effort_levels` and absence of suffixed `agy-*-high`/`low` clutter.
- **Modify (Host Configs):** `/containers/ai/litellm/config/config.yaml`, `/containers/ai/litellm/config/router_rules.yaml`, `/containers/ai/openclaw/config/openclaw.json`, `/containers/ai/openclaw/config/system_prompt.md`, `/containers/ai/openclaw/data/workspace/skills/model-switcher/SKILL.md`, `/containers/ai/openclaw/data/models.json`.

---

### Task 1: Dynamic AGY Base-Model Catalog & Effort-Aware Normalization

**Files:**
- Modify: `src/cli_agent_dispatch/core/models.py:439-449`
- Modify: `src/cli_agent_dispatch/executors/agy.py:50-240,718-765,887-935`
- Create: `tests/test_agy_catalog_and_normalization.py`

**Interfaces:**
- Consumes: `settings.agy_default_model` from `cli_agent_dispatch.config`
- Produces:
  - `AgyBaseModelSpec` (Pydantic model in `cli_agent_dispatch.core.models`)
  - `LiteLLMSyncResult` (Pydantic model in `cli_agent_dispatch.core.models`)
  - `ModelCard` with optional `reasoning_effort_levels: list[str] | None = None` and `default_reasoning_effort: str | None = None`
  - `discover_agy_models_with_status(force_refresh: bool = False) -> tuple[dict[str, str], bool]` in `cli_agent_dispatch.executors.agy`
  - `build_agy_model_catalog(discovered: dict[str, str] | None = None, include_default_alias: bool = True) -> list[AgyBaseModelSpec]` in `cli_agent_dispatch.executors.agy`
  - `normalize_agy_model(model: str | None, explicit_effort: str | None = None) -> str` in `cli_agent_dispatch.executors.agy`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_agy_catalog_and_normalization.py`:

```python
from unittest.mock import MagicMock, patch

from cli_agent_dispatch.executors import agy as agy_mod


SAMPLE_AGY_DISCOVERED = {
    "gemini-4-neon-high": "Gemini 4 Neon (High)",
    "gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)",
    "gemini-4-neon-medium": "Gemini 4 Neon (Medium)",
    "gemini-4-neon-low": "Gemini 4 Neon (Low)",
    "gemini-4-carbon-medium": "Gemini 4 Carbon (Medium)",
    "gemini-4-carbon-high": "Gemini 4 Carbon (High)",
    "gemini-4-carbon-low": "Gemini 4 Carbon (Low)",
    "gemini-4-argon-medium": "Gemini 4 Argon (Medium)",
    "gemini-4-argon-high": "Gemini 4 Argon (High)",
    "gemini-4-argon-low": "Gemini 4 Argon (Low)",
    "gemini-4-hydrogen-high": "Gemini 4 Hydrogen (High)",
    "gemini-4-hydrogen-xhigh": "Gemini 4 Hydrogen (XHigh)",
    "gemini-4-hydrogen-medium": "Gemini 4 Hydrogen (Medium)",
    "gemini-4-hydrogen-low": "Gemini 4 Hydrogen (Low)",
    "gemini-pro-orchestrator-neon": "Orchestrator (Neon)",
    "gemini-pro-orchestrator": "Orchestrator (Carbon)",
    "gemini-pro-orchestrator-argon": "Orchestrator (Argon)",
    "gemini-3.8-flash-high": "Gemini 3.8 Flash (High)",
    "gemini-3.8-flash-medium": "Gemini 3.8 Flash (Medium)",
    "gemini-3.8-flash-low": "Gemini 3.8 Flash (Low)",
    "claude-opus-5-5-low": "Claude Opus 5.5 (Low)",
    "claude-opus-5-5-medium": "Claude Opus 5.5 (Medium)",
    "claude-opus-5-5-high": "Claude Opus 5.5 (High)",
    "claude-sonnet-5-5-low": "Claude Sonnet 5.5 (Low)",
    "claude-sonnet-5-5-medium": "Claude Sonnet 5.5 (Medium)",
    "claude-sonnet-5-5-high": "Claude Sonnet 5.5 (High)",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
    "gemini-5-titanium-low": "Gemini 5 Titanium (Low)",
    "gemini-5-titanium-high": "Gemini 5 Titanium (High)",
    "gemini-5-titanium-xhigh": "Gemini 5 Titanium (XHigh)",
}


def test_build_agy_model_catalog_groups_tiers_and_derives_future_models():
    catalog = agy_mod.build_agy_model_catalog(discovered=SAMPLE_AGY_DISCOVERED)
    by_name = {spec.litellm_model_name: spec for spec in catalog}

    assert "agy" in by_name
    assert by_name["agy-hydrogen"].base_slug == "gemini-4-hydrogen"
    assert by_name["agy-hydrogen"].reasoning_effort_levels == ["low", "medium", "high", "xhigh"]
    assert by_name["agy-hydrogen"].default_reasoning_effort == "medium"
    assert by_name["agy-hydrogen"].brand == "google"
    assert by_name["agy-hydrogen"].max_input_tokens == 1048576

    assert by_name["agy-neon"].reasoning_effort_levels == ["low", "medium", "high", "xhigh"]
    assert by_name["agy-carbon"].reasoning_effort_levels == ["low", "medium", "high"]
    assert by_name["agy-argon"].reasoning_effort_levels == ["low", "medium", "high"]
    assert by_name["agy-flash"].default_reasoning_effort == "high"
    assert by_name["agy-opus"].brand == "anthropic"
    assert by_name["agy-opus"].max_input_tokens == 200000
    assert by_name["agy-opus"].default_reasoning_effort == "high"
    assert by_name["agy-sonnet"].default_reasoning_effort == "high"
    assert by_name["agy-gpt-oss"].reasoning_effort_levels == ["medium"]
    assert by_name["agy-gpt-oss"].default_reasoning_effort == "medium"

    assert by_name["agy-orchestrator-neon"].reasoning_effort_levels == []
    assert by_name["agy-orchestrator-neon"].default_reasoning_effort is None
    assert by_name["agy-orchestrator-carbon"].base_slug == "gemini-pro-orchestrator"
    assert by_name["agy-orchestrator-argon"].base_slug == "gemini-pro-orchestrator-argon"

    # Brand-new unknown future model automatically gets agy-<base_slug>
    assert "agy-gemini-5-titanium" in by_name
    assert by_name["agy-gemini-5-titanium"].reasoning_effort_levels == ["low", "high", "xhigh"]
    assert by_name["agy-gemini-5-titanium"].default_reasoning_effort == "high"


def test_normalize_agy_model_uses_explicit_effort_for_known_and_future_models():
    with patch.object(agy_mod, "discover_agy_models", return_value=SAMPLE_AGY_DISCOVERED):
        assert (
            agy_mod.normalize_agy_model("agy-hydrogen", explicit_effort="xhigh")
            == "Gemini 4 Hydrogen (XHigh)"
        )
        assert (
            agy_mod.normalize_agy_model("agy-hydrogen", explicit_effort="low")
            == "Gemini 4 Hydrogen (Low)"
        )
        assert agy_mod.normalize_agy_model("agy-hydrogen") == "Gemini 4 Hydrogen (Medium)"
        assert (
            agy_mod.normalize_agy_model("agy-opus", explicit_effort="low")
            == "Claude Opus 5.5 (Low)"
        )
        assert (
            agy_mod.normalize_agy_model("agy-gemini-5-titanium", explicit_effort="xhigh")
            == "Gemini 5 Titanium (XHigh)"
        )
        assert (
            agy_mod.normalize_agy_model("agy-gemini-5-titanium", explicit_effort="low")
            == "Gemini 5 Titanium (Low)"
        )
        assert agy_mod.normalize_agy_model("agy-gemini-5-titanium") == "Gemini 5 Titanium (High)"
        # Requested effort higher than available falls back to highest available tier
        assert (
            agy_mod.normalize_agy_model("agy-argon", explicit_effort="xhigh")
            == "Gemini 4 Argon (High)"
        )


def test_discover_agy_models_with_status_tracks_live_vs_fallback():
    mock_ok = MagicMock()
    mock_ok.returncode = 0
    mock_ok.stdout = "gemini-5-titanium-high  Gemini 5 Titanium (High)\n"
    with (
        patch.object(agy_mod, "_resolve_agy_binary", return_value="/usr/bin/agy"),
        patch("subprocess.run", return_value=mock_ok),
    ):
        models, from_live = agy_mod.discover_agy_models_with_status(force_refresh=True)
        assert from_live is True
        assert models["gemini-5-titanium-high"] == "Gemini 5 Titanium (High)"
        # Live discovery must not leak stale DEFAULT_AGY_MODELS not returned by CLI
        assert "gemini-4-argon-low" not in models

    with (
        patch.object(agy_mod, "_resolve_agy_binary", return_value="/usr/bin/agy"),
        patch("subprocess.run", side_effect=OSError("boom")),
    ):
        models, from_live = agy_mod.discover_agy_models_with_status(force_refresh=True)
        assert from_live is False
        assert "gemini-4-hydrogen-high" in models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agy_catalog_and_normalization.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'build_agy_model_catalog'`

- [ ] **Step 3: Add `AgyBaseModelSpec`, `LiteLLMSyncResult`, and optional reasoning fields on `ModelCard` in `src/cli_agent_dispatch/core/models.py`**

Edit `src/cli_agent_dispatch/core/models.py` around line 439:

```python
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "cli-agent-dispatch"
    reasoning_effort_levels: list[str] | None = Field(default=None, exclude_none=True)
    default_reasoning_effort: str | None = Field(default=None, exclude_none=True)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class AgyBaseModelSpec(BaseModel):
    base_slug: str
    litellm_model_name: str
    display_name: str
    reasoning_effort_levels: list[str] = Field(default_factory=list)
    default_reasoning_effort: str | None = None
    tier_display_names: dict[str, str] = Field(default_factory=dict)
    brand: str = "antigravity"
    max_input_tokens: int = 131072
    max_output_tokens: int = 32768
    supports_video: bool = False


class LiteLLMSyncResult(BaseModel):
    status: str
    synced_at: str
    from_live_cli: bool = False
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    skipped_conflicts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
```

*(Note: verify `model_dump(exclude_none=True)` or FastAPI response serialization omits `None` fields on `ModelCard` — in FastAPI, `response_model_exclude_none=True` on `/v1/models` guarantees clean OpenAI JSON.)*

- [ ] **Step 4: Implement `discover_agy_models_with_status`, `build_agy_model_catalog`, and effort-aware `normalize_agy_model` in `src/cli_agent_dispatch/executors/agy.py`**

In `src/cli_agent_dispatch/executors/agy.py`:
1. Import `AgyBaseModelSpec` from `cli_agent_dispatch.core.models`.
2. Add constants right below `DEFAULT_AGY_MODELS`:

```python
_TIER_ORDER: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_TIER_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<tier>low|medium|high|xhigh|max)$", re.IGNORECASE)
_DISPLAY_TIER_RE = re.compile(r"\s*\((low|medium|high|xhigh|max)\)\s*$", re.IGNORECASE)

KNOWN_AGY_LITELLM_NAMES: dict[str, str] = {
    "gemini-4-hydrogen": "agy-hydrogen",
    "gemini-4-neon": "agy-neon",
    "gemini-4-carbon": "agy-carbon",
    "gemini-4-argon": "agy-argon",
    "gemini-3.8-flash": "agy-flash",
    "claude-opus-5-5": "agy-opus",
    "claude-sonnet-5-5": "agy-sonnet",
    "gpt-oss-120b": "agy-gpt-oss",
    "gemini-pro-orchestrator-neon": "agy-orchestrator-neon",
    "gemini-pro-orchestrator": "agy-orchestrator-carbon",
    "gemini-pro-orchestrator-argon": "agy-orchestrator-argon",
}

_HIGH_DEFAULT_BASE_SLUGS: set[str] = {
    "gemini-3.8-flash",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
}
```

3. Track `_AGY_MODELS_CACHE_LIVE: bool = False` alongside `_AGY_MODELS_CACHE` and `_AGY_MODELS_CACHE_TS`.
4. Implement `discover_agy_models_with_status(force_refresh: bool = False) -> tuple[dict[str, str], bool]`:
   - If cached and fresh, return `(dict(_AGY_MODELS_CACHE), _AGY_MODELS_CACHE_LIVE)`.
   - Run `[agy_bin, "models"]` with timeout `10s`.
   - If `returncode == 0` and `_parse_agy_models_output(proc.stdout)` is non-empty: store **only** `discovered` (so retired `agy` models are actually detected as gone!) and set `_AGY_MODELS_CACHE_LIVE = True`.
   - Otherwise fall back to `dict(DEFAULT_AGY_MODELS)` and set `_AGY_MODELS_CACHE_LIVE = False`.
   - Update `discover_agy_models(force_refresh: bool = False) -> dict[str, str]` to return `discover_agy_models_with_status(force_refresh=force_refresh)[0]`.

5. Implement `build_agy_model_catalog(discovered: dict[str, str] | None = None, include_default_alias: bool = True) -> list[AgyBaseModelSpec]`:
   - Uses `discovered if discovered is not None else discover_agy_models()`.
   - Groups each `(model_id, display_name)` by `base_slug` (stripping trailing `-low|-medium|-high|-xhigh|-max` via `_TIER_SUFFIX_RE`).
   - Collects `tier_display_names[tier] = display_name` and base `display_name` (`_DISPLAY_TIER_RE.sub("", display_name).strip()`).
   - Sorts `reasoning_effort_levels` by `_TIER_ORDER`.
   - Selects `default_reasoning_effort`:
     - If `base_slug in _HIGH_DEFAULT_BASE_SLUGS` and `"high"` in levels → `"high"`
     - Else if `"medium"` in levels → `"medium"`
     - Else if `"high"` in levels → `"high"`
     - Else `levels[0]` if `levels` else `None`.
   - Infers `brand`, `max_input_tokens`, `max_output_tokens`, `supports_video`:
     - `"gemini"` in `base_slug` → `("google", 1048576, 65536, True)`
     - `"claude"` in `base_slug` → `("anthropic", 200000, 65536, False)`
     - `"gpt"` in `base_slug` → `("openai", 131072, 32768, False)`
     - else → `("antigravity", 131072, 32768, False)`
   - Assigns `litellm_model_name = KNOWN_AGY_LITELLM_NAMES.get(base_slug, f"agy-{base_slug}")`.
   - If `include_default_alias` is `True`, prepends an `"agy"` `AgyBaseModelSpec` cloned from the family matching `settings.agy_default_model` (falling back to `"agy-flash"` or first spec if not found) with `litellm_model_name="agy"`.

6. Update `normalize_agy_model(model: str | None, explicit_effort: str | None = None) -> str`:
   - Helper `_pick_tier_display_name(spec: AgyBaseModelSpec, desired_effort: str | None) -> str`:
     - If `desired_effort` in `spec.tier_display_names`: return it.
     - If `desired_effort` is `"max"` or `"xhigh"`, prefer `"xhigh"` → `"high"` → `"medium"` → `"low"`.
     - If `spec.default_reasoning_effort` in `spec.tier_display_names`: return it.
     - Fallback to first value in `spec.tier_display_names` or `spec.display_name`.
   - Resolve `resolved_effort = resolve_agy_effort(raw, explicit_effort)` (Note: extend `resolve_agy_effort` only for explicit effort or suffixed models — keep existing behavior for `resolve_agy_effort(model, explicit_effort)` intact so `test_agy_streaming.py` keeps passing!).
   - Match `m` against catalog specs:
     - Exact match on `spec.litellm_model_name.lower()`, `spec.base_slug.lower()`, or `f"agy-{spec.base_slug.lower()}"` → return `_pick_tier_display_name(spec, resolved_effort)`.
     - Exact match on `discovered` keys or values: if `explicit_effort` is given, find the parent `spec` for that `model_id` and return `_pick_tier_display_name(spec, resolved_effort)`; else return `display_name`.
     - If `m in ["agy", "default", "auto", "agy-gemini"]`: if `explicit_effort`, resolve default spec with `resolved_effort`, else return `settings.agy_default_model`.
   - Keep existing family fallback branches (`neon`, `carbon`, `hydrogen`, `argon`, `opus`, `sonnet`, `oss`, `flash`) honoring `resolved_effort` when provided.

7. Update `AgyExecutor.execute()` (line 721) and `AgyExecutor.execute_stream()` (line 889):
   - Change `model = normalize_agy_model(request.model)` → `model = normalize_agy_model(request.model, explicit_effort=request.effort)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_agy_catalog_and_normalization.py tests/test_agy_streaming.py tests/test_agy_sessions.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli_agent_dispatch/core/models.py src/cli_agent_dispatch/executors/agy.py tests/test_agy_catalog_and_normalization.py
git commit -m "feat(agy): add dynamic base-model catalog and effort-aware normalization"
```

---

### Task 2: Clean `/v1/models` Base Surface in OpenAI & T3 Routes

**Files:**
- Modify: `src/cli_agent_dispatch/api/openai_routes.py:24-143`
- Modify: `src/cli_agent_dispatch/executors/t3.py:36-89`
- Modify: `tests/test_openai_bridge.py:24-63`

**Interfaces:**
- Consumes: `build_agy_model_catalog`, `discover_agy_models` from `cli_agent_dispatch.executors.agy`; `ModelCard` from `cli_agent_dispatch.core.models`
- Produces: Clean base-model `/v1/models` response with `reasoning_effort_levels` and `default_reasoning_effort` populated on AGY `ModelCard`s; zero tier-suffixed `agy-*-low/medium/high/xhigh/max` or `t3-agy-*-low/medium/high/xhigh` entries in `/v1/models`.

- [ ] **Step 1: Update `tests/test_openai_bridge.py` with failing assertions for clean base models**

Update `test_list_models` and `test_resolve_executor_and_model_argon` in `tests/test_openai_bridge.py`:

```python
@pytest.mark.asyncio
async def test_list_models(async_client):
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    cards_by_id = {m["id"]: m for m in data["data"]}
    assert "opencode" in cards_by_id
    assert "agy" in cards_by_id
    assert "agy-hydrogen" in cards_by_id
    assert "agy-neon" in cards_by_id
    assert "agy-carbon" in cards_by_id
    assert "agy-argon" in cards_by_id
    assert "agy-opus" in cards_by_id
    assert "agy-sonnet" in cards_by_id
    assert "agy-flash" in cards_by_id
    assert cards_by_id["agy-hydrogen"]["reasoning_effort_levels"] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert cards_by_id["agy-hydrogen"]["default_reasoning_effort"] == "medium"
    # Tier-suffixed clutter must not be advertised in /v1/models
    assert "agy-argon-high" not in cards_by_id
    assert "agy-argon-low" not in cards_by_id
    assert "agy-hydrogen-high" not in cards_by_id
    assert "t3-agy-argon-high" not in cards_by_id


def test_resolve_executor_and_model_argon():
    from cli_agent_dispatch.api.openai_routes import _resolve_executor_and_model
    from cli_agent_dispatch.core.models import ExecutorType

    assert _resolve_executor_and_model("agy-argon") == (
        ExecutorType.AGY,
        "agy-argon",
    )
    assert _resolve_executor_and_model("agy-gemini-5-titanium") == (
        ExecutorType.AGY,
        "agy-gemini-5-titanium",
    )
    assert _resolve_executor_and_model("openai/agy-hydrogen") == (
        ExecutorType.AGY,
        "agy-hydrogen",
    )
    assert _resolve_executor_and_model("opencode") == (
        ExecutorType.OPENCODE,
        "opencode-default",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_bridge.py::test_list_models -v`
Expected: FAIL (`assert 'agy-argon-high' not in cards_by_id` and missing `reasoning_effort_levels`)

- [ ] **Step 3: Clean `STATIC_MODEL_ALIASES`, `_build_available_models()`, and `T3_STATIC_ALIASES`**

1. In `src/cli_agent_dispatch/executors/t3.py`, remove tier-suffixed entries (`-low`, `-medium`, `-high`, `-xhigh`) from `T3_STATIC_ALIASES`, keeping all base T3 aliases (`t3`, `t3-vertex`, `t3-vertex-flash`, `t3-vertex-flash-lite`, `t3-openrouter`, `t3-openrouter-deepseek`, `t3-deepseek`, `t3-deepseek-flash`, `t3-deepseek-pro`, `t3-glm`, `t3-glm-flash`, `t3-qwen`, `t3-qwen-flash`, `t3-qwen-max`, `t3-agy`, `t3-agy-flash`, `t3-agy-argon`, `t3-agy-neon`, `t3-agy-carbon`, `t3-agy-hydrogen`, `t3-agy-opus`, `t3-agy-sonnet`, `t3-opencode`, `t3-opencode-flash`, `t3-opencode-flash-lite`, `t3-opencode-argon`, `t3-opencode-opus`, `t3-opencode-sonnet`, `t3-opencode-glm`, `t3-opencode-glm-flash`, `t3-opencode-deepseek-flash`, `t3-opencode-deepseek-pro`, `t3-opencode-qwen-flash`, `t3-opencode-qwen-max`).
2. In `src/cli_agent_dispatch/api/openai_routes.py`:
   - Import `build_agy_model_catalog` from `cli_agent_dispatch.executors.agy`.
   - Slim `STATIC_MODEL_ALIASES` to base OpenCode + T3 aliases (`opencode`, `opencode-default`, `opencode-small`, `*T3_STATIC_ALIASES`).
   - In `_build_available_models()`:
     - Emit OpenCode aliases (`opencode`, `opencode-default`, `opencode-small`).
     - Emit every `spec` from `build_agy_model_catalog()` as `ModelCard(id=spec.litellm_model_name, reasoning_effort_levels=spec.reasoning_effort_levels or None, default_reasoning_effort=spec.default_reasoning_effort)`.
     - Emit T3 models from `discover_t3_models()`.
   - Set `response_model_exclude_none=True` on `@router.get("/models", response_model=ModelListResponse, response_model_exclude_none=True)`.
   - In `_resolve_executor_and_model()`: also check `base_slug`s from `build_agy_model_catalog(include_default_alias=False)` so any dynamically discovered base slug (e.g. `gemini-5-titanium` or `agy-gemini-5-titanium`) routes to `ExecutorType.AGY`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_bridge.py tests/test_t3_executor.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cli_agent_dispatch/api/openai_routes.py src/cli_agent_dispatch/executors/t3.py tests/test_openai_bridge.py
git commit -m "feat(api): expose clean base models with reasoning_effort_levels on /v1/models"
```

---

### Task 3: LiteLLM Live Sync Service (`LiteLLMSyncService`)

**Files:**
- Modify: `src/cli_agent_dispatch/config.py:45-58`
- Create: `src/cli_agent_dispatch/core/litellm_sync.py`
- Create: `tests/test_litellm_sync.py`

**Interfaces:**
- Consumes:
  - `AgyBaseModelSpec`, `LiteLLMSyncResult` from `cli_agent_dispatch.core.models`
  - `discover_agy_models_with_status`, `build_agy_model_catalog` from `cli_agent_dispatch.executors.agy`
  - `settings` from `cli_agent_dispatch.config`
- Produces:
  - `MANAGED_BY_TAG = "cli-agent-dispatch"`
  - `resolve_litellm_master_key() -> str | None`
  - `build_litellm_deployment_payload(spec: AgyBaseModelSpec, api_base: str) -> dict[str, Any]`
  - `LiteLLMSyncService` class with `sync_once(force_refresh: bool = False) -> LiteLLMSyncResult` and `status_snapshot() -> dict[str, Any]`
  - Singleton `litellm_sync_service = LiteLLMSyncService()`

- [ ] **Step 1: Write the failing unit tests in `tests/test_litellm_sync.py`**

Create `tests/test_litellm_sync.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cli_agent_dispatch.core.litellm_sync import (
    MANAGED_BY_TAG,
    LiteLLMSyncService,
    build_litellm_deployment_payload,
    resolve_litellm_master_key,
)
from cli_agent_dispatch.executors.agy import build_agy_model_catalog

LIVE_MODELS = {
    "gemini-4-hydrogen-low": "Gemini 4 Hydrogen (Low)",
    "gemini-4-hydrogen-medium": "Gemini 4 Hydrogen (Medium)",
    "gemini-4-hydrogen-high": "Gemini 4 Hydrogen (High)",
    "gemini-4-hydrogen-xhigh": "Gemini 4 Hydrogen (XHigh)",
    "gemini-3.8-flash-low": "Gemini 3.8 Flash (Low)",
    "gemini-3.8-flash-high": "Gemini 3.8 Flash (High)",
}


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or "")
    return resp


def test_resolve_litellm_master_key_prefers_setting_then_env_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('OTHER=1\nLITELLM_MASTER_KEY="sk-from-file"\n', encoding="utf-8")

    monkeypatch.setattr("cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", None)
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_env_file", str(env_file)
    )
    assert resolve_litellm_master_key() == "sk-from-file"

    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", "sk-explicit"
    )
    assert resolve_litellm_master_key() == "sk-explicit"


def test_build_litellm_deployment_payload_exposes_reasoning_metadata():
    catalog = build_agy_model_catalog(discovered=LIVE_MODELS, include_default_alias=False)
    hydrogen = next(s for s in catalog if s.litellm_model_name == "agy-hydrogen")
    payload = build_litellm_deployment_payload(hydrogen, api_base="http://10.0.0.10:8032/v1")

    assert payload["model_name"] == "agy-hydrogen"
    assert payload["litellm_params"] == {
        "model": "openai/agy-hydrogen",
        "api_base": "http://10.0.0.10:8032/v1",
        "api_key": "not-needed",
    }
    info = payload["model_info"]
    assert info["id"] == "cad-sync::agy-hydrogen"
    assert info["managed_by"] == MANAGED_BY_TAG
    assert info["supports_reasoning"] is True
    assert info["reasoning_effort_levels"] == ["low", "medium", "high", "xhigh"]
    assert info["default_reasoning_effort"] == "medium"
    assert info["supports_low_reasoning_effort"] is True
    assert info["supports_xhigh_reasoning_effort"] is True
    assert info["supports_max_reasoning_effort"] is False


@pytest.mark.asyncio
async def test_sync_creates_updates_prunes_and_preserves_non_managed(monkeypatch):
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", "sk-test"
    )
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_base_url", "http://litellm:4000"
    )
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.cad_public_api_base",
        "http://10.0.0.10:8032/v1",
    )

    existing_info = {
        "data": [
            # Managed & outdated (missing xhigh) -> should UPDATE
            {
                "model_name": "agy-hydrogen",
                "litellm_params": {
                    "model": "openai/agy-hydrogen",
                    "api_base": "http://10.0.0.10:8032/v1",
                },
                "model_info": {
                    "id": "cad-sync::agy-hydrogen",
                    "db_model": True,
                    "managed_by": MANAGED_BY_TAG,
                    "source_base_slug": "gemini-4-hydrogen",
                    "max_input_tokens": 1048576,
                    "max_output_tokens": 65536,
                    "supports_reasoning": True,
                    "reasoning_effort_levels": ["low", "medium", "high"],
                    "default_reasoning_effort": "medium",
                    "supports_low_reasoning_effort": True,
                    "supports_xhigh_reasoning_effort": False,
                    "supports_max_reasoning_effort": False,
                    "brand": "google",
                },
            },
            # Managed & retired -> should DELETE (because from_live_cli=True)
            {
                "model_name": "agy-retired",
                "litellm_params": {"model": "openai/agy-retired"},
                "model_info": {
                    "id": "cad-sync::agy-retired",
                    "db_model": True,
                    "managed_by": MANAGED_BY_TAG,
                },
            },
            # Unmanaged DB model -> MUST NOT TOUCH
            {
                "model_name": "local-gemma4",
                "litellm_params": {"model": "ollama_chat/gemma4"},
                "model_info": {"id": "ui-123", "db_model": True},
            },
            # Static YAML conflict for agy-flash -> MUST SKIP
            {
                "model_name": "agy-flash",
                "litellm_params": {"model": "openai/agy-flash"},
                "model_info": {"id": "yaml-flash", "db_model": False},
            },
        ]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(200, existing_info))
    mock_client.post = AsyncMock(return_value=_mock_response(200, {"status": "ok"}))
    mock_client.patch = AsyncMock(return_value=_mock_response(200, {"status": "ok"}))

    service = LiteLLMSyncService()
    with (
        patch(
            "cli_agent_dispatch.core.litellm_sync.discover_agy_models_with_status",
            return_value=(LIVE_MODELS, True),
        ),
        patch("httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        result = await service.sync_once(force_refresh=True)

    assert result.status == "success"
    assert result.from_live_cli is True
    assert "agy" in result.created
    assert result.updated == ["agy-hydrogen"]
    assert result.deleted == ["agy-retired"]
    assert result.skipped_conflicts == ["agy-flash"]
    assert result.errors == []

    # Verify non-managed DB model was never deleted
    delete_calls = [
        call for call in mock_client.post.call_args_list if "/model/delete" in str(call.args[0])
    ]
    assert len(delete_calls) == 1
    assert delete_calls[0].kwargs["json"] == {"id": "cad-sync::agy-retired"}


@pytest.mark.asyncio
async def test_sync_refuses_to_prune_when_discovery_fell_back(monkeypatch):
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", "sk-test"
    )
    existing_info = {
        "data": [
            {
                "model_name": "agy-retired",
                "litellm_params": {"model": "openai/agy-retired"},
                "model_info": {
                    "id": "cad-sync::agy-retired",
                    "db_model": True,
                    "managed_by": MANAGED_BY_TAG,
                },
            }
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(200, existing_info))
    mock_client.post = AsyncMock(return_value=_mock_response(200, {"status": "ok"}))

    service = LiteLLMSyncService()
    with (
        patch(
            "cli_agent_dispatch.core.litellm_sync.discover_agy_models_with_status",
            return_value=(LIVE_MODELS, False),
        ),
        patch("httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        result = await service.sync_once()

    assert result.from_live_cli is False
    assert result.deleted == []


@pytest.mark.asyncio
async def test_sync_handles_missing_key_and_http_errors_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr("cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", None)
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_env_file",
        str(tmp_path / "missing.env"),
    )
    service = LiteLLMSyncService()
    res_no_key = await service.sync_once()
    assert res_no_key.status == "skipped"

    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", "sk-test"
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    with (
        patch(
            "cli_agent_dispatch.core.litellm_sync.discover_agy_models_with_status",
            return_value=(LIVE_MODELS, True),
        ),
        patch("httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        res_err = await service.sync_once()

    assert res_err.status == "error"
    assert res_err.errors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cli_agent_dispatch.core.litellm_sync'`

- [ ] **Step 3: Add LiteLLM sync settings to `src/cli_agent_dispatch/config.py`**

Add fields to `Settings` in `src/cli_agent_dispatch/config.py`:

```python
# LiteLLM Live Model Sync Configuration
litellm_sync_enabled: bool = Field(default=True, alias="LITELLM_SYNC_ENABLED")
litellm_base_url: str = Field(default="http://10.0.0.10:8448", alias="LITELLM_BASE_URL")
litellm_master_key: str | None = Field(default=None, alias="LITELLM_MASTER_KEY")
litellm_env_file: str = Field(default="/containers/ai/litellm/.env", alias="LITELLM_ENV_FILE")
litellm_sync_interval_seconds: int = Field(default=3600, alias="LITELLM_SYNC_INTERVAL_SECONDS")
cad_public_api_base: str = Field(default="http://10.0.0.10:8032/v1", alias="CAD_PUBLIC_API_BASE")
```

- [ ] **Step 4: Implement `src/cli_agent_dispatch/core/litellm_sync.py`**

Create `src/cli_agent_dispatch/core/litellm_sync.py` with:
- `MANAGED_BY_TAG = "cli-agent-dispatch"`
- `resolve_litellm_master_key()` (reads `settings.litellm_master_key`, else parses `LITELLM_MASTER_KEY=...` from `settings.litellm_env_file` without ever logging its value).
- `build_litellm_deployment_payload(spec: AgyBaseModelSpec, api_base: str) -> dict[str, Any]` populating all native LiteLLM `model_info` reasoning fields (`supports_reasoning`, `reasoning_effort_levels`, `default_reasoning_effort`, `supports_low_reasoning_effort`, `supports_xhigh_reasoning_effort`, `supports_max_reasoning_effort`, `max_input_tokens`, `max_output_tokens`, `brand`, `pricing_tier`, `capabilities`, `managed_by`, `source_base_slug`, `id`).
- `_deployment_needs_update(existing_entry: dict[str, Any], desired_payload: dict[str, Any]) -> bool` comparing `litellm_params.model`, `litellm_params.api_base`, and the managed `model_info` metadata keys.
- `LiteLLMSyncService`:
  - `self._lock = asyncio.Lock()`
  - `self.last_result: LiteLLMSyncResult | None = None`
  - `async def sync_once(self, force_refresh: bool = False) -> LiteLLMSyncResult`
  - Update fallback: tries `PATCH /model/{existing_id}/update`; if status >= 400, deletes via `POST /model/delete` and recreates via `POST /model/new`.
  - `def status_snapshot(self) -> dict[str, Any]` returning `{enabled, base_url, interval_seconds, last_result}`.
- `litellm_sync_service = LiteLLMSyncService()`

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm_sync.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli_agent_dispatch/config.py src/cli_agent_dispatch/core/litellm_sync.py tests/test_litellm_sync.py
git commit -m "feat(litellm): add live Admin API sync service for AGY base models"
```

---

### Task 4: FastAPI Lifespan Loop, `POST /api/litellm/sync`, and `/api/diagnostics`

**Files:**
- Modify: `src/cli_agent_dispatch/main.py:50-128`
- Modify: `tests/test_litellm_sync.py` (append API & diagnostics tests)

**Interfaces:**
- Consumes: `litellm_sync_service`, `LiteLLMSyncResult`
- Produces:
  - `litellm_sync_loop(interval_seconds: float | None = None)` background task started in `lifespan`
  - `POST /api/litellm/sync` (query param `force_refresh: bool = True`, returns `LiteLLMSyncResult`)
  - `GET /api/diagnostics` enriched with `"litellm_sync": litellm_sync_service.status_snapshot()`

- [ ] **Step 1: Add failing API tests to `tests/test_litellm_sync.py`**

Append to `tests/test_litellm_sync.py`:

```python
@pytest.mark.asyncio
async def test_api_litellm_sync_endpoint_and_diagnostics(async_client):
    from cli_agent_dispatch.core.models import LiteLLMSyncResult

    fake_res = LiteLLMSyncResult(
        status="success",
        synced_at="2026-11-23T23:30:00Z",
        from_live_cli=True,
        created=["agy-hydrogen"],
        unchanged=["agy"],
    )
    with patch(
        "cli_agent_dispatch.main.litellm_sync_service.sync_once",
        new_callable=AsyncMock,
        return_value=fake_res,
    ) as mock_sync:
        resp = await async_client.post("/api/litellm/sync?force_refresh=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["created"] == ["agy-hydrogen"]
        mock_sync.assert_awaited_once_with(force_refresh=True)

    diag = await async_client.get("/api/diagnostics")
    assert diag.status_code == 200
    assert "litellm_sync" in diag.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm_sync.py::test_api_litellm_sync_endpoint_and_diagnostics -v`
Expected: FAIL with `404 Not Found` on `/api/litellm/sync`

- [ ] **Step 3: Wire `litellm_sync_loop`, `POST /api/litellm/sync`, and `/api/diagnostics` in `src/cli_agent_dispatch/main.py`**

In `src/cli_agent_dispatch/main.py`:
1. Import `litellm_sync_service` from `cli_agent_dispatch.core.litellm_sync` and `LiteLLMSyncResult` from `cli_agent_dispatch.core.models`.
2. Add:

```python
async def litellm_sync_loop(interval_seconds: float | None = None) -> None:
    """Background task syncing discovered AGY base models into LiteLLM on startup and periodically."""
    interval = (
        interval_seconds
        if interval_seconds is not None
        else max(60, settings.litellm_sync_interval_seconds)
    )
    if not settings.litellm_sync_enabled:
        logger.info("LiteLLM auto-sync disabled via LITELLM_SYNC_ENABLED=false.")
        return
    while True:
        try:
            await litellm_sync_service.sync_once(force_refresh=True)
        except asyncio.CancelledError:
            logger.info("LiteLLM sync task cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in LiteLLM sync loop: {e}", exc_info=True)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("LiteLLM sync task cancelled.")
            break
```

3. Start `sync_task = asyncio.create_task(litellm_sync_loop())` in `lifespan` and cancel/await it cleanly in `finally`.
4. Add route:

```python
@app.post("/api/litellm/sync", response_model=LiteLLMSyncResult, tags=["LiteLLM Sync"])
async def trigger_litellm_sync(force_refresh: bool = True) -> LiteLLMSyncResult:
    """Trigger an immediate AGY base-model sync into LiteLLM."""
    return await litellm_sync_service.sync_once(force_refresh=force_refresh)
```

5. Update `GET /api/diagnostics` to return `{"events": diagnostic_buffer.get_events(limit=limit), "litellm_sync": litellm_sync_service.status_snapshot()}`.

- [ ] **Step 4: Run full `CLIAgentDispatch` test suite & linter**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
Expected: Zero ruff errors, all tests pass, coverage $\ge 80\%$.

- [ ] **Step 5: Commit**

```bash
git add src/cli_agent_dispatch/main.py tests/test_litellm_sync.py
git commit -m "feat(api): wire LiteLLM auto-sync lifespan loop, sync endpoint, and diagnostics"
```

---

### Task 5: Downstream Caller & Config Cleanup (LiteLLM + OpenClaw) & End-to-End Verification

**Files (Host Configs):**
- Modify: `/containers/ai/litellm/config/config.yaml` — Remove all 13 manual `agy*` entries (lines 2–378); keep `router_settings.fallbacks` intact.
- Modify: `/containers/ai/litellm/config/router_rules.yaml` — Remove stale suffixed `agy-*-high/low/medium/xhigh/max` and `t3-*` entries (lines 79–113); keep clean base rewrites (`openai/agy -> agy`, `openai/gemini-4-hydrogen -> agy-hydrogen`, `openai/gemini-4-neon -> agy-neon`, `openai/gemini-4-carbon -> agy-carbon`, `openai/gemini-4-argon -> agy-argon`, `openai/gemini-3.8-flash -> agy-flash`, `openai/claude-opus-5-5 -> agy-opus`, `openai/claude-sonnet-5-5 -> agy-sonnet`).
- Modify: `/containers/ai/openclaw/config/openclaw.json` — Remove `litellm/agy-argon-high`, `litellm/agy-argon-low`, `litellm/agy-opus-medium`, `litellm/agy-sonnet-medium`; add `litellm/agy-hydrogen`, `litellm/agy-neon`, `litellm/agy-carbon`; update `agy-argon` display name to `AGY Gemini 4 Argon`.
- Modify: `/containers/ai/openclaw/config/system_prompt.md` — Update `openclaw-model set` alias list to clean base aliases.
- Modify: `/containers/ai/openclaw/data/workspace/skills/model-switcher/SKILL.md` — Update alias examples & table to clean base aliases.
- Modify: `/containers/ai/openclaw/data/models.json` — Remove stale suffixed `agy-*-<tier>` and `t3-*-<tier>` entries.

- [ ] **Step 1: Clean LiteLLM YAML configs**
  - Edit `/containers/ai/litellm/config/config.yaml` and `/containers/ai/litellm/config/router_rules.yaml`.
  - Validate YAML parses cleanly with `python3 -c "import yaml; yaml.safe_load(open('/containers/ai/litellm/config/config.yaml')); yaml.safe_load(open('/containers/ai/litellm/config/router_rules.yaml'))"`.

- [ ] **Step 2: Clean OpenClaw configs & docs**
  - Edit `/containers/ai/openclaw/config/openclaw.json`, `system_prompt.md`, `data/workspace/skills/model-switcher/SKILL.md`, and `data/models.json`.
  - Run OpenClaw validation tests: `pytest /containers/ai/openclaw/tests/test_config_validation.py`
  - Expected: PASS (including `"LiteLLM must not host T3 models"`).

- [ ] **Step 3: Restart `litellm` and verify live sync**
  - Restart `litellm`: `docker restart litellm` and wait for `http://10.0.0.10:8448/health/readiness` HTTP 200.
  - Run `LiteLLMSyncService.sync_once(force_refresh=True)` (or trigger `POST /api/litellm/sync` once deployed to host service) and verify `GET http://10.0.0.10:8448/v1/model/info` returns:
    - All AGY base models (`agy`, `agy-hydrogen`, `agy-neon`, `agy-carbon`, `agy-argon`, `agy-flash`, `agy-opus`, `agy-sonnet`, `agy-gpt-oss`, `agy-orchestrator-*`) with `db_model: true`, `managed_by: "cli-agent-dispatch"`, `supports_reasoning: true`, and accurate `reasoning_effort_levels`.
    - Zero duplicate `agy-*` entries, zero `t3*` entries in LiteLLM, and all 6 existing non-managed DB models (`dall-e-3`, `local-gemma4`, etc.) intact.

- [ ] **Step 4: Final verification of `CLIAgentDispatch` branch**
  - Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
  - Expected: All passing, ready for PR/merge.

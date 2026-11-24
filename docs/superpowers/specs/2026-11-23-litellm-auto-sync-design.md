# Design: Automatic CLIAgentDispatch → LiteLLM Base-Model Sync

**Date:** 2026-11-23
**Status:** Proposed
**Branch:** `t3/litellm-sync-latest-models`

## 1. Problem & Goals

`CLIAgentDispatch` dynamically discovers available Antigravity (`agy`) models via `agy models`, but LiteLLM's model catalog is currently maintained manually in `/containers/ai/litellm/config/config.yaml`. As a result:

1. New `agy` families (`gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-carbon`, etc.) do not appear in LiteLLM until manually added to `config.yaml` and the `litellm` container is restarted.
2. `litellm-manager` only discovers OpenRouter, Vertex AI, and local Ollama models—it preserves custom `openai/*` entries as-is and never queries `CLIAgentDispatch`.
3. Exposing every `agy` model × thinking tier (`-low`, `-medium`, `-high`, `-xhigh`, `-max`) as separate LiteLLM entries creates model-list bloat (26+ entries for ~8 actual models) even though LiteLLM's `/v1/model/info` natively supports per-model reasoning metadata (`supports_reasoning`, `reasoning_effort_levels`, `default_reasoning_effort`).

### Goals
- **Automatic zero-restart sync:** `CLIAgentDispatch` reconciles discovered `agy` base models directly into LiteLLM via LiteLLM's live Admin API (`/model/new`, `/model/{id}/update`, `/model/delete`), leveraging `STORE_MODEL_IN_DB=True` (already enabled in `/containers/ai/litellm/.env`).
- **Clean base-model surface with rich `/v1/model/info`:** Expose 1 entry per `agy` base model (`agy`, `agy-hydrogen`, `agy-neon`, `agy-carbon`, `agy-argon`, `agy-flash`, `agy-opus`, `agy-sonnet`, `agy-gpt-oss`, plus any future `agy` models automatically) with exact discovered `reasoning_effort_levels` and `default_reasoning_effort` in LiteLLM's `model_info`.
- **Zero-touch support for future `agy` models:** When `agy` ships a new model family, both LiteLLM sync and `AgyExecutor` execution (`model` + `reasoning_effort`) work automatically without code changes.
- **No backward-compatibility shims:** Remove all stale manual `agy-*` YAML entries, stale `router_rules.yaml` suffixed hijacks, and suffixed model references across downstream callers (OpenClaw) directly.
- **Respect T3/LiteLLM boundary:** Keep T3 models out of LiteLLM (per `/containers/ai/openclaw/tests/test_config_validation.py`, OpenClaw enforces `"LiteLLM must not host T3 models"` and routes T3 directly to `CLIAgentDispatch` on `:8032/v1` with session affinity headers).

### Non-Goals
- Modifying `litellm-manager` or `custom_router.py`.
- Adding legacy suffixed-model rewrite shims in LiteLLM.
- Managing OpenRouter, Vertex AI, local Ollama, or UI-added LiteLLM DB models (`dall-e-3`, `local-gemma4`, etc.).

---

## 2. Architecture Overview

```
agy CLI (`agy models`)
        │
        ▼
CLIAgentDispatch (:8032)
  ├── executors/agy.py
  │     ├── discover_agy_models_with_status()  (live vs fallback flag)
  │     ├── build_agy_model_catalog()          (group tiers -> base models)
  │     └── normalize_agy_model(model, effort) (resolve base + effort -> exact agy tier)
  │
  ├── core/litellm_sync.py
  │     └── LiteLLMSyncService
  │           ├── GET  /v1/model/info          (read existing deployments)
  │           ├── POST /model/new              (register new base models)
  │           ├── PATCH /model/{id}/update     (update changed effort levels/limits)
  │           └── POST /model/delete           (prune retired managed models)
  │
  ├── main.py (lifespan startup + periodic loop)
  └── api/ (POST /api/litellm/sync, GET /api/diagnostics, GET /v1/models)
        │
        ▼ HTTP (Authorization: Bearer <LITELLM_MASTER_KEY>)
LiteLLM Proxy (:8448, STORE_MODEL_IN_DB=True)
  ├── /v1/models       (clean base model IDs: agy, agy-hydrogen, agy-neon, ...)
  ├── /v1/model/info   (supports_reasoning + reasoning_effort_levels + default_reasoning_effort)
  └── /v1/chat/completions (forwards model + reasoning_effort -> http://10.0.0.10:8032/v1)
```

---

## 3. Component Design

### 3.1 Dynamic AGY Base-Model Catalog & Effort-Aware Normalization (`src/cli_agent_dispatch/executors/agy.py`)

#### Data Model
Add a typed `AgyBaseModelSpec` (Pydantic `BaseModel` in `core/models.py` or `executors/agy.py`):
- `base_slug: str` — canonical `agy` base slug after stripping trailing tier suffix (e.g. `gemini-4-hydrogen`, `claude-opus-5-5`, `gpt-oss-120b`, `gemini-pro-orchestrator-neon`).
- `litellm_model_name: str` — public model ID exposed in LiteLLM and `CLIAgentDispatch`:
  - Known friendly names:
    - `gemini-4-hydrogen` → `agy-hydrogen`
    - `gemini-4-neon` → `agy-neon`
    - `gemini-4-carbon` → `agy-carbon`
    - `gemini-4-argon` → `agy-argon`
    - `gemini-3.8-flash` → `agy-flash`
    - `claude-opus-5-5` → `agy-opus`
    - `claude-sonnet-5-5` → `agy-sonnet`
    - `gpt-oss-120b` → `agy-gpt-oss`
    - `gemini-pro-orchestrator-neon` → `agy-orchestrator-neon`
    - `gemini-pro-orchestrator` → `agy-orchestrator-carbon`
    - `gemini-pro-orchestrator-argon` → `agy-orchestrator-argon`
  - Any unknown future `base_slug` → `f"agy-{base_slug}"` (collision-free and deterministic).
- `display_name: str` — human-readable base display name with trailing `(Low)`/`(Medium)`/`(High)`/`(XHigh)`/`(Max)` stripped (e.g. `Gemini 4 Hydrogen`).
- `reasoning_effort_levels: list[str]` — ordered subset of `["low", "medium", "high", "xhigh", "max"]` discovered from `agy models` for this base family (empty list `[]` if the model has no tier variants, e.g. orchestrator models).
- `default_reasoning_effort: str | None` — `"high"` for Opus/Sonnet/Flash if present, else `"medium"` if present, else the first available tier, else `None`.
- `tier_display_names: dict[str, str]` — maps each tier (`"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`) to the exact `agy` CLI display name (e.g. `{"xhigh": "Gemini 4 Hydrogen (XHigh)"}`).
- `brand: str` — inferred from `base_slug` (`gemini` → `"google"`, `claude` → `"anthropic"`, `gpt` → `"openai"`, else `"antigravity"`).
- `max_input_tokens: int` / `max_output_tokens: int` — `1048576 / 65536` for Gemini, `200000 / 65536` for Claude, `131072 / 32768` for GPT-OSS/other.
- `supports_video: bool` — `True` for Gemini models, `False` otherwise.

Additionally, include the default entrypoint `agy` in the sync catalog (using `settings.agy_default_model`'s family metadata and `reasoning_effort_levels`) so LiteLLM's `router_settings.fallbacks` (`agy -> ...`) and `agy` callers continue to work.

#### Discovery Status Tracking
Add `discover_agy_models_with_status(force_refresh: bool = False) -> tuple[dict[str, str], bool]`:
- Returns `(models_dict, from_live_cli)`.
- `discover_agy_models(force_refresh=...)` remains as a thin wrapper returning just `models_dict` so existing callers do not break.
- `from_live_cli` is `True` only when `[agy_bin, "models"]` exits `0` and yields at least one parsed model. When `False` (fallback to `DEFAULT_AGY_MODELS`), the LiteLLM sync service **refuses to prune** existing managed models.

#### Effort-Aware Dynamic Normalization
Update `normalize_agy_model(model: str | None, explicit_effort: str | None = None) -> str`:
1. Strip `openai/` or `litellm/` prefixes.
2. Look up the requested name against the dynamic catalog (matching `litellm_model_name`, `base_slug`, `agy-<base_slug>`, exact discovered `model_id`, or exact `display_name`).
3. When matched to a base family:
   - Determine desired effort: `resolve_agy_effort(raw, explicit_effort)` → fallback to tier embedded in `raw` → fallback to `spec.default_reasoning_effort`.
   - If `spec.tier_display_names` has that tier, return the exact discovered display name (e.g. `agy-hydrogen` + `reasoning_effort="xhigh"` → `"Gemini 4 Hydrogen (XHigh)"`).
   - If the requested effort is not a separate tier variant (e.g. `max` on a model that only has `high`), pick the closest highest available tier and still pass `--effort` to `agy`.
   - This works identically for brand-new models discovered at runtime (e.g. `agy-gemini-5-titanium` + `reasoning_effort="high"` → `"Gemini 5 Titanium (High)"`).
4. Update both `AgyExecutor.execute()` and `AgyExecutor.execute_stream()` to call `normalize_agy_model(request.model, explicit_effort=request.effort)`.

#### Clean `/v1/models` Surface on `CLIAgentDispatch`
- Remove tier-suffixed `agy-*-low/medium/high/xhigh/max` and `t3-agy-*-low/medium/high/xhigh` entries from `STATIC_MODEL_ALIASES` ([openai_routes.py](file:///home/steve/.t3/worktrees/CLIAgentDispatch/t3-888650d5-9a560e12ca29b053/src/cli_agent_dispatch/api/openai_routes.py#L33-L64)) and `T3_STATIC_ALIASES` ([t3.py](file:///home/steve/.t3/worktrees/CLIAgentDispatch/t3-888650d5-9a560e12ca29b053/src/cli_agent_dispatch/executors/t3.py#L36-L88)).
- Build AGY entries in `_build_available_models()` directly from `build_agy_model_catalog()`, adding optional fields `reasoning_effort_levels: list[str] | None = None` and `default_reasoning_effort: str | None = None` to `ModelCard` (excluded when `None` to remain clean OpenAI-compatible JSON).
- Ensure `_resolve_executor_and_model()` routes any `agy-*` or discovered `base_slug` to `ExecutorType.AGY`.

---

### 3.2 Live LiteLLM Sync Service (`src/cli_agent_dispatch/core/litellm_sync.py`)

#### Configuration (`src/cli_agent_dispatch/config.py`)
Add settings with sensible host defaults:
- `litellm_sync_enabled: bool = Field(default=True, alias="LITELLM_SYNC_ENABLED")`
- `litellm_base_url: str = Field(default="http://10.0.0.10:8448", alias="LITELLM_BASE_URL")`
- `litellm_master_key: str | None = Field(default=None, alias="LITELLM_MASTER_KEY")`
- `litellm_env_file: str = Field(default="/containers/ai/litellm/.env", alias="LITELLM_ENV_FILE")`
- `litellm_sync_interval_seconds: int = Field(default=3600, alias="LITELLM_SYNC_INTERVAL_SECONDS")`
- `cad_public_api_base: str = Field(default="http://10.0.0.10:8032/v1", alias="CAD_PUBLIC_API_BASE")`

Key resolution helper: use `settings.litellm_master_key` if set; otherwise parse `LITELLM_MASTER_KEY=...` from `settings.litellm_env_file` if readable. If neither is available, log a warning and skip sync without erroring.

#### Ownership Tagging & Deployment Payload
Every synced deployment sent to `POST /model/new` (and compared/updated via `PATCH /model/{model_id}/update`):

```json
{
  "model_name": "agy-hydrogen",
  "litellm_params": {
    "model": "openai/agy-hydrogen",
    "api_base": "http://10.0.0.10:8032/v1",
    "api_key": "not-needed"
  },
  "model_info": {
    "id": "cad-sync::agy-hydrogen",
    "managed_by": "cli-agent-dispatch",
    "source_base_slug": "gemini-4-hydrogen",
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "max_input_tokens": 1048576,
    "max_output_tokens": 65536,
    "supports_reasoning": true,
    "reasoning_effort_levels": ["low", "medium", "high", "xhigh"],
    "default_reasoning_effort": "medium",
    "supports_low_reasoning_effort": true,
    "supports_xhigh_reasoning_effort": true,
    "supports_max_reasoning_effort": false,
    "brand": "google",
    "pricing_tier": "free",
    "capabilities": {
      "text_in": true,
      "text_out": true,
      "image_in": true,
      "image_out": false,
      "audio_in": false,
      "audio_out": false,
      "video_in": true,
      "video_out": false,
      "pdf_in": true,
      "function_calling": true,
      "streaming": true
    }
  }
}
```

#### Reconciliation Algorithm (`LiteLLMSyncService.sync_once(force_refresh: bool = False)`)
1. Guard with an `asyncio.Lock` so concurrent manual + periodic syncs never race.
2. Run `discover_agy_models_with_status(force_refresh=force_refresh)` in `asyncio.to_thread` and build desired payloads via `build_agy_model_catalog()`.
3. Call `GET {litellm_base_url}/v1/model/info` (`Authorization: Bearer <key>`, timeout `15s`).
4. Partition existing LiteLLM entries:
   - **`managed_by_name`**: entries where `model_info.get("managed_by") == "cli-agent-dispatch"` and `model_info.get("db_model") is True` (keyed by `model_name`, retaining LiteLLM's `model_info.id`).
   - **`static_yaml_names`**: `model_name`s where `model_info.get("db_model") is False`.
   - All other DB models (e.g. `dall-e-3`, `local-gemma4`) are ignored and never modified or deleted.
5. Reconcile desired specs:
   - If `model_name in static_yaml_names`: skip and append to `skipped_conflicts` (log a warning so static YAML duplicates are never created).
   - If `model_name not in managed_by_name`: `POST /model/new`.
   - If `model_name in managed_by_name` and managed fields differ (`litellm_params.model`, `api_base`, `source_base_slug`, `max_input_tokens`, `max_output_tokens`, `supports_reasoning`, `reasoning_effort_levels`, `default_reasoning_effort`, `supports_*_reasoning_effort`, `brand`): `PATCH /model/{existing_id}/update` (if `PATCH` returns 4xx/5xx, fall back to `POST /model/delete` + `POST /model/new`).
   - Otherwise count as `unchanged`.
6. Prune retired models **only if `from_live_cli is True` and desired catalog is non-empty**:
   - For any `model_name` in `managed_by_name` not in desired specs: `POST /model/delete` with `{"id": existing_id}`.
7. Store structured `LiteLLMSyncResult` (`status`, `synced_at`, `from_live_cli`, `created`, `updated`, `deleted`, `unchanged`, `skipped_conflicts`, `errors`) on the service instance for `/api/diagnostics`.

#### Lifecycle & API Integration
- **`src/cli_agent_dispatch/main.py`:** Start `litellm_sync_loop()` as a background task in `lifespan` (runs an initial sync on startup after a short non-blocking yield, then sleeps `settings.litellm_sync_interval_seconds`). Cancel cleanly on shutdown.
- **`POST /api/litellm/sync`** (query param `force_refresh: bool = True`): Triggers `litellm_sync_service.sync_once(force_refresh=force_refresh)` and returns `LiteLLMSyncResult`. Never returns an unhandled 500; network/auth failures return a structured `LiteLLMSyncResult(status="error", errors=[...])`.
- **`GET /api/diagnostics`**: Includes `"litellm_sync": litellm_sync_service.status_snapshot()`.

---

## 4. Downstream Caller & Config Cleanup (Zero Backcompat)

All existing references to suffixed `agy-*-<tier>` models and LiteLLM-hosted `t3-*` models were audited across `/containers/` and `~/Repos/`. Every caller will be updated directly:

1. **`/containers/ai/litellm/config/config.yaml`**
   - Remove all 13 manual `agy*` entries (`agy`, `agy-argon`, `agy-argon-high`, `agy-argon-xhigh`, `agy-argon-max`, `agy-argon-low`, `agy-opus`, `agy-opus-xhigh`, `agy-opus-max`, `agy-opus-medium`, `agy-sonnet`, `agy-sonnet-medium`, `agy-flash`).
   - Keep `router_settings.fallbacks` (`agy -> ...`) — satisfied by the live-synced `agy` deployment.
   - Restart `litellm` once after cleanup so LiteLLM drops the static YAML entries and uses the live DB deployments exclusively.

2. **`/containers/ai/litellm/config/router_rules.yaml`**
   - Remove all stale suffixed `agy-*-high/low/medium/xhigh/max` and `t3-*` entries from `exact_hijacks` (lines 79–113).
   - Keep clean base rewrites only where useful (e.g. `openai/agy: agy`, `openai/gemini-4-argon: agy-argon`, `openai/gemini-4-hydrogen: agy-hydrogen`, `openai/gemini-4-neon: agy-neon`, `openai/gemini-4-carbon: agy-carbon`, `openai/gemini-3.8-flash: agy-flash`, `openai/claude-opus-5-5: agy-opus`, `openai/claude-sonnet-5-5: agy-sonnet`) and keep `vertex_budget_guard.spillover_map` pointing at `agy-argon`.
   - Leave `custom_router.py` untouched.

3. **`/containers/ai/openclaw/config/openclaw.json`**
   - Remove `litellm/agy-argon-high`, `litellm/agy-argon-low`, `litellm/agy-opus-medium`, `litellm/agy-sonnet-medium` from both `agents.defaults.models` and `models.providers.litellm.models`.
   - Add base models `litellm/agy-hydrogen`, `litellm/agy-neon`, `litellm/agy-carbon` under `models.providers.litellm.models` and `agents.defaults.models` (with aliases `litellm-hydrogen`, `neon`, `carbon` — keeping `hydrogen` pointing to `cli-agent-dispatch/agy-hydrogen` and all `cli-agent-dispatch/t3*` entries intact).
   - Update `agy-argon` display name from `"AGY Argon (Gemini 3.8 Pro Thinking)"` to `"AGY Gemini 4 Argon"`.

4. **`/containers/ai/openclaw/config/system_prompt.md` & `/containers/ai/openclaw/data/workspace/skills/model-switcher/SKILL.md`**
   - Remove deprecated suffixed aliases (`argon-high`, `argon-low`, `opus-medium`, `sonnet-medium`, `t3-argon-high`, `t3-argon-low`).
   - Document the clean base aliases (`hydrogen`, `neon`, `carbon`, `argon`, `opus`, `sonnet`, `agy`, `agy-flash`, plus `t3`, `t3-agy`, `t3-vertex`, `t3-openrouter`) and reasoning effort selection.

5. **`/containers/ai/openclaw/data/models.json`**
   - Remove stale suffixed `agy-*-<tier>` and `t3-*-<tier>` entries so `openclaw-model` matches the clean base catalog.

---

## 5. Error Handling & Safety

- **Transient `agy` failures:** If `agy models` fails or times out, `from_live_cli=False` prevents pruning any existing models in LiteLLM.
- **LiteLLM unavailable / auth error:** Sync logs a warning, updates `litellm_sync_service.last_result` with `status="error"`, and retries on the next interval without crashing `CLIAgentDispatch` or blocking FastAPI startup.
- **Non-destructive ownership:** Only deployments with `model_info.managed_by == "cli-agent-dispatch"` and `model_info.db_model is True` are ever updated or deleted.
- **Static YAML collision guard:** If a deployment with the same `model_name` already exists in `config.yaml` (`db_model: false`), sync skips creating a duplicate DB deployment and reports it in `skipped_conflicts`.
- **Subprocess & secret safety:** Continues using sanitized argument arrays (`[agy_bin, "models"]`), never logs `LITELLM_MASTER_KEY`, and never uses `shell=True`.

---

## 6. Verification & Testing Strategy

1. **Unit tests (`tests/test_agy_catalog_and_normalization.py`):**
   - `build_agy_model_catalog()` grouping tiers, preserving exact `reasoning_effort_levels` order (`low`, `medium`, `high`, `xhigh`, `max`), handling orchestrator models without tiers, and deriving `agy-<base_slug>` for unknown future models.
   - `normalize_agy_model(model, explicit_effort=...)` resolving base model + `reasoning_effort` to the exact discovered tier display name for both known and dynamically discovered future models.
   - `discover_agy_models_with_status()` returning `from_live_cli=True` on live output and `False` on subprocess error/empty output.
2. **Sync service tests (`tests/test_litellm_sync.py`):**
   - Creating missing models with complete `model_info` (`supports_reasoning`, `reasoning_effort_levels`, `default_reasoning_effort`, `managed_by`).
   - Updating models when `reasoning_effort_levels` or limits change (including `PATCH` failure fallback to delete + recreate).
   - Pruning retired `managed_by="cli-agent-dispatch"` models when `from_live_cli=True`, and **not** pruning when `from_live_cli=False`.
   - Leaving non-managed DB models (`dall-e-3`, `local-gemma4`) untouched.
   - Skipping `db_model=False` collisions cleanly.
   - Graceful handling of missing master key, HTTP 401/500, and timeouts.
   - `POST /api/litellm/sync` and `GET /api/diagnostics` API tests.
3. **Repo verification:**
   - `uv run ruff check .` and `uv run ruff format --check .`
   - `uv run pytest` ($\ge 80\%$ coverage)
   - `pytest` in `/containers/ai/openclaw/tests/` for OpenClaw config validation.
4. **Live smoke test (post-deploy):**
   - Restart `CLIAgentDispatch` and `litellm`, call `POST http://10.0.0.10:8032/api/litellm/sync`, confirm `GET http://10.0.0.10:8448/v1/model/info` exposes every `agy*` base model with accurate `reasoning_effort_levels` and zero duplicates.

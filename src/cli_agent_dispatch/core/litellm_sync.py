import asyncio
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import AgyBaseModelSpec, LiteLLMSyncResult
from cli_agent_dispatch.executors.agy import (
    build_agy_model_catalog,
    discover_agy_models_with_status,
)

logger = logging.getLogger("CLIAgentDispatch.LiteLLMSync")

MANAGED_BY_TAG = "cli-agent-dispatch"
_MASTER_KEY_LINE_RE = re.compile(r"^\s*LITELLM_MASTER_KEY\s*=\s*(.+?)\s*$")
_COMPARED_INFO_KEYS: tuple[str, ...] = (
    "source_base_slug",
    "max_input_tokens",
    "max_output_tokens",
    "supports_reasoning",
    "reasoning_effort_levels",
    "default_reasoning_effort",
    "supports_low_reasoning_effort",
    "supports_xhigh_reasoning_effort",
    "supports_max_reasoning_effort",
    "brand",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_litellm_master_key() -> str | None:
    """Resolve LiteLLM master key from settings or fallback env file without logging it."""
    if settings.litellm_master_key and settings.litellm_master_key.strip():
        return settings.litellm_master_key.strip()

    env_path = settings.litellm_env_file
    if not env_path or not os.path.exists(env_path):
        return None
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                match = _MASTER_KEY_LINE_RE.match(stripped)
                if match:
                    val = match.group(1).strip().strip('"').strip("'")
                    if val:
                        return val
    except OSError:
        return None
    return None


def build_litellm_deployment_payload(
    spec: AgyBaseModelSpec,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Build a LiteLLM `/model/new` deployment payload for an AGY base model."""
    resolved_api_base = (api_base or settings.cad_public_api_base).rstrip("/")
    levels = list(spec.reasoning_effort_levels)
    has_reasoning = bool(levels)
    model_id = f"cad-sync::{spec.litellm_model_name}"

    return {
        "model_name": spec.litellm_model_name,
        "litellm_params": {
            "model": f"openai/{spec.litellm_model_name}",
            "api_base": resolved_api_base,
            "api_key": "not-needed",
        },
        "model_info": {
            "id": model_id,
            "managed_by": MANAGED_BY_TAG,
            "source_base_slug": spec.base_slug,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "max_input_tokens": spec.max_input_tokens,
            "max_output_tokens": spec.max_output_tokens,
            "supports_reasoning": has_reasoning,
            "reasoning_effort_levels": levels or None,
            "default_reasoning_effort": spec.default_reasoning_effort,
            "supports_low_reasoning_effort": "low" in levels,
            "supports_xhigh_reasoning_effort": "xhigh" in levels,
            "supports_max_reasoning_effort": "max" in levels,
            "brand": spec.brand,
            "pricing_tier": "free",
            "capabilities": {
                "text_in": True,
                "text_out": True,
                "image_in": True,
                "image_out": False,
                "audio_in": False,
                "audio_out": False,
                "video_in": spec.supports_video,
                "video_out": False,
                "pdf_in": True,
                "function_calling": True,
                "streaming": True,
            },
        },
    }


def _deployment_needs_update(
    existing_entry: dict[str, Any],
    desired_payload: dict[str, Any],
) -> bool:
    existing_params = existing_entry.get("litellm_params") or {}
    desired_params = desired_payload["litellm_params"]
    if existing_params.get("model") != desired_params.get("model"):
        return True
    if existing_params.get("api_base") != desired_params.get("api_base"):
        return True

    existing_info = existing_entry.get("model_info") or {}
    desired_info = desired_payload["model_info"]
    for key in _COMPARED_INFO_KEYS:
        if existing_info.get(key) != desired_info.get(key):
            return True
    return False


class LiteLLMSyncService:
    """Reconciles discovered AGY base models into LiteLLM via its live Admin API."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.last_result: LiteLLMSyncResult | None = None

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": settings.litellm_sync_enabled,
            "base_url": settings.litellm_base_url,
            "interval_seconds": settings.litellm_sync_interval_seconds,
            "last_result": self.last_result.model_dump() if self.last_result else None,
        }

    async def sync_once(self, force_refresh: bool = False) -> LiteLLMSyncResult:
        async with self._lock:
            result = await self._reconcile(force_refresh=force_refresh)
            self.last_result = result
            return result

    async def _reconcile(self, force_refresh: bool) -> LiteLLMSyncResult:
        now_iso = _utc_now_iso()
        master_key = resolve_litellm_master_key()
        if not master_key:
            logger.warning("Skipping LiteLLM sync: LITELLM_MASTER_KEY not configured.")
            return LiteLLMSyncResult(
                status="skipped",
                synced_at=now_iso,
                from_live_cli=False,
                errors=["LITELLM_MASTER_KEY not configured"],
            )

        discovered, from_live_cli = await asyncio.to_thread(
            discover_agy_models_with_status, force_refresh
        )
        catalog = build_agy_model_catalog(discovered=discovered, include_default_alias=True)
        desired_by_name = {
            spec.litellm_model_name: build_litellm_deployment_payload(spec) for spec in catalog
        }

        base_url = settings.litellm_base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        }

        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        unchanged: list[str] = []
        skipped_conflicts: list[str] = []
        errors: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                info_resp = await client.get(f"{base_url}/v1/model/info", headers=headers)
                if info_resp.status_code != 200:
                    msg = f"GET /v1/model/info failed with HTTP {info_resp.status_code}"
                    logger.warning(msg)
                    return LiteLLMSyncResult(
                        status="error",
                        synced_at=now_iso,
                        from_live_cli=from_live_cli,
                        errors=[msg],
                    )

                existing_entries = info_resp.json().get("data") or []
                managed_by_name: dict[str, dict[str, Any]] = {}
                static_yaml_names: set[str] = set()

                for entry in existing_entries:
                    if not isinstance(entry, dict):
                        continue
                    m_name = entry.get("model_name")
                    if not isinstance(m_name, str) or not m_name:
                        continue
                    m_info = entry.get("model_info") or {}
                    if m_info.get("db_model") is False:
                        static_yaml_names.add(m_name)
                    elif (
                        m_info.get("db_model") is True
                        and m_info.get("managed_by") == MANAGED_BY_TAG
                    ):
                        managed_by_name[m_name] = entry

                for model_name, payload in desired_by_name.items():
                    if model_name in static_yaml_names:
                        skipped_conflicts.append(model_name)
                        logger.warning(
                            "Skipping LiteLLM DB sync for '%s': static config.yaml deployment exists.",
                            model_name,
                        )
                        continue

                    existing = managed_by_name.get(model_name)
                    if existing is None:
                        resp = await client.post(
                            f"{base_url}/model/new", headers=headers, json=payload
                        )
                        if resp.status_code < 300:
                            created.append(model_name)
                        else:
                            errors.append(f"create {model_name}: HTTP {resp.status_code}")
                    elif _deployment_needs_update(existing, payload):
                        existing_id = str((existing.get("model_info") or {}).get("id") or "")
                        patch_resp = await client.patch(
                            f"{base_url}/model/{existing_id}/update",
                            headers=headers,
                            json=payload,
                        )
                        if patch_resp.status_code < 300:
                            updated.append(model_name)
                        else:
                            del_resp = await client.post(
                                f"{base_url}/model/delete",
                                headers=headers,
                                json={"id": existing_id},
                            )
                            new_resp = await client.post(
                                f"{base_url}/model/new", headers=headers, json=payload
                            )
                            if del_resp.status_code < 300 and new_resp.status_code < 300:
                                updated.append(model_name)
                            else:
                                errors.append(
                                    f"update {model_name}: patch HTTP {patch_resp.status_code}, "
                                    f"recreate HTTP {del_resp.status_code}/{new_resp.status_code}"
                                )
                    else:
                        unchanged.append(model_name)

                if from_live_cli and desired_by_name:
                    for model_name, existing in managed_by_name.items():
                        if model_name not in desired_by_name:
                            existing_id = str((existing.get("model_info") or {}).get("id") or "")
                            if not existing_id:
                                continue
                            del_resp = await client.post(
                                f"{base_url}/model/delete",
                                headers=headers,
                                json={"id": existing_id},
                            )
                            if del_resp.status_code < 300:
                                deleted.append(model_name)
                            else:
                                errors.append(f"delete {model_name}: HTTP {del_resp.status_code}")
        except Exception as exc:
            msg = f"LiteLLM sync request failed: {exc}"
            logger.warning(msg)
            return LiteLLMSyncResult(
                status="error",
                synced_at=now_iso,
                from_live_cli=from_live_cli,
                created=created,
                updated=updated,
                deleted=deleted,
                unchanged=unchanged,
                skipped_conflicts=skipped_conflicts,
                errors=[*errors, msg],
            )

        status = "error" if errors else "success"
        logger.info(
            "LiteLLM sync %s (live=%s): created=%d updated=%d deleted=%d unchanged=%d conflicts=%d errors=%d",
            status,
            from_live_cli,
            len(created),
            len(updated),
            len(deleted),
            len(unchanged),
            len(skipped_conflicts),
            len(errors),
        )
        return LiteLLMSyncResult(
            status=status,
            synced_at=now_iso,
            from_live_cli=from_live_cli,
            created=created,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            skipped_conflicts=skipped_conflicts,
            errors=errors,
        )


litellm_sync_service = LiteLLMSyncService()

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

    delete_calls = [
        call for call in mock_client.post.call_args_list if "/model/delete" in str(call.args[0])
    ]
    assert len(delete_calls) == 1
    assert delete_calls[0].kwargs["json"] == {"id": "cad-sync::agy-retired"}


@pytest.mark.asyncio
async def test_sync_falls_back_to_delete_and_create_when_patch_fails(monkeypatch):
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_master_key", "sk-test"
    )
    monkeypatch.setattr(
        "cli_agent_dispatch.core.litellm_sync.settings.litellm_base_url", "http://litellm:4000"
    )
    existing_info = {
        "data": [
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
                    "reasoning_effort_levels": ["low"],
                },
            }
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(200, existing_info))
    mock_client.patch = AsyncMock(return_value=_mock_response(400, {"error": "patch unsupported"}))
    mock_client.post = AsyncMock(return_value=_mock_response(200, {"status": "ok"}))

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
    assert "agy-hydrogen" in result.updated
    delete_calls = [c for c in mock_client.post.call_args_list if "/model/delete" in str(c.args[0])]
    assert any(c.kwargs["json"] == {"id": "cad-sync::agy-hydrogen"} for c in delete_calls)


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

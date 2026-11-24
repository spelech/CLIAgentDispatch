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
    with (
        patch.object(
            agy_mod,
            "discover_agy_models_with_status",
            return_value=(SAMPLE_AGY_DISCOVERED, True),
        ),
        patch.object(agy_mod, "discover_agy_models", return_value=SAMPLE_AGY_DISCOVERED),
    ):
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
        assert "gemini-4-argon-low" not in models

    with (
        patch.object(agy_mod, "_resolve_agy_binary", return_value="/usr/bin/agy"),
        patch("subprocess.run", side_effect=OSError("boom")),
    ):
        models, from_live = agy_mod.discover_agy_models_with_status(force_refresh=True)
        assert from_live is False
        assert "gemini-4-hydrogen-high" in models

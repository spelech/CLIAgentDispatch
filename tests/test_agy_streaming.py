from unittest.mock import MagicMock, patch

import pytest

from cli_agent_dispatch.core.models import ExecutorType, TaskRequest
from cli_agent_dispatch.executors.agy import (
    DEFAULT_AGY_MODELS,
    AgyExecutor,
    discover_agy_models,
    normalize_agy_model,
)


def test_normalize_agy_model():
    from cli_agent_dispatch.config import settings

    assert normalize_agy_model(None) == settings.agy_default_model
    assert normalize_agy_model("claude-3-5-sonnet") == "Claude Sonnet 5.5 (Medium)"
    assert normalize_agy_model("claude-opus") == "Claude Opus 5.5 (Medium)"
    assert normalize_agy_model("gpt-oss-120b") == "GPT-OSS 120B (Medium)"
    assert normalize_agy_model("gemini-flash-low") == "Gemini 3.8 Flash (Low)"
    assert normalize_agy_model("gemini-flash-high") == "Gemini 3.8 Flash (High)"
    assert normalize_agy_model("custom-exact-name") == "custom-exact-name"

    # Exact IDs & Argon models
    assert normalize_agy_model("gemini-4-argon-medium") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("gemini-4-argon-high") == "Gemini 4 Argon (High)"
    assert normalize_agy_model("gemini-4-argon-low") == "Gemini 4 Argon (Low)"
    assert normalize_agy_model("agy-argon") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("agy-argon-high") == "Gemini 4 Argon (High)"
    assert normalize_agy_model("agy-argon-low") == "Gemini 4 Argon (Low)"
    assert normalize_agy_model("argon") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("agy-opus") == "Claude Opus 5.5 (High)"
    assert normalize_agy_model("agy-sonnet") == "Claude Sonnet 5.5 (High)"
    assert normalize_agy_model("openai/gemini-4-argon-medium") == "Gemini 4 Argon (Medium)"
    assert normalize_agy_model("litellm/agy-argon-high") == "Gemini 4 Argon (High)"


def test_discover_agy_models_parses_spinner_and_ansi():
    fake_stdout = (
        "\r\x1b[36m⠋\x1b[0m Fetching available models..."
        "\r\x1b[36m⠙\x1b[0m Fetching available models...\r"
        "gemini-4-argon-medium     Gemini 4 Argon (Medium)\n"
        "gemini-4-argon-high       Gemini 4 Argon (High)\n"
        "custom-agy-exp-high       Custom Experimental (High)\n"
    )
    mock_proc = MagicMock(returncode=0, stdout=fake_stdout, stderr="")
    with (
        patch("cli_agent_dispatch.executors.agy._resolve_agy_binary", return_value="/usr/bin/agy"),
        patch("subprocess.run", return_value=mock_proc),
    ):
        models = discover_agy_models(force_refresh=True)

    for k, v in DEFAULT_AGY_MODELS.items():
        assert models[k] == v
    assert models["custom-agy-exp-high"] == "Custom Experimental (High)"
    assert normalize_agy_model("custom-agy-exp-high") == "Custom Experimental (High)"


@pytest.mark.asyncio
async def test_agy_execute_stream_ndjson(tmp_path):
    fake_agy = tmp_path / "agy"
    script = (
        "#!/bin/sh\n"
        'echo \'{"type":"init","conversation_id":"conv-test-999"}\'\n'
        'echo \'{"type":"step_update","text_delta":"First token "}\'\n'
        'echo \'{"type":"step_update","text_delta":"second token"}\'\n'
        'echo \'{"type":"result","response":"full response","usage":{"total_tokens":42}}\'\n'
    )
    fake_agy.write_text(script)
    fake_agy.chmod(0o755)

    executor = AgyExecutor()
    req = TaskRequest(
        prompt="Streaming task",
        executor=ExecutorType.AGY,
        workspace=str(tmp_path),
    )

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", str(fake_agy)):
        chunks = []
        async for chunk in executor.execute_stream(req):
            chunks.append(chunk)

    assert len(chunks) >= 3
    # Check conversation_id propagation
    assert any(c.session_id == "conv-test-999" for c in chunks)
    deltas = "".join(c.delta for c in chunks)
    assert "First token second token" in deltas
    # Final chunk verification
    final_chunk = next(c for c in chunks if c.is_final)
    assert final_chunk.finish_reason == "stop"


@pytest.mark.asyncio
async def test_agy_execute_stream_binary_not_found():
    executor = AgyExecutor()
    req = TaskRequest(prompt="task", executor=ExecutorType.AGY)

    with patch("cli_agent_dispatch.executors.agy.settings.agy_path", "/nonexistent/agy"):
        chunks = []
        async for chunk in executor.execute_stream(req):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].is_final is True
    assert "not found" in chunks[0].delta

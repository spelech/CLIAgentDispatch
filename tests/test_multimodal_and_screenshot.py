import base64
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from cli_agent_dispatch.api.anthropic_routes import _extract_text_content
from cli_agent_dispatch.api.session_routes import normalize_preview_url
from cli_agent_dispatch.core.models import (
    ChatCompletionMessage,
    normalize_container_media_paths,
    save_attached_image,
)
from cli_agent_dispatch.main import app

SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
SAMPLE_PNG_B64 = base64.b64encode(SAMPLE_PNG_BYTES).decode("ascii")


def test_openai_multimodal_image_url_persisted_and_annotated(tmp_path, monkeypatch):
    monkeypatch.setenv("CLI_AGENT_DISPATCH_UPLOAD_DIR", str(tmp_path))
    msg = ChatCompletionMessage(
        role="user",
        content=[
            {"type": "text", "text": "Please fix the alignment in this screenshot:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{SAMPLE_PNG_B64}"},
            },
        ],
    )
    assert "Please fix the alignment in this screenshot:" in msg.content
    assert "[Attached Image:" in msg.content
    assert "use view_file to inspect this image]" in msg.content

    files = list(tmp_path.glob("img_*.png"))
    assert len(files) == 1
    assert files[0].read_bytes() == SAMPLE_PNG_BYTES
    assert str(files[0]) in msg.content

    # Second message with identical image should deduplicate by SHA-256 hash
    msg2 = ChatCompletionMessage(
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{SAMPLE_PNG_B64}"}}
        ],
    )
    assert str(files[0]) in msg2.content
    assert len(list(tmp_path.glob("img_*.png"))) == 1


def test_anthropic_multimodal_base64_image_extracted(tmp_path, monkeypatch):
    monkeypatch.setenv("CLI_AGENT_DISPATCH_UPLOAD_DIR", str(tmp_path))
    extracted = _extract_text_content(
        [
            {"type": "text", "text": "Inspect this UI card:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": SAMPLE_PNG_B64,
                },
            },
        ]
    )
    assert "Inspect this UI card:" in extracted
    assert "[Attached Image:" in extracted
    files = list(tmp_path.glob("img_*.png"))
    assert len(files) == 1
    assert str(files[0]) in extracted


def test_container_media_path_translation():
    raw = (
        "Downloaded Slack file to /home/node/.openclaw/media/inbound/ui_mockup.png "
        "and config in /app/data/modes.json"
    )
    translated = normalize_container_media_paths(raw)
    assert "/containers/ai/openclaw/data/uploads/slack_media/inbound/ui_mockup.png" in translated
    assert "/containers/ai/openclaw/data/modes.json" in translated


def test_save_attached_image_local_path(tmp_path):
    img_file = tmp_path / "local_test.png"
    img_file.write_bytes(SAMPLE_PNG_BYTES)
    resolved = save_attached_image(str(img_file), upload_dir=str(tmp_path))
    assert resolved == str(img_file)


def test_normalize_preview_url():
    assert (
        normalize_preview_url("https://preview.wileyriley.com/my-ui-slot/")
        == "http://preview.lan/my-ui-slot/"
    )
    assert (
        normalize_preview_url("https://p-8040.wileyriley.com/dashboard?tab=1")
        == "http://10.0.0.10:8040/dashboard?tab=1"
    )
    assert normalize_preview_url("my-slot") == "http://preview.lan/my-slot/"


def test_screenshot_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CLI_AGENT_DISPATCH_UPLOAD_DIR", str(tmp_path))

    async def fake_exec(*args, **kwargs):
        cfg = json.loads(args[3])
        with open(cfg["out_path"], "wb") as f:
            f.write(SAMPLE_PNG_BYTES)
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    with patch(
        "cli_agent_dispatch.api.session_routes.asyncio.create_subprocess_exec",
        side_effect=fake_exec,
    ):
        client = TestClient(app)
        resp = client.post(
            "/v1/screenshot",
            json={
                "url": "https://preview.wileyriley.com/demo-slot/",
                "mobile": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["rendered_url"] == "http://preview.lan/demo-slot/"
        assert data["path"].startswith(str(tmp_path))
        assert data["path"].endswith("_mobile.png")
        assert data["size_bytes"] == len(SAMPLE_PNG_BYTES)

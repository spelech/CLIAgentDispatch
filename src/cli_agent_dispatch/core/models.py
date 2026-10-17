import base64
import hashlib
import os
import re
import urllib.request
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutorType(StrEnum):
    OPENCODE = "opencode"
    AGY = "agy"


class TaskRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or task instructions for the agent.")
    session_id: str | None = Field(
        default=None,
        description="Session or conversation ID for multi-turn persistence.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt or persona instructions for the session.",
    )
    executor: ExecutorType = Field(
        default=ExecutorType.OPENCODE,
        description="Target executor engine ('opencode' or 'agy').",
    )
    workspace: str | None = Field(
        default=None,
        description="Working directory for code and file operations.",
    )
    model: str | None = Field(
        default=None,
        description="Override model ID (e.g. 'opencode-default', 'gemini-3.8-flash', 'claude-sonnet-5.5').",
    )
    provider: str | None = Field(
        default=None,
        description="Provider ID (primarily for OpenCode, e.g. 'litellm').",
    )
    timeout: int | None = Field(
        default=None,
        description="Execution timeout in seconds.",
    )


class TaskResult(BaseModel):
    success: bool = Field(..., description="Whether the task succeeded.")
    executor: str = Field(..., description="Executor engine that processed the task.")
    output: str = Field(..., description="Raw execution text output or summary.")
    error: str | None = Field(default=None, description="Error message if execution failed.")
    duration_seconds: float = Field(default=0.0, description="Elapsed execution time.")
    session_id: str | None = Field(
        default=None, description="Session ID if handled by OpenCode server."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic and execution metadata."
    )


# --- SRE Investigation Schemas ---


class InvestigationRequest(BaseModel):
    target: str = Field(..., description="Target service, container, or component name.")
    exit_code: str | int | None = Field(default=None, description="Exit code or error status code.")
    error_logs: str = Field(default="", description="Error logs or failure output to investigate.")
    workspace: str | None = Field(
        default=None, description="Working directory for code and file operations."
    )
    executor: ExecutorType = Field(
        default=ExecutorType.OPENCODE,
        description="Target executor engine ('opencode' or 'agy').",
    )
    max_turns: int = Field(default=5, description="Maximum number of diagnostic turns allowed.")
    model: str | None = Field(default=None, description="Override model ID.")
    provider: str | None = Field(default=None, description="Provider ID.")
    timeout: int | None = Field(default=None, description="Timeout in seconds per turn.")
    session_id: str | None = Field(
        default=None, description="Optional existing session ID to resume."
    )


class InvestigationResult(BaseModel):
    success: bool = Field(..., description="Whether the investigation completed successfully.")
    session_id: str | None = Field(
        default=None, description="Active session ID for follow-up turns."
    )
    target: str = Field(..., description="Target service or container investigated.")
    executor: str | None = Field(
        default=None, description="Executor engine used for the investigation."
    )
    root_cause: str = Field(default="", description="Diagnosed root cause.")
    proposed_fix: str = Field(default="", description="Remediation bash commands or proposed fix.")
    category: str = Field(default="unknown", description="Issue category classification.")
    turns_used: int = Field(default=1, description="Number of turns used in the investigation.")
    duration_seconds: float = Field(
        default=0.0, description="Total investigation duration in seconds."
    )
    transcript: list[dict[str, Any]] = Field(
        default_factory=list, description="Transcript of interaction turns."
    )
    error: str | None = Field(default=None, description="Error message if investigation failed.")
    raw_output: str | None = Field(default=None, description="Raw model text output.")


# --- Interactive Session Schemas ---


class SessionCreateRequest(BaseModel):
    executor: ExecutorType = Field(
        default=ExecutorType.OPENCODE,
        description="Target executor engine ('opencode' or 'agy').",
    )
    prompt: str | None = Field(
        default=None, description="Optional initial prompt to start the session."
    )
    initial_prompt: str | None = Field(
        default=None, description="Alternative field for initial prompt."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt or persona instructions."
    )
    workspace: str | None = Field(
        default=None, description="Working directory for code and file operations."
    )
    model: str | None = Field(default=None, description="Override model ID.")
    provider: str | None = Field(default=None, description="Provider ID.")
    timeout: int | None = Field(default=None, description="Timeout in seconds.")

    @model_validator(mode="after")
    def populate_prompt(self) -> "SessionCreateRequest":
        if not self.prompt and self.initial_prompt:
            self.prompt = self.initial_prompt
        return self


class SessionMessageRequest(BaseModel):
    prompt: str | None = Field(
        default=None, description="Message or instructions for the active session."
    )
    message: str | None = Field(default=None, description="Alternative field for message.")
    executor: ExecutorType | None = Field(
        default=None, description="Override executor engine if needed."
    )
    workspace: str | None = Field(
        default=None, description="Working directory for code and file operations."
    )
    model: str | None = Field(default=None, description="Override model ID.")
    provider: str | None = Field(default=None, description="Provider ID.")
    timeout: int | None = Field(default=None, description="Timeout in seconds.")

    @model_validator(mode="after")
    def validate_message_content(self) -> "SessionMessageRequest":
        if not self.prompt and self.message:
            self.prompt = self.message
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Either 'prompt' or 'message' must be provided and non-empty.")
        return self


class StreamChunk(BaseModel):
    delta: str = Field(default="", description="Incremental text token or chunk.")
    reasoning_delta: str = Field(
        default="",
        description="Incremental reasoning or intermediate tool execution progress delta.",
    )
    is_final: bool = Field(
        default=False, description="Whether this is the final termination chunk."
    )
    session_id: str | None = Field(default=None, description="Active session/conversation ID.")
    finish_reason: str | None = Field(default=None, description="Stop or finish reason.")
    usage: dict[str, int] | None = Field(
        default=None, description="Usage metrics if reported by executor."
    )


# --- Multimodal Image Attachment & Media Path Helpers ---

_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "application/pdf": "pdf",
}

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[a-zA-Z0-9.+/-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$")


def resolve_upload_dir(upload_dir: str | None = None) -> str:
    """Resolve writable directory for decoded image/file attachments."""
    target = upload_dir or os.environ.get(
        "CLI_AGENT_DISPATCH_UPLOAD_DIR", "/containers/ai/openclaw/data/uploads"
    )
    try:
        os.makedirs(target, exist_ok=True)
        return target
    except OSError:
        fallback = "/tmp/cli-agent-dispatch-uploads"
        os.makedirs(fallback, exist_ok=True)
        return fallback


def normalize_container_media_paths(text: str) -> str:
    """Translate OpenClaw container-internal media paths to host-accessible paths for agy."""
    if not text:
        return text
    out = text.replace(
        "/home/node/.openclaw/media/", "/containers/ai/openclaw/data/uploads/slack_media/"
    )
    out = re.sub(
        r"(?<![A-Za-z0-9_.-])/app/data/",
        "/containers/ai/openclaw/data/",
        out,
    )
    return out


def _read_slack_bot_token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if token:
        return token
    env_file = "/containers/ai/openclaw/.env"
    if os.path.exists(env_file):
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("SLACK_BOT_TOKEN="):
                        return s.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def save_attached_image(
    image_source: str | dict[str, Any], upload_dir: str | None = None
) -> str | None:
    """
    Persist an inbound multimodal image attachment (data URL, Anthropic base64 block,
    local path, or HTTP/Slack URL) to disk so `agy` can inspect it with `view_file`.
    """
    target_dir = resolve_upload_dir(upload_dir)
    raw_bytes: bytes | None = None
    ext = "png"

    if isinstance(image_source, dict):
        # Anthropic base64 block: {"type": "base64", "media_type": "image/png", "data": "..."}
        src_type = str(image_source.get("type", "")).lower()
        if src_type == "base64" or ("data" in image_source and "media_type" in image_source):
            media_type = str(image_source.get("media_type", "image/png")).lower().strip()
            ext = _MIME_TO_EXT.get(media_type, "png")
            b64_data = str(image_source.get("data", "")).strip()
            if b64_data:
                try:
                    raw_bytes = base64.b64decode(b64_data)
                except Exception:
                    return None
        else:
            # OpenAI image_url dict: {"url": "..."}
            url_val = image_source.get("url") or image_source.get("path") or ""
            if isinstance(url_val, str) and url_val:
                return save_attached_image(url_val, upload_dir=target_dir)
            return None
    elif isinstance(image_source, str):
        src = image_source.strip()
        if not src:
            return None
        m = _DATA_URL_RE.match(src)
        if m:
            mime = m.group("mime").lower().strip()
            ext = _MIME_TO_EXT.get(mime, mime.split("/")[-1] if "/" in mime else "png")
            if ext == "jpeg":
                ext = "jpg"
            try:
                raw_bytes = base64.b64decode(m.group("data"))
            except Exception:
                return None
        elif src.startswith("file://"):
            local_path = normalize_container_media_paths(src[len("file://") :])
            if os.path.exists(local_path):
                return local_path
            return None
        elif src.startswith("/"):
            local_path = normalize_container_media_paths(src)
            if os.path.exists(local_path):
                return local_path
            return None
        elif src.startswith(("http://", "https://")):
            headers: dict[str, str] = {"User-Agent": "CLIAgentDispatch/0.3"}
            if "slack.com" in src:
                slack_token = _read_slack_bot_token()
                if slack_token:
                    headers["Authorization"] = f"Bearer {slack_token}"
            try:
                req = urllib.request.Request(src, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    ctype = (
                        (resp.headers.get("Content-Type") or "image/png")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    ext = _MIME_TO_EXT.get(ctype, "png")
                    raw_bytes = resp.read()
            except Exception:
                return None

    if not raw_bytes:
        return None

    digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
    filename = f"img_{digest}.{ext}"
    out_path = os.path.join(target_dir, filename)
    if not os.path.exists(out_path):
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
    return out_path


def format_image_attachment_directive(saved_path: str) -> str:
    return f"[Attached Image: {saved_path} — use view_file to inspect this image]"


# --- OpenAI Compatibility Schemas ---


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return normalize_container_media_paths(v)
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                if isinstance(item, str):
                    parts.append(normalize_container_media_paths(item))
                elif isinstance(item, dict):
                    item_type = str(item.get("type", "")).lower()
                    if item_type in ("image_url", "input_image", "image") or "image_url" in item:
                        img_src = (
                            item.get("image_url") or item.get("source") or item.get("url") or item
                        )
                        saved_path = save_attached_image(img_src)
                        if saved_path:
                            parts.append(format_image_attachment_directive(saved_path))
                        continue
                    text_val = item.get("text") or item.get("content") or ""
                    if isinstance(text_val, str) and text_val:
                        parts.append(normalize_container_media_paths(text_val))
            return "\n".join(parts)
        return normalize_container_media_paths(str(v))


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatCompletionMessage]
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None
    session_id: str | None = None
    user: str | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "cli-agent-dispatch"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# --- Anthropic Messages Compatibility Schemas ---


class AnthropicTextBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[AnthropicTextBlock] | list[dict[str, Any]]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicTextBlock] | list[dict[str, Any]] | None = None
    max_tokens: int = Field(default=4096)
    temperature: float | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    session_id: str | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[AnthropicContentBlock]
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# --- Headless UI Preview Screenshot Schemas ---


class ScreenshotRequest(BaseModel):
    url: str = Field(..., description="Target preview or localhost URL to capture.")
    output_path: str | None = Field(
        default=None, description="Optional destination path for the PNG screenshot."
    )
    width: int = Field(default=1280, ge=240, le=3840)
    height: int = Field(default=800, ge=240, le=3840)
    full_page: bool = Field(default=False)
    wait_ms: int = Field(default=800, ge=0, le=15000)
    mobile: bool = Field(default=False, description="Use mobile viewport (390x844, 2x scale).")


class ScreenshotResponse(BaseModel):
    success: bool
    path: str | None = None
    rendered_url: str | None = None
    size_bytes: int = 0
    error: str | None = None

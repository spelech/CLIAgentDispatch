import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
    ExecutorType,
    StreamChunk,
    TaskRequest,
    format_image_attachment_directive,
    normalize_container_media_paths,
    save_attached_image,
)
from cli_agent_dispatch.core.session_store import session_store

router = APIRouter(prefix="/v1", tags=["Anthropic Compatibility"])


def _extract_text_content(content: str | list[Any]) -> str:
    """Extract string content (and decoded image directives) from string or Anthropic content block list."""
    if isinstance(content, str):
        return normalize_container_media_paths(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).lower()
                if item_type == "text":
                    parts.append(normalize_container_media_paths(str(item.get("text", ""))))
                elif item_type in ("image", "image_url", "input_image") or "image_url" in item:
                    img_src = item.get("source") or item.get("image_url") or item.get("url") or item
                    saved_path = save_attached_image(img_src)
                    if saved_path:
                        parts.append(format_image_attachment_directive(saved_path))
            elif hasattr(item, "text"):
                parts.append(normalize_container_media_paths(str(item.text)))
        return "\n".join(parts)
    return normalize_container_media_paths(str(content))


def _resolve_executor_and_model(model_name: str) -> tuple[ExecutorType, str | None]:
    raw = model_name.strip()
    for prefix in ("openai/", "litellm/", "anthropic/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    name_lower = raw.lower()
    if name_lower == "t3" or name_lower.startswith(("t3-", "t3/", "t3_")):
        return ExecutorType.T3, raw
    if any(
        k in name_lower for k in ["claude", "agy", "argon", "gemini", "opus", "sonnet", "gpt-oss"]
    ):
        exec_type = ExecutorType.AGY
        model_override = raw
    else:
        exec_type = ExecutorType.OPENCODE
        if "small" in name_lower:
            model_override = "opencode-small"
        elif "default" in name_lower or "opencode" in name_lower:
            model_override = "opencode-default"
        else:
            model_override = raw
    return exec_type, model_override


def _format_anthropic_prompt(messages: list[AnthropicMessage]) -> str:
    """Format full chat messages into an agent prompt."""
    prompt_lines = []
    for msg in messages:
        role = msg.role.capitalize()
        text = _extract_text_content(msg.content)
        prompt_lines.append(f"{role}: {text}")
    return "\n\n".join(prompt_lines)


def _extract_anthropic_delta_turn(messages: list[AnthropicMessage]) -> str:
    """Extract latest user turn message for active multi-turn sessions."""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            return _extract_text_content(msg.content)
    return _extract_text_content(messages[-1].content) if messages else ""


def _extract_anthropic_effort(request: AnthropicMessagesRequest) -> str | None:
    """Resolve reasoning effort from Anthropic request fields, metadata, or thinking budget."""
    if request.effort:
        return request.effort
    if request.reasoning_effort:
        return request.reasoning_effort
    if request.metadata and isinstance(request.metadata, dict):
        meta_effort = request.metadata.get("effort") or request.metadata.get("reasoning_effort")
        if isinstance(meta_effort, str) and meta_effort.strip():
            return meta_effort.strip()
    if request.thinking and isinstance(request.thinking, dict):
        t_effort = request.thinking.get("effort")
        if isinstance(t_effort, str) and t_effort.strip():
            return t_effort.strip()
        if str(request.thinking.get("type", "")).lower() == "enabled":
            budget = request.thinking.get("budget_tokens")
            if isinstance(budget, (int, float)):
                if budget >= 32000:
                    return "max"
                if budget >= 24000:
                    return "xhigh"
                if budget >= 16000:
                    return "high"
                if budget <= 4000:
                    return "low"
                return "medium"
    return None


async def _stream_anthropic_sse(
    message_id: str,
    model: str,
    stream_gen: AsyncGenerator[StreamChunk, None],
    client_session_id: str | None,
    exec_type: ExecutorType,
    input_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    # 1. message_start
    start_event = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

    total_output_tokens = 0
    block_index = -1
    current_block_type: str | None = None

    # 2. Stream deltas (dynamically switching between thinking and text content blocks)
    async for chunk in stream_gen:
        if chunk.session_id and client_session_id:
            await session_store.get_or_create(
                client_id=client_session_id,
                executor_type=exec_type,
                native_session_id=chunk.session_id,
            )

        if chunk.reasoning_delta:
            if current_block_type != "thinking":
                if current_block_type is not None:
                    stop_evt = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n"
                block_index += 1
                current_block_type = "thinking"
                start_evt = {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n"

            total_output_tokens += max(1, len(chunk.reasoning_delta) // 4)
            thinking_evt = {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": chunk.reasoning_delta},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(thinking_evt)}\n\n"

        if chunk.delta:
            if current_block_type != "text":
                if current_block_type is not None:
                    stop_evt = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n"
                block_index += 1
                current_block_type = "text"
                start_evt = {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n"

            total_output_tokens += max(1, len(chunk.delta) // 4)
            delta_event = {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": chunk.delta},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

    # Ensure at least one text block is emitted if the stream produced no blocks
    if current_block_type is None:
        block_index = 0
        block_start_event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(block_start_event)}\n\n"

    # Close the final active content block
    block_stop_event = {
        "type": "content_block_stop",
        "index": block_index,
    }
    yield f"event: content_block_stop\ndata: {json.dumps(block_stop_event)}\n\n"

    # message_delta
    msg_delta_event = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": total_output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta_event)}\n\n"

    # message_stop
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


@router.post("/messages")
async def create_anthropic_message(
    request: AnthropicMessagesRequest,
    response: Response,
    x_session_id: str | None = Header(default=None),
    x_thread_id: str | None = Header(default=None),
):
    """Anthropic Messages API (/v1/messages) compatibility endpoint for OpenClaw/LiteLLM."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    exec_type, model_override = _resolve_executor_and_model(request.model)

    metadata_session = None
    if request.metadata and isinstance(request.metadata, dict):
        metadata_session = request.metadata.get("session_id") or request.metadata.get("thread_id")
    client_session_id = request.session_id or metadata_session or x_session_id or x_thread_id

    native_session_id = None
    is_resumed_turn = False

    system_text = None
    if request.system:
        system_text = _extract_text_content(request.system)

    if client_session_id:
        existing_session = await session_store.get(client_session_id)
        if existing_session and existing_session.native_session_id:
            native_session_id = existing_session.native_session_id
            exec_type = existing_session.executor_type
            is_resumed_turn = True

    if is_resumed_turn:
        prompt_text = _extract_anthropic_delta_turn(request.messages)
        system_text = None
    else:
        prompt_text = _format_anthropic_prompt(request.messages)

    task_req = TaskRequest(
        prompt=prompt_text,
        system_prompt=system_text,
        session_id=native_session_id,
        executor=exec_type,
        model=model_override,
        effort=_extract_anthropic_effort(request),
    )

    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    active_client_id = client_session_id or f"session_{uuid.uuid4().hex[:12]}"
    in_tokens = max(1, len(prompt_text) // 4)

    if request.stream:
        stream_gen = engine.dispatch_stream(task_req)
        resp = StreamingResponse(
            _stream_anthropic_sse(
                message_id=message_id,
                model=request.model,
                stream_gen=stream_gen,
                client_session_id=active_client_id,
                exec_type=exec_type,
                input_tokens=in_tokens,
            ),
            media_type="text/event-stream",
        )
        resp.headers["x-session-id"] = active_client_id
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        return resp

    # Non-streaming response
    result = await engine.dispatch(task_req)
    reply_content = (
        result.output
        if result.success
        else f"[Dispatch Error via {result.executor}]: {result.error}\n\n{result.output}"
    )

    resolved_native_id = result.session_id or native_session_id
    await session_store.get_or_create(
        client_id=active_client_id,
        executor_type=exec_type,
        native_session_id=resolved_native_id,
    )

    response.headers["x-session-id"] = active_client_id
    response.headers["x-native-session-id"] = str(resolved_native_id or "")

    out_tokens = max(1, len(reply_content) // 4)
    return AnthropicMessagesResponse(
        id=message_id,
        type="message",
        role="assistant",
        model=request.model,
        content=[AnthropicContentBlock(type="text", text=reply_content)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=in_tokens, output_tokens=out_tokens),
    )

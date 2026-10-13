import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.models import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionUsage,
    ExecutorType,
    ModelCard,
    ModelListResponse,
    StreamChunk,
    TaskRequest,
)
from cli_agent_dispatch.core.session_store import session_store

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])

AVAILABLE_MODELS = [
    ModelCard(id="opencode"),
    ModelCard(id="opencode-default"),
    ModelCard(id="opencode-small"),
    ModelCard(id="agy"),
    ModelCard(id="agy-gemini"),
    ModelCard(id="agy-claude"),
    ModelCard(id="gemini-3.8-flash"),
    ModelCard(id="claude-sonnet-5.5"),
    ModelCard(id="claude-opus-5.5"),
    ModelCard(id="gpt-oss-120b"),
]


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available local agent models for OpenAI-compatible clients."""
    return ModelListResponse(data=AVAILABLE_MODELS)


def _resolve_executor_and_model(model_name: str) -> tuple[ExecutorType, str | None]:
    name_lower = model_name.lower()
    if any(k in name_lower for k in ["agy", "claude", "gemini", "opus", "sonnet", "gpt-oss"]):
        exec_type = ExecutorType.AGY
        model_override = model_name
    else:
        exec_type = ExecutorType.OPENCODE
        if "small" in name_lower:
            model_override = "opencode-small"
        elif "default" in name_lower or "opencode" in name_lower:
            model_override = "opencode-default"
        else:
            model_override = model_name
    return exec_type, model_override


def _format_prompt(messages: list[ChatCompletionMessage]) -> str:
    """Format full chat messages into an agent prompt."""
    prompt_lines = []
    for msg in messages:
        role = msg.role.capitalize()
        prompt_lines.append(f"{role}: {msg.content}")
    return "\n\n".join(prompt_lines)


def _extract_delta_turn(messages: list[ChatCompletionMessage]) -> str:
    """Extract only the latest user message for follow-up turns in active sessions."""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            return msg.content
    return messages[-1].content if messages else ""


async def _stream_chat_completion_sse(
    completion_id: str,
    model: str,
    stream_gen: AsyncGenerator[StreamChunk, None],
    client_session_id: str | None,
    exec_type: ExecutorType,
) -> AsyncGenerator[str, None]:
    created = int(time.time())

    async for chunk in stream_gen:
        if chunk.session_id and client_session_id:
            await session_store.get_or_create(
                client_id=client_session_id,
                executor_type=exec_type,
                native_session_id=chunk.session_id,
            )

        if chunk.delta:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": chunk.delta},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        if chunk.is_final:
            stop_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": chunk.finish_reason or "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_payload)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    response: Response,
    x_session_id: str | None = Header(default=None),
    x_thread_id: str | None = Header(default=None),
):
    """OpenAI-compatible chat completions endpoint delegating to OpenCode or agy."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    exec_type, model_override = _resolve_executor_and_model(request.model)
    client_session_id = request.session_id or request.user or x_session_id or x_thread_id

    native_session_id = None
    is_resumed_turn = False
    system_prompt = next((m.content for m in request.messages if m.role.lower() == "system"), None)

    if client_session_id:
        existing_session = await session_store.get(client_session_id)
        if existing_session and existing_session.native_session_id:
            native_session_id = existing_session.native_session_id
            exec_type = existing_session.executor_type
            is_resumed_turn = True

    if is_resumed_turn:
        # Avoid quadratic history duplication by sending only the latest delta turn
        prompt_text = _extract_delta_turn(request.messages)
        system_text = None
    else:
        prompt_text = _format_prompt(request.messages)
        system_text = system_prompt

    task_req = TaskRequest(
        prompt=prompt_text,
        system_prompt=system_text,
        session_id=native_session_id,
        executor=exec_type,
        model=model_override,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    active_client_id = client_session_id or f"session_{uuid.uuid4().hex[:12]}"

    if request.stream:
        stream_gen = engine.dispatch_stream(task_req)
        resp = StreamingResponse(
            _stream_chat_completion_sse(
                completion_id=completion_id,
                model=request.model,
                stream_gen=stream_gen,
                client_session_id=active_client_id,
                exec_type=exec_type,
            ),
            media_type="text/event-stream",
        )
        resp.headers["x-session-id"] = active_client_id
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        return resp

    # Non-streaming execution
    result = await engine.dispatch(task_req)
    reply_content = (
        result.output
        if result.success
        else f"[Dispatch Error via {result.executor}]: {result.error}\n\n{result.output}"
    )

    # Persist session mapping
    resolved_native_id = result.session_id or native_session_id
    await session_store.get_or_create(
        client_id=active_client_id,
        executor_type=exec_type,
        native_session_id=resolved_native_id,
    )

    response.headers["x-session-id"] = active_client_id
    response.headers["x-native-session-id"] = str(resolved_native_id or "")

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=reply_content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(prompt_text) // 4,
            completion_tokens=len(reply_content) // 4,
            total_tokens=(len(prompt_text) + len(reply_content)) // 4,
        ),
    )

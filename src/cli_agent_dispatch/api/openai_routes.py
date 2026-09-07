import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
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
    TaskRequest,
)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])

AVAILABLE_MODELS = [
    ModelCard(id="opencode"),
    ModelCard(id="opencode-qwen"),
    ModelCard(id="agy"),
    ModelCard(id="agy-gemini"),
]


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available local agent models for OpenAI-compatible clients."""
    return ModelListResponse(data=AVAILABLE_MODELS)


def _resolve_executor_and_model(model_name: str) -> tuple[ExecutorType, str | None]:
    name_lower = model_name.lower()
    if "agy" in name_lower:
        exec_type = ExecutorType.AGY
        model_override = "gemini-2.5-flash" if "gemini" in name_lower else None
    else:
        exec_type = ExecutorType.OPENCODE
        model_override = "qwen3.7-flash" if "qwen" in name_lower else None
    return exec_type, model_override


def _format_prompt(messages: list) -> str:
    """Format chat messages into an agent prompt."""
    prompt_lines = []
    for msg in messages:
        role = msg.role.capitalize()
        prompt_lines.append(f"{role}: {msg.content}")
    return "\n\n".join(prompt_lines)


async def _stream_chat_completion(
    completion_id: str, model: str, content: str
) -> AsyncGenerator[str, None]:
    created = int(time.time())

    # Send content chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Send final stop chunk
    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
):
    """OpenAI-compatible chat completions endpoint delegating to OpenCode or agy."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    exec_type, model_override = _resolve_executor_and_model(request.model)
    full_prompt = _format_prompt(request.messages)

    task_req = TaskRequest(
        prompt=full_prompt,
        executor=exec_type,
        model=model_override,
    )

    result = await engine.dispatch(task_req)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    reply_content = (
        result.output
        if result.success
        else f"[Dispatch Error via {result.executor}]: {result.error}\n\n{result.output}"
    )

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(completion_id, request.model, reply_content),
            media_type="text/event-stream",
        )

    # Return non-streaming response
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
            prompt_tokens=len(full_prompt) // 4,
            completion_tokens=len(reply_content) // 4,
            total_tokens=(len(full_prompt) + len(reply_content)) // 4,
        ),
    )

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ExecutorType(StrEnum):
    OPENCODE = "opencode"
    AGY = "agy"


class TaskRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or task instructions for the agent.")
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
        description="Override model ID (e.g. 'qwen3.7-flash', 'gemini-2.5-flash').",
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


# --- OpenAI Compatibility Schemas ---


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None


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

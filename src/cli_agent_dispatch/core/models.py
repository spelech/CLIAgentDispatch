from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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


# --- SRE Investigation Schemas ---


class InvestigationRequest(BaseModel):
    target: str = Field(..., description="Target service, container, or component name.")
    exit_code: str | int | None = Field(
        default=None, description="Exit code or error status code."
    )
    error_logs: str = Field(
        default="", description="Error logs or failure output to investigate."
    )
    workspace: str | None = Field(
        default=None, description="Working directory for code and file operations."
    )
    executor: ExecutorType = Field(
        default=ExecutorType.OPENCODE,
        description="Target executor engine ('opencode' or 'agy').",
    )
    max_turns: int = Field(
        default=5, description="Maximum number of diagnostic turns allowed."
    )
    model: str | None = Field(
        default=None, description="Override model ID."
    )
    provider: str | None = Field(
        default=None, description="Provider ID."
    )
    timeout: int | None = Field(
        default=None, description="Timeout in seconds per turn."
    )
    session_id: str | None = Field(
        default=None, description="Optional existing session ID to resume."
    )


class InvestigationResult(BaseModel):
    success: bool = Field(..., description="Whether the investigation completed successfully.")
    session_id: str | None = Field(
        default=None, description="Active session ID for follow-up turns."
    )
    target: str = Field(..., description="Target service or container investigated.")
    root_cause: str = Field(
        default="", description="Diagnosed root cause."
    )
    proposed_fix: str = Field(
        default="", description="Remediation bash commands or proposed fix."
    )
    category: str = Field(
        default="unknown", description="Issue category classification."
    )
    turns_used: int = Field(
        default=1, description="Number of turns used in the investigation."
    )
    duration_seconds: float = Field(
        default=0.0, description="Total investigation duration in seconds."
    )
    transcript: list[dict[str, Any]] = Field(
        default_factory=list, description="Transcript of interaction turns."
    )
    error: str | None = Field(
        default=None, description="Error message if investigation failed."
    )
    raw_output: str | None = Field(
        default=None, description="Raw model text output."
    )


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
    model: str | None = Field(
        default=None, description="Override model ID."
    )
    provider: str | None = Field(
        default=None, description="Provider ID."
    )
    timeout: int | None = Field(
        default=None, description="Timeout in seconds."
    )

    @model_validator(mode="after")
    def populate_prompt(self) -> "SessionCreateRequest":
        if not self.prompt and self.initial_prompt:
            self.prompt = self.initial_prompt
        return self


class SessionMessageRequest(BaseModel):
    prompt: str | None = Field(
        default=None, description="Message or instructions for the active session."
    )
    message: str | None = Field(
        default=None, description="Alternative field for message."
    )
    executor: ExecutorType | None = Field(
        default=None, description="Override executor engine if needed."
    )
    workspace: str | None = Field(
        default=None, description="Working directory for code and file operations."
    )
    model: str | None = Field(
        default=None, description="Override model ID."
    )
    provider: str | None = Field(
        default=None, description="Provider ID."
    )
    timeout: int | None = Field(
        default=None, description="Timeout in seconds."
    )

    @model_validator(mode="after")
    def validate_message_content(self) -> "SessionMessageRequest":
        if not self.prompt and self.message:
            self.prompt = self.message
        if not self.prompt:
            raise ValueError("Either 'prompt' or 'message' must be provided.")
        return self


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

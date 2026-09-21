from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.engine import (
    DispatchEngine,
    delegate_task,
    engine,
    investigate_target,
)
from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationRequest,
    InvestigationResult,
    SessionCreateRequest,
    SessionMessageRequest,
    TaskRequest,
    TaskResult,
)
from cli_agent_dispatch.core.sre_prompts import (
    MCG_READ_ONLY_DIRECTIVE,
    build_investigation_prompt,
)
from cli_agent_dispatch.executors.agy import AgyExecutor
from cli_agent_dispatch.executors.base import BaseExecutor
from cli_agent_dispatch.executors.opencode import OpenCodeExecutor

__version__ = "0.1.0"

__all__ = [
    "settings",
    "engine",
    "DispatchEngine",
    "delegate_task",
    "investigate_target",
    "MCG_READ_ONLY_DIRECTIVE",
    "build_investigation_prompt",
    "TaskRequest",
    "TaskResult",
    "InvestigationRequest",
    "InvestigationResult",
    "SessionCreateRequest",
    "SessionMessageRequest",
    "ExecutorType",
    "BaseExecutor",
    "OpenCodeExecutor",
    "AgyExecutor",
]



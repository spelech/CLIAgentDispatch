import logging

from fastapi import APIRouter, HTTPException

from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationRequest,
    InvestigationResult,
    SessionCreateRequest,
    SessionMessageRequest,
    TaskRequest,
    TaskResult,
)

logger = logging.getLogger("CLIAgentDispatch.SessionRoutes")

router = APIRouter(prefix="/v1", tags=["Sessions & SRE"])


def _resolve_session_executor(session_id: str, executor: ExecutorType | None) -> ExecutorType:
    """Infers the target executor from session ID if not explicitly specified."""
    if executor:
        return executor
    if session_id.startswith("conv_") or "agy" in session_id.lower():
        return ExecutorType.AGY
    return ExecutorType.OPENCODE


@router.post("/investigate", response_model=InvestigationResult)
async def investigate(request: InvestigationRequest) -> InvestigationResult:
    """Executes an interactive SRE investigation session for a target failure."""
    try:
        return await engine.investigate(request)
    except Exception as e:
        logger.error(
            f"Error during SRE investigation for target '{request.target}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions")
async def create_session(
    request: SessionCreateRequest = SessionCreateRequest(),
) -> dict[str, str | None]:
    """Initializes a new interactive session via the requested executor."""
    prompt = request.prompt or "Initialize session"
    task_req = TaskRequest(
        prompt=prompt,
        executor=request.executor,
        system_prompt=request.system_prompt,
        workspace=request.workspace,
        model=request.model,
        provider=request.provider,
        timeout=request.timeout,
    )
    try:
        result = await engine.dispatch(task_req)
    except Exception as e:
        logger.error(
            f"Failed to create session via executor '{request.executor}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=result.error or "Failed to initialize session",
        )

    return {
        "session_id": result.session_id,
        "executor": result.executor,
        "status": "ready",
        "output": result.output,
    }


@router.post("/sessions/{session_id}/message", response_model=TaskResult)
async def send_session_message(
    session_id: str,
    request: SessionMessageRequest,
) -> TaskResult:
    """Sends a follow-up message to an active interactive session."""
    target_executor = _resolve_session_executor(session_id, request.executor)
    task_req = TaskRequest(
        prompt=request.prompt,  # Pydantic validator guarantees prompt is set
        session_id=session_id,
        executor=target_executor,
        workspace=request.workspace,
        model=request.model,
        provider=request.provider,
        timeout=request.timeout,
    )
    try:
        return await engine.dispatch(task_req)
    except ExecutorNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to dispatch message for session '{session_id}' via '{target_executor}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

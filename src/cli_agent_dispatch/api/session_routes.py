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
from cli_agent_dispatch.core.session_store import session_store

logger = logging.getLogger("CLIAgentDispatch.SessionRoutes")

router = APIRouter(prefix="/v1", tags=["Sessions & SRE"])

_SESSION_EXECUTORS: dict[str, ExecutorType] = {}


def register_session_executor(session_id: str | None, executor: ExecutorType) -> None:
    if session_id:
        _SESSION_EXECUTORS[session_id] = executor


def _resolve_session_executor(session_id: str, executor: ExecutorType | None) -> ExecutorType:
    """Infers the target executor from session ID if not explicitly specified."""
    if executor:
        return executor
    if session_id in _SESSION_EXECUTORS:
        return _SESSION_EXECUTORS[session_id]
    if session_id.startswith("conv_") or "agy" in session_id.lower():
        return ExecutorType.AGY
    return ExecutorType.OPENCODE


@router.post("/investigate", response_model=InvestigationResult)
async def investigate(request: InvestigationRequest) -> InvestigationResult:
    """Executes an interactive SRE investigation session for a target failure."""
    try:
        result = await engine.investigate(request)
        if result.session_id:
            register_session_executor(result.session_id, ExecutorType(result.executor))
        return result
    except Exception as e:
        logger.error(
            f"Error during SRE investigation for target '{request.target}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sessions")
async def create_session(
    request: SessionCreateRequest | None = None,
) -> dict[str, str | None]:
    """Initializes a new interactive session via the requested executor."""
    req = request or SessionCreateRequest()
    prompt = req.prompt or "Initialize session"
    task_req = TaskRequest(
        prompt=prompt,
        executor=req.executor,
        system_prompt=req.system_prompt,
        workspace=req.workspace,
        model=req.model,
        provider=req.provider,
        timeout=req.timeout,
    )
    try:
        result = await engine.dispatch(task_req)
    except Exception as e:
        logger.error(
            f"Failed to create session via executor '{req.executor}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=result.error or "Failed to initialize session",
        )

    if result.session_id:
        register_session_executor(result.session_id, ExecutorType(result.executor))

    return {
        "session_id": result.session_id,
        "executor": result.executor,
        "status": "ready",
        "output": result.output,
    }


@router.get("/sessions/active")
async def list_active_processes() -> dict[str, list[str] | int]:
    """List currently running executor subprocess IDs."""
    active_ids = session_store.active_process_ids()
    return {
        "active_ids": active_ids,
        "count": len(active_ids),
    }


@router.post("/sessions/cancel-active")
async def cancel_all_active_sessions() -> dict[str, list[str] | int | str]:
    """Immediately terminate all currently running agent turns/subprocesses."""
    cancelled = await session_store.cancel_all_active()
    logger.warning(f"Cancelled active agent subprocesses: {cancelled}")
    return {
        "status": "cancelled",
        "cancelled_ids": cancelled,
        "count": len(cancelled),
    }


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(session_id: str) -> dict[str, str | bool | list[str]]:
    """Terminate an active subprocess for a specific session or run ID."""
    if session_id in ("all", "active"):
        cancelled = await session_store.cancel_all_active()
        return {
            "status": "cancelled",
            "session_id": session_id,
            "cancelled": bool(cancelled),
            "cancelled_ids": cancelled,
        }
    killed = await session_store.kill_active_process(session_id)
    if not killed:
        # Fallback: check if client_id maps to a native_session_id with an active process
        record = await session_store.get(session_id)
        if record and record.native_session_id:
            killed = await session_store.kill_active_process(record.native_session_id)
    return {
        "status": "cancelled" if killed else "not_running",
        "session_id": session_id,
        "cancelled": killed,
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
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(
            f"Failed to dispatch message for session '{session_id}' via '{target_executor}': {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

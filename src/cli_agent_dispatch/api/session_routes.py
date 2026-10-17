import asyncio
import json
import logging
import os
import re
import time
import urllib.parse

from fastapi import APIRouter, HTTPException

from cli_agent_dispatch.core.engine import engine
from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationRequest,
    InvestigationResult,
    ScreenshotRequest,
    ScreenshotResponse,
    SessionCreateRequest,
    SessionMessageRequest,
    TaskRequest,
    TaskResult,
    resolve_upload_dir,
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


_PORT_SUBDOMAIN_RE = re.compile(r"^p-(\d+)\.wileyriley\.com$", re.IGNORECASE)


def normalize_preview_url(url: str) -> str:
    """
    Map external TinyAuth-protected preview URLs to direct internal LAN preview URLs
    so headless Playwright captures the rendered UI instead of an SSO login screen.
    """
    raw = url.strip()
    if not raw.startswith(("http://", "https://", "file://")):
        raw = f"http://preview.lan/{raw.lstrip('/')}/"
    parsed = urllib.parse.urlsplit(raw)
    host = (parsed.hostname or "").lower()
    if host == "preview.wileyriley.com":
        path = parsed.path or "/"
        return urllib.parse.urlunsplit(("http", "preview.lan", path, parsed.query, parsed.fragment))
    m = _PORT_SUBDOMAIN_RE.match(host)
    if m:
        port = m.group(1)
        path = parsed.path or "/"
        return urllib.parse.urlunsplit(
            ("http", f"10.0.0.10:{port}", path, parsed.query, parsed.fragment)
        )
    return raw


@router.post("/screenshot", response_model=ScreenshotResponse)
async def capture_screenshot(request: ScreenshotRequest) -> ScreenshotResponse:
    """Capture a headless Chromium screenshot of a preview or local dev URL via Playwright."""
    rendered_url = normalize_preview_url(request.url)
    upload_dir = resolve_upload_dir()
    suffix = "mobile" if request.mobile else "desktop"
    out_path = os.path.join(upload_dir, f"preview_{int(time.time() * 1000)}_{suffix}.png")

    width = 390 if request.mobile else request.width
    height = 844 if request.mobile else request.height
    scale = 2 if request.mobile else 1

    pw_script = (
        "import json, sys\n"
        "from playwright.sync_api import sync_playwright\n"
        "cfg = json.loads(sys.argv[1])\n"
        "with sync_playwright() as p:\n"
        "    browser = p.chromium.launch(\n"
        "        headless=True,\n"
        "        args=['--host-resolver-rules=MAP *.wileyriley.com 10.0.0.10,MAP preview.lan 10.0.0.10,MAP preview 10.0.0.10']\n"
        "    )\n"
        "    ctx = browser.new_context(\n"
        "        viewport={'width': cfg['width'], 'height': cfg['height']},\n"
        "        device_scale_factor=cfg['scale'],\n"
        "        ignore_https_errors=True,\n"
        "    )\n"
        "    page = ctx.new_page()\n"
        "    page.goto(cfg['url'], wait_until='networkidle', timeout=20000)\n"
        "    if cfg['wait_ms'] > 0:\n"
        "        page.wait_for_timeout(cfg['wait_ms'])\n"
        "    page.screenshot(path=cfg['out_path'], full_page=cfg['full_page'])\n"
        "    browser.close()\n"
    )
    cfg_payload = json.dumps(
        {
            "url": rendered_url,
            "out_path": out_path,
            "width": width,
            "height": height,
            "scale": scale,
            "full_page": request.full_page,
            "wait_ms": request.wait_ms,
        }
    )

    python_bin = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else "python3"
    try:
        proc = await asyncio.create_subprocess_exec(
            python_bin,
            "-c",
            pw_script,
            cfg_payload,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            return ScreenshotResponse(
                success=False,
                rendered_url=rendered_url,
                error=err_msg or f"Playwright exited with {proc.returncode}",
            )
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        return ScreenshotResponse(
            success=True,
            path=out_path,
            rendered_url=rendered_url,
            size_bytes=size,
        )
    except Exception as e:
        return ScreenshotResponse(
            success=False,
            rendered_url=rendered_url,
            error=str(e),
        )

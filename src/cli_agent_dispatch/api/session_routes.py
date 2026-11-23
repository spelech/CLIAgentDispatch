import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
from typing import Any

from fastapi import APIRouter, HTTPException

from cli_agent_dispatch.config import settings
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
    T3HandoffRequest,
    T3HandoffResponse,
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
    if session_id.startswith("mcp:"):
        return ExecutorType.T3
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
        effort=req.effort,
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


@router.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str) -> dict[str, Any]:
    """Archives an active or ephemeral session and its underlying T3 thread if applicable."""
    record = await session_store.get(session_id)
    native_id = record.native_session_id if record else None
    target_exec = record.executor_type if record else None

    if not record:
        if session_id.startswith("mcp:"):
            native_id = session_id
            target_exec = ExecutorType.T3
        else:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    thread_id = native_id or session_id
    if target_exec == ExecutorType.T3 and thread_id:
        try:
            from cli_agent_dispatch.executors.t3 import T3McpClient

            async with T3McpClient() as client:
                await client.call_tool(
                    "t3_thread_organize",
                    {"threadId": thread_id, "action": "archive"},
                )
        except Exception as e:
            logger.error(f"Failed to archive T3 thread '{thread_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to archive T3 thread: {e}") from e

    await session_store.mark_archived(session_id)
    if native_id:
        await session_store.mark_archived(native_id)

    return {"status": "archived", "thread_id": thread_id}


def _build_t3_thread_ui_url(thread_id: str, raw_link: str | None = None) -> str:
    """Build a direct T3 Code UI link pointing to settings.t3_ui_url."""
    base = settings.t3_ui_url.rstrip("/")
    if raw_link:
        parsed = urllib.parse.urlsplit(raw_link)
        if parsed.path:
            return f"{base}{parsed.path}"
    encoded_tid = urllib.parse.quote(thread_id, safe="")
    return f"{base}/threads/{encoded_tid}"


def _build_handoff_prompt(
    *,
    full_prompt: str,
    last_user_turn: str,
    recent_steps: list[dict[str, Any]],
    aborted: bool,
    note: str | None = None,
) -> str:
    """Assemble a self-starting continuation prompt for T3 Code handoff."""
    sections: list[str] = [
        "=== OPENCLAW → T3 CODE HANDOFF ===",
        (
            "The user clicked 'Hand off to T3' while the previous OpenClaw agent turn was actively "
            "running. The previous turn was aborted so you can take over cleanly. Immediately "
            "continue and complete the user's request below without waiting for manual confirmation."
            if aborted
            else "The user handed off this OpenClaw session to T3 Code so they can continue here. "
            "Review the context below and immediately continue any remaining work (or concisely "
            "confirm current state and next step if the last turn already completed)."
        ),
    ]

    if recent_steps:
        step_lines: list[str] = []
        for st in recent_steps[-15:]:
            state = str(st.get("state") or "DONE").upper()
            tool_name = str(st.get("tool_name") or "tool")
            detail = str(st.get("detail") or "").strip()
            step_lines.append(
                f"- [{state}] {tool_name}: {detail}" if detail else f"- [{state}] {tool_name}"
            )
        sections.append("## Tool Steps Already Executed Before Handoff\n" + "\n".join(step_lines))

    if full_prompt:
        sections.append("## Conversation Context\n" + full_prompt[-30000:])
    elif last_user_turn:
        sections.append("## Latest User Request\n" + last_user_turn[-8000:])

    if note:
        sections.append("## Handoff Note\n" + note.strip())

    return "\n\n".join(sections)


@router.post("/sessions/{session_id}/handoff-t3", response_model=T3HandoffResponse)
async def handoff_session_to_t3(
    session_id: str,
    request: T3HandoffRequest | None = None,
) -> T3HandoffResponse:
    """
    Out-of-band handoff from an active or completed agy/OpenClaw session into a new T3 Code
    thread. Aborts any in-flight subprocess (when abort_active=True), immediately launches and
    starts the continuation prompt in T3 Code, and binds the client session to T3 for future turns.
    """
    req = request or T3HandoffRequest()
    record = await session_store.get_by_any_id(session_id)
    client_id = record.client_id if record else session_id

    # Idempotency: if already bound to a live T3 thread, return it directly
    if (
        record
        and record.executor_type == ExecutorType.T3
        and record.native_session_id
        and record.native_session_id.startswith("mcp:")
        and not record.archived
    ):
        existing_url = str(
            record.metadata.get("t3_url") or _build_t3_thread_ui_url(record.native_session_id)
        )
        return T3HandoffResponse(
            status="already_in_t3",
            session_id=client_id,
            thread_id=record.native_session_id,
            run_id=None,
            t3_url=existing_url,
            aborted=False,
            bound=True,
            model=str(record.metadata.get("t3_model") or settings.t3_default_model),
        )

    # 1. Abort any currently running subprocess for this session (never blocks OpenClaw loop)
    aborted = False
    if req.abort_active:
        candidate_ids = [session_id]
        if client_id not in candidate_ids:
            candidate_ids.append(client_id)
        if record and record.native_session_id and record.native_session_id not in candidate_ids:
            candidate_ids.append(record.native_session_id)
        for cid in candidate_ids:
            if await session_store.kill_active_process(cid):
                aborted = True

    # 2. Gather conversation context & recent tool steps
    meta = record.metadata if record else {}
    full_prompt = str(meta.get("last_full_prompt") or "")
    last_user_turn = str(meta.get("last_user_turn") or "")
    recent_steps = meta.get("recent_steps") if isinstance(meta.get("recent_steps"), list) else []
    workspace = req.workspace or str(meta.get("workspace") or settings.default_workspace)
    if workspace and not os.path.exists(workspace):
        workspace = os.getcwd()

    from cli_agent_dispatch.executors.t3 import (
        T3McpClient,
        _title_from_prompt,
        resolve_t3_model_selection,
    )

    instance_id, model_slug, model_options = resolve_t3_model_selection(
        req.model or "t3",
        effort=req.effort,
    )
    handoff_prompt = _build_handoff_prompt(
        full_prompt=full_prompt,
        last_user_turn=last_user_turn,
        recent_steps=recent_steps,
        aborted=aborted,
        note=req.note,
    )
    title = _title_from_prompt(last_user_turn or full_prompt or f"OpenClaw Handoff ({client_id})")

    # 3. Launch T3 thread immediately with the continuation prompt so it auto-starts
    try:
        async with T3McpClient() as client:
            project_id = await client.resolve_or_create_project(workspace)
            launched = await client.launch_thread(
                project_id=project_id,
                title=title,
                message=handoff_prompt,
                instance_id=instance_id,
                model_slug=model_slug,
                runtime_mode="full-access",
                interaction_mode="default",
                model_options=model_options,
            )
    except Exception as e:
        logger.error(f"Failed to hand off session '{client_id}' to T3 Code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to hand off session to T3: {e}") from e

    thread_id = str(launched.get("threadId") or "")
    run_id = str(launched.get("runId") or "") or None
    if not thread_id:
        raise HTTPException(status_code=500, detail="T3 launch did not return a threadId.")

    t3_url = _build_t3_thread_ui_url(thread_id, launched.get("link"))

    # 4. Bind the Slack/client session to T3 so future replies continue in this T3 thread
    await session_store.upgrade_executor(
        client_id=client_id,
        executor_type=ExecutorType.T3,
        native_session_id=thread_id,
        metadata={
            "t3_url": t3_url,
            "t3_model": model_slug,
            "handed_off_at": time.time(),
            "handed_off_from": str(record.executor_type) if record else "unknown",
        },
    )
    register_session_executor(thread_id, ExecutorType.T3)

    return T3HandoffResponse(
        status="handed_off",
        session_id=client_id,
        thread_id=thread_id,
        run_id=run_id,
        t3_url=t3_url,
        aborted=aborted,
        bound=True,
        model=f"{instance_id}/{model_slug}",
    )


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
        effort=request.effort,
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

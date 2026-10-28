import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.transport_security import TransportSecuritySettings

from cli_agent_dispatch.api.anthropic_routes import router as anthropic_router
from cli_agent_dispatch.api.openai_routes import router as openai_router
from cli_agent_dispatch.api.session_routes import router as session_router
from cli_agent_dispatch.api.t3_pair_routes import router as t3_pair_router
from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.models import ExecutorType
from cli_agent_dispatch.core.session_store import session_store
from cli_agent_dispatch.infrastructure.taps import diagnostic_buffer
from cli_agent_dispatch.mcp.server import mcp_server

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CLIAgentDispatch")


async def reap_expired_ephemeral_sessions(now: float | None = None) -> int:
    """Perform a single pass of archiving expired ephemeral sessions."""
    expired = await session_store.get_expired_ephemeral_sessions(now=now)
    archived_count = 0
    for rec in expired:
        try:
            thread_id = rec.native_session_id or rec.client_id
            if rec.executor_type == ExecutorType.T3 and thread_id:
                from cli_agent_dispatch.executors.t3 import T3McpClient

                async with T3McpClient() as client:
                    await client.call_tool(
                        "t3_thread_organize",
                        {"threadId": thread_id, "action": "archive"},
                    )
            await session_store.mark_archived(rec.client_id)
            archived_count += 1
            logger.info(f"Archived expired ephemeral session {rec.client_id} (thread: {thread_id})")
        except Exception as exc:
            logger.warning(
                f"Failed to archive ephemeral session {rec.client_id}: {exc}",
                exc_info=True,
            )
    return archived_count


async def reap_expired_ephemeral_sessions_loop(interval_seconds: float = 1800) -> None:
    """Background task running periodically during server lifespan to archive expired ephemeral sessions."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await reap_expired_ephemeral_sessions()
        except asyncio.CancelledError:
            logger.info("Ephemeral session reaper task cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in ephemeral reaper loop: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version} on port {settings.port}")
    reaper_task = asyncio.create_task(reap_expired_ephemeral_sessions_loop())
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except asyncio.CancelledError:
            pass
        logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Universal CLI & Headless AI Agent Delegation Gateway",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register OpenAI Compatibility Router (/v1/models, /v1/chat/completions)
app.include_router(openai_router)

# Register Anthropic Compatibility Router (/v1/messages)
app.include_router(anthropic_router)

# Register Interactive Sessions and SRE Investigation Router (/v1/investigate, /v1/sessions)
app.include_router(session_router)

# Register T3 Code SSO Pairing & ntfy Push Router (/sso-pair, /v1/t3/pair)
app.include_router(t3_pair_router)

# Mount MCP SSE Application (handles /sse and /messages)
transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
sse_app = mcp_server.sse_app(transport_security=transport_security)
app.mount("/mcp", sse_app)
app.mount("/sse", sse_app)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/api/diagnostics", tags=["Diagnostics"])
async def get_diagnostics(limit: int = 50):
    return {
        "events": diagnostic_buffer.get_events(limit=limit),
    }

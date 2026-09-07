import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.transport_security import TransportSecuritySettings

from cli_agent_dispatch.api.openai_routes import router as openai_router
from cli_agent_dispatch.config import settings
from cli_agent_dispatch.infrastructure.taps import diagnostic_buffer
from cli_agent_dispatch.mcp.server import mcp_server

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CLIAgentDispatch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} on port {settings.port}"
    )
    yield
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

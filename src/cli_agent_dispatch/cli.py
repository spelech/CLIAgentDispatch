import asyncio
import sys

import typer
import uvicorn

from cli_agent_dispatch.config import settings
from cli_agent_dispatch.core.engine import delegate_task

app = typer.Typer(
    name="cli-agent-dispatch",
    help="Universal CLI & Headless AI Agent Delegation Engine",
)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt or task instructions for the agent"),
    executor: str = typer.Option(
        "opencode", "--executor", "-e", help="Executor engine ('opencode' or 'agy')"
    ),
    workspace: str = typer.Option(
        settings.default_workspace,
        "--workspace",
        "-w",
        help="Target workspace path",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model override ID"),
    timeout: int = typer.Option(
        settings.default_timeout, "--timeout", "-t", help="Timeout in seconds"
    ),
) -> None:
    """Run an agent task from the command line."""
    typer.echo(f"Dispatching task to {executor}...")
    result = asyncio.run(
        delegate_task(
            prompt=prompt,
            executor=executor,
            workspace=workspace,
            model=model,
            timeout=timeout,
        )
    )
    if result.success:
        typer.echo(result.output)
    else:
        typer.secho(f"Execution Failed: {result.error}", fg=typer.colors.RED, err=True)
        if result.output:
            typer.echo(result.output)
        sys.exit(1)


@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Bind host address"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn hot reload"),
) -> None:
    """Start the FastAPI HTTP + FastMCP server."""
    uvicorn.run(
        "cli_agent_dispatch.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()

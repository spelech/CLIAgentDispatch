"""SRE investigation prompt builder and MCG safety directives."""

from cli_agent_dispatch.core.models import InvestigationRequest

MCG_READ_ONLY_DIRECTIVE: str = """=== CRITICAL MODEL CONTEXT GATEWAY (MCG) SAFETY DIRECTIVE ===
You have access to Model Context Gateway tools (via `mcg`).
You are STRICTLY CONFINED to READ-ONLY inspection when using MCG tools.
ALLOWED MCG ACTIONS:
- search_tools
- execute_tool for queries, read-only status inspections, reading notes, and diagnostic queries.
FORBIDDEN MCG ACTIONS:
- DO NOT execute mutating, editing, restarting, creating, or deleting tool calls via MCG.
- All proposed fixes must be returned as clean bash commands in your structured JSON response.
"""


def build_investigation_prompt(request: InvestigationRequest) -> str:
    """Constructs the initial SRE investigation prompt with incident context and safety directives."""
    exit_code_str = str(request.exit_code) if request.exit_code is not None else "Unknown"
    workspace_str = request.workspace or "Default / System"
    error_logs = request.error_logs.strip() if request.error_logs else "No error logs provided."

    return f"""{MCG_READ_ONLY_DIRECTIVE}

=== SRE INCIDENT INVESTIGATION REQUEST ===
Target Service / Container: {request.target}
Exit Code / Status: {exit_code_str}
Workspace: {workspace_str}

=== ERROR LOGS & DIAGNOSTIC CONTEXT ===
{error_logs}

=== INVESTIGATION OBJECTIVE & INSTRUCTIONS ===
1. Investigate the failure of target '{request.target}'.
2. You may perform read-only diagnostic inspections (using available tools or MCG read-only tools).
3. Identify the true root cause of the incident.
4. Formulate the precise remediation fix as executable bash commands.
5. DO NOT apply destructive or mutating actions directly.

=== REQUIRED OUTPUT FORMAT ===
When you conclude your investigation, you MUST output a JSON object (either directly or enclosed in a ```json code block) adhering to the following schema:
```json
{{
  "root_cause": "<concise explanation of why the target failed>",
  "proposed_fix": "<pure executable bash command(s) to remediate the issue>",
  "category": "<issue category, e.g. configuration, network, dependency, hardware, transient, resource, storage, permission>",
  "notes": "<any additional context or observations>"
}}
```
Important: The "proposed_fix" field must contain clean, executable bash command(s) that can be run to resolve the failure.
"""

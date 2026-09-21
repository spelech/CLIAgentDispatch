import json
import logging
import re
import time
from typing import Any

from cli_agent_dispatch.core.exceptions import ExecutorNotFoundError
from cli_agent_dispatch.core.models import (
    ExecutorType,
    InvestigationRequest,
    InvestigationResult,
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

logger = logging.getLogger("CLIAgentDispatch.Engine")


def _parse_investigation_output(text: str) -> tuple[bool, dict[str, Any]]:
    """Attempts to extract a structured resolution JSON object from model output."""
    if not text:
        return False, {}

    # 1. Check markdown code blocks first
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    for block in code_blocks:
        block = block.strip()
        try:
            data = json.loads(block)
            if isinstance(data, dict) and ("root_cause" in data or "proposed_fix" in data):
                return True, data
        except Exception:
            pass

    # 2. Check for raw JSON object in text
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            data, end_idx = decoder.raw_decode(text[match:])
            if isinstance(data, dict) and ("root_cause" in data or "proposed_fix" in data):
                return True, data
            pos = match + max(1, end_idx)
        except Exception:
            pos = match + 1

    return False, {}


def _extract_fallback_fields(text: str) -> dict[str, str]:
    """Gracefully extracts best-effort fields from unstructured text output."""
    if not text:
        return {"root_cause": "", "proposed_fix": "", "category": "unknown"}

    root_cause = ""
    proposed_fix = ""
    category = "unknown"

    # Match Root cause: ...
    rc_match = re.search(
        r"(?:root\s*cause|cause|issue)[:\-]\s*(.*?)(?=\n(?:proposed|fix|category|notes)|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if rc_match:
        root_cause = rc_match.group(1).strip()

    # Match Proposed fix: ...
    fix_match = re.search(
        r"(?:proposed\s*fix|remediation|fix)[:\-]\s*(.*?)(?=\n(?:root|category|notes)|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if fix_match:
        proposed_fix = fix_match.group(1).strip()
    else:
        bash_match = re.search(r"```(?:bash|sh)?\s*(.*?)\s*```", text, re.DOTALL)
        if bash_match:
            proposed_fix = bash_match.group(1).strip()

    # Match Category: ...
    cat_match = re.search(
        r"category[:\-]\s*([a-zA-Z0-9_\-]+)",
        text,
        re.IGNORECASE,
    )
    if cat_match:
        category = cat_match.group(1).strip().lower()

    if not root_cause:
        root_cause = text.strip()[:300]

    return {
        "root_cause": root_cause,
        "proposed_fix": proposed_fix,
        "category": category,
    }


class DispatchEngine:
    """Central orchestration engine managing executors and routing task requests."""

    def __init__(self) -> None:
        self._executors: dict[str, BaseExecutor] = {
            ExecutorType.OPENCODE.value: OpenCodeExecutor(),
            ExecutorType.AGY.value: AgyExecutor(),
        }

    def register_executor(self, executor: BaseExecutor) -> None:
        self._executors[executor.name] = executor

    def get_executor(self, name: str) -> BaseExecutor:
        executor = self._executors.get(name.lower())
        if not executor:
            raise ExecutorNotFoundError(
                f"Executor '{name}' not found. Available: {list(self._executors.keys())}"
            )
        return executor

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """Route and execute a task request against the target executor."""
        executor = self.get_executor(request.executor.value)
        logger.info(
            f"Dispatching task to executor='{executor.name}', model='{request.model}', "
            f"workspace='{request.workspace}'"
        )
        result = await executor.execute(request)
        if result.success:
            logger.info(f"Task succeeded via '{executor.name}' in {result.duration_seconds}s")
        else:
            logger.warning(
                f"Task failed via '{executor.name}' in {result.duration_seconds}s: {result.error}"
            )
        return result

    async def investigate(self, request: InvestigationRequest) -> InvestigationResult:
        """Executes an interactive SRE investigation session with structured resolution."""
        start_time = time.perf_counter()
        session_id = request.session_id
        transcript: list[dict[str, Any]] = []
        turns_used = 0
        raw_output = ""
        executor_name = (
            request.executor.value
            if isinstance(request.executor, ExecutorType)
            else str(request.executor)
        )

        current_prompt = build_investigation_prompt(request)
        system_prompt = MCG_READ_ONLY_DIRECTIVE

        max_turns = max(1, request.max_turns)
        for turn_idx in range(1, max_turns + 1):
            turns_used = turn_idx
            task_req = TaskRequest(
                prompt=current_prompt,
                session_id=session_id,
                system_prompt=system_prompt if turn_idx == 1 and not session_id else None,
                executor=request.executor,
                workspace=request.workspace,
                model=request.model,
                provider=request.provider,
                timeout=request.timeout,
            )

            logger.info(
                f"SRE investigation turn {turn_idx}/{max_turns} for target='{request.target}', "
                f"executor='{executor_name}', session_id='{session_id}'"
            )

            transcript.append(
                {
                    "role": "user",
                    "content": current_prompt,
                    "turn": turn_idx,
                }
            )

            result = await self.dispatch(task_req)
            raw_output = result.output or ""
            if result.session_id:
                session_id = result.session_id

            transcript.append(
                {
                    "role": "assistant",
                    "content": raw_output,
                    "turn": turn_idx,
                    "error": result.error,
                }
            )

            if not result.success:
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    f"SRE investigation failed on turn {turn_idx} for target='{request.target}': {result.error}"
                )
                return InvestigationResult(
                    success=False,
                    session_id=session_id,
                    target=request.target,
                    executor=executor_name,
                    error=result.error or "Dispatch execution failed",
                    turns_used=turns_used,
                    duration_seconds=round(elapsed, 3),
                    transcript=transcript,
                    raw_output=raw_output,
                )

            parsed_ok, data = _parse_investigation_output(raw_output)
            if parsed_ok:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"SRE investigation concluded successfully on turn {turn_idx} for target='{request.target}'"
                )
                return InvestigationResult(
                    success=True,
                    session_id=session_id,
                    target=request.target,
                    executor=executor_name,
                    root_cause=str(data.get("root_cause", "")),
                    proposed_fix=str(data.get("proposed_fix", "")),
                    category=str(data.get("category", "unknown")),
                    turns_used=turns_used,
                    duration_seconds=round(elapsed, 3),
                    transcript=transcript,
                    raw_output=raw_output,
                )

            if turn_idx < max_turns:
                current_prompt = (
                    "Please provide your final root cause diagnosis and remediation fix in the required JSON format:\n"
                    "```json\n"
                    "{\n"
                    '  "root_cause": "<concise explanation of why the target failed>",\n'
                    '  "proposed_fix": "<pure executable bash command(s) to remediate the issue>",\n'
                    '  "category": "<issue category>",\n'
                    '  "notes": "<any additional context or observations>"\n'
                    "}\n"
                    "```"
                )

        # Reached max turns without structured JSON; fall back gracefully
        elapsed = time.perf_counter() - start_time
        fallback_data = _extract_fallback_fields(raw_output)
        logger.info(
            f"SRE investigation completed using fallback extraction for target='{request.target}' after {turns_used} turns"
        )
        return InvestigationResult(
            success=True,
            session_id=session_id,
            target=request.target,
            executor=executor_name,
            root_cause=fallback_data["root_cause"],
            proposed_fix=fallback_data["proposed_fix"],
            category=fallback_data["category"],
            turns_used=turns_used,
            duration_seconds=round(elapsed, 3),
            transcript=transcript,
            raw_output=raw_output,
        )


# Singleton instance
engine = DispatchEngine()


async def delegate_task(
    prompt: str,
    executor: str = "opencode",
    workspace: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
) -> TaskResult:
    """Convenience functional helper for external Python callers (e.g. monitorbot)."""
    req = TaskRequest(
        prompt=prompt,
        executor=ExecutorType(executor.lower()),
        workspace=workspace,
        model=model,
        provider=provider,
        timeout=timeout,
    )
    return await engine.dispatch(req)


async def investigate_target(
    target: str,
    exit_code: str | int | None = None,
    error_logs: str = "",
    workspace: str | None = None,
    executor: str = "opencode",
    max_turns: int = 5,
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
    session_id: str | None = None,
) -> InvestigationResult:
    """Convenience functional helper for SRE investigation calls (e.g. monitorbot)."""
    req = InvestigationRequest(
        target=target,
        exit_code=exit_code,
        error_logs=error_logs,
        workspace=workspace,
        executor=ExecutorType(executor.lower()),
        max_turns=max_turns,
        model=model,
        provider=provider,
        timeout=timeout,
        session_id=session_id,
    )
    return await engine.investigate(req)

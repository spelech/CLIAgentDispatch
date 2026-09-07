from unittest.mock import AsyncMock

import pytest

from cli_agent_dispatch.core.engine import DispatchEngine
from cli_agent_dispatch.core.models import ExecutorType, TaskRequest, TaskResult


class AgentFeedbackEnvelope:
    @staticmethod
    def format_failure(
        inputs: dict,
        active_settings: dict,
        action_history: list[str],
        output_delta: dict,
        captured_logs: list[str],
        reproduction_command: str,
    ) -> str:
        return (
            f"\n--- AGENT FEEDBACK ENVELOPE ---\n"
            f"Inputs: {inputs}\n"
            f"Active Settings: {active_settings}\n"
            f"Action History: {action_history}\n"
            f"Output Delta: {output_delta}\n"
            f"Captured Logs: {captured_logs}\n"
            f"Reproduction: {reproduction_command}\n"
            f"-------------------------------"
        )


@pytest.mark.asyncio
async def test_simulation_high_volume_stress_loop():
    """Simulate rapid burst delegations under concurrent load."""
    engine = DispatchEngine()
    history = []

    mock_exec = AsyncMock()
    mock_exec.name = "opencode"
    mock_exec.execute.return_value = TaskResult(
        success=True, executor="opencode", output="batch ok", duration_seconds=0.01
    )
    engine.register_executor(mock_exec)

    for i in range(50):
        req = TaskRequest(prompt=f"Task #{i}", executor=ExecutorType.OPENCODE)
        history.append(f"dispatch_{i}")
        res = await engine.dispatch(req)
        assert res.success is True, AgentFeedbackEnvelope.format_failure(
            inputs={"iteration": i},
            active_settings={"engine": "opencode"},
            action_history=history[-5:],
            output_delta={"expected": True, "actual": res.success},
            captured_logs=[str(res.error)],
            reproduction_command="pytest tests/test_simulation_harness.py",
        )


@pytest.mark.asyncio
async def test_simulation_disturbance_injection():
    """Inject malformed and abrupt failures to test recovery."""
    engine = DispatchEngine()

    mock_exec = AsyncMock()
    mock_exec.name = "agy"
    mock_exec.execute.return_value = TaskResult(
        success=False,
        executor="agy",
        output="",
        error="SIGKILL received during process execution",
        duration_seconds=0.05,
    )
    engine.register_executor(mock_exec)

    req = TaskRequest(prompt="Crash test", executor=ExecutorType.AGY)
    res = await engine.dispatch(req)

    assert res.success is False
    assert "SIGKILL" in res.error

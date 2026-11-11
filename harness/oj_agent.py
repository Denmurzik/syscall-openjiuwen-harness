"""The harness as an openJiuwen agent.

`HarnessAgent` is an `openjiuwen.core.single_agent.BaseAgent` with an `AgentCard`; `invoke` solves a task set
(`scheduler.run`). Its model calls go through openJiuwen's `Model` (`harness.llm`), and what it runs for the model
(a program in the sandbox, a test file against a candidate) are openJiuwen `LocalFunction` tools with `ToolCard`s
registered in the agent's ability manager.

Nothing here decides anything: the planner, the solvers and the budget are the harness's own. Every use of the
framework layer falls back to the plain call (`HARNESS_OJ_AGENT=0` turns the layer off altogether), so a failure
in it can never cost a task.
"""
import asyncio
import contextvars
import logging
import os
from pathlib import Path

log = logging.getLogger("harness")

AGENT_ID = "syscall-harness"
_TOOLS: dict = {}   # name -> openJiuwen tool, filled by build_agent(); empty: the plain functions are called
# what the tool's own function did in the current call_tool: {"res": [result]} or {"exc": exception}
_CALL: contextvars.ContextVar = contextvars.ContextVar("harness_oj_call", default=None)


def enabled() -> bool:
    return (os.environ.get("HARNESS_OJ_AGENT") or "1").strip().lower() not in ("0", "false", "no", "off")


async def _recorded(awaitable):
    """Await the tool's own work and note in call_tool's holder how it ended: once the function has run, a failure
    of the framework around it never runs the unit a second time, and an error of the function itself comes out
    as the plain call would raise it."""
    holder = _CALL.get()
    try:
        res = await awaitable
    except Exception as exc:
        if holder is not None:
            holder["exc"] = exc
        raise
    if holder is not None:
        holder["res"] = [res]
    return res


async def _run_program(code: str, workspace: str):
    from harness.sandbox import run_program
    return await _recorded(asyncio.to_thread(run_program, code, Path(workspace)))   # up to 20 s: off the event loop


async def _run_tests(workspace: str, source: str, target: str | None = None):
    from harness.llm_routes import _run_candidate
    # the framework fills a parameter the caller left out with None
    return await _recorded(asyncio.to_thread(_run_candidate, Path(workspace), source, target or "mylib.py"))


_SPECS = {
    "run_program": (_run_program, "Run a Python program in a throw-away copy of the task workspace; returns its "
                                  "stdout, stderr and exit code.",
                    {"type": "object", "properties": {"code": {"type": "string"}, "workspace": {"type": "string"}},
                     "required": ["code", "workspace"]}),
    "run_tests": (_run_tests, "Run the task's test file against a candidate text of the module under repair, in a "
                              "throw-away copy of the workspace; returns the failed tests and the output.",
                  {"type": "object", "properties": {"workspace": {"type": "string"}, "source": {"type": "string"},
                                                    "target": {"type": "string"}},
                   "required": ["workspace", "source"]}),
}


def build_agent(client):
    """The agent with its tools; raises when the framework layer cannot be built (the caller then runs the
    scheduler directly)."""
    from harness.llm import _quiet_openjiuwen_logs
    _quiet_openjiuwen_logs()
    from openjiuwen.core.foundation.tool.base import ToolCard
    from openjiuwen.core.foundation.tool.function.function import LocalFunction
    from openjiuwen.core.single_agent import AgentCard, BaseAgent

    class HarnessAgent(BaseAgent):
        """Budget-aware solver of a task set: free deterministic solvers first, then budgeted model routes."""

        def __init__(self, llm_client):
            super().__init__(AgentCard(id=AGENT_ID, name="SysCallHarness",
                                       description="Solves a directory of tasks within a token budget and writes "
                                                   "answer.txt for every task it can answer."))
            self.client = llm_client
            self.tools: dict = {}
            self.started = False
            self.result = None
            for name, (func, description, params) in _SPECS.items():
                card = ToolCard(id=f"{AGENT_ID}.{name}", name=name, description=description, input_params=params)
                self.tools[name] = LocalFunction(card, func)
                self.ability_manager.add(card)

        def configure(self, config):
            self._config = config
            return self

        async def invoke(self, inputs, session=None):
            from harness import scheduler
            self.started = True
            # the scheduler owns its own event loop for phase 2: it runs in a worker thread of this one
            self.result = await asyncio.to_thread(scheduler.run, Path(inputs["tasks_dir"]), float(inputs["budget"]),
                                                  self.client)
            return self.result

        async def stream(self, inputs, session=None, stream_modes=None):
            yield await self.invoke(inputs, session)

    agent = HarnessAgent(client)
    _TOOLS.clear()
    _TOOLS.update(agent.tools)
    return agent


def run(tasks_dir: Path, budget: float, client) -> dict | None:
    """Solve the task set as the openJiuwen agent. None: the layer is off or could not start, and nothing has run
    (the caller runs the scheduler itself). Once the scheduler has started its result is returned whatever the
    framework layer does around it: a task set is never solved twice."""
    if not enabled():
        return None
    try:
        log_dir = os.environ.get("HARNESS_LOG_DIR")
        if log_dir and os.path.isdir(log_dir):
            os.chdir(log_dir)   # openjiuwen writes ./logs relative to the cwd; /app is read-only in the jury's run
        agent = build_agent(client)
    except Exception as exc:
        log.warning("openJiuwen agent layer unavailable (%s: %s); running the scheduler directly",
                    type(exc).__name__, exc)
        _TOOLS.clear()
        return None
    try:
        return asyncio.run(agent.invoke({"tasks_dir": str(tasks_dir), "budget": budget, "user_input": str(tasks_dir),
                                         "session_id": AGENT_ID}))
    except Exception as exc:
        log.warning("openJiuwen agent layer failed (%s: %s)", type(exc).__name__, exc)
        if agent.started:
            return agent.result if agent.result is not None else {}
        _TOOLS.clear()
        return None


async def call_tool(name: str, plain, *args):
    """Run tool `name` through its openJiuwen wrapper; `plain(*args)` in a thread when the layer is off or the
    framework fails before the tool's function ran. Once the function has run, its own result or its own error
    is what comes out (the unit is never run twice). `args` follow the order of the tool's parameters."""
    tool = _TOOLS.get(name)
    if tool is not None:
        holder: dict = {}
        token = _CALL.set(holder)
        try:
            keys = list(_SPECS[name][2]["properties"])
            inputs = {k: (str(v) if isinstance(v, Path) else v) for k, v in zip(keys, args)}
            out = await tool.invoke(inputs)
            if "exc" in holder:   # the function raised and the framework swallowed it: raise it as the plain call
                raise holder["exc"]
            return holder["res"][0] if "res" in holder else out
        except Exception as exc:
            if "exc" in holder:   # the function's own error (OSError of a full disk...): never run it twice
                raise holder["exc"]
            if "res" in holder:   # the function finished; only the framework after it failed
                return holder["res"][0]
            log.warning("openJiuwen tool %s failed (%s: %s); plain call", name, type(exc).__name__, str(exc)[:120])
        finally:
            _CALL.reset(token)
    return await asyncio.to_thread(plain, *args)

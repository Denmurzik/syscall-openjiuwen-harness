"""Expert pass 8: task shapes far from the examples (plain test functions, another module name, `import *`, prose
knowledge bases), the hop check of batched 2a specs, and the harness as an openJiuwen agent."""
import asyncio
import json

from harness import json_route, llm_routes as L, oj_agent
from harness.llm import ChatResult
from harness.result import SolveResult
from harness.solvers import probe
from harness.solvers.testrun import run_tests
from harness.tasks import Task

INS = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
       "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the "
       "word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost=6)


def _ws(tmp_path, files, name="t"):
    ws = tmp_path / name / "workspace"
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return ws


# ------------------------------------------------------------------ the harness as an openJiuwen agent
def test_the_agent_is_an_openjiuwen_agent_with_tools(tmp_path, monkeypatch):
    from openjiuwen.core.foundation.tool.base import Tool
    from openjiuwen.core.single_agent import BaseAgent
    agent = oj_agent.build_agent(None)
    try:
        assert isinstance(agent, BaseAgent) and agent.card.id == oj_agent.AGENT_ID
        assert set(agent.tools) == {"run_program", "run_tests"} and all(isinstance(t, Tool) for t in agent.tools.values())
        seen = {}

        def fake_run(tasks_dir, budget, client):
            seen["args"] = (tasks_dir, budget, client)
            return {"solved": 1, "spent": 0.0}
        from harness import scheduler
        monkeypatch.setattr(scheduler, "run", fake_run)
        out = asyncio.run(agent.invoke({"tasks_dir": str(tmp_path), "budget": 100.0}))
        assert out == {"solved": 1, "spent": 0.0} and seen["args"] == (tmp_path, 100.0, None)
    finally:
        oj_agent._TOOLS.clear()


def test_tools_run_through_openjiuwen_and_give_the_same_result(tmp_path):
    from harness.sandbox import run_program
    ws = _ws(tmp_path, {"data.txt": "7\n"})
    plain = run_program("print(int(open('data.txt').read()) * 6)", ws)
    oj_agent.build_agent(None)
    try:
        via = asyncio.run(oj_agent.call_tool("run_program", run_program, "print(int(open('data.txt').read()) * 6)", ws))
    finally:
        oj_agent._TOOLS.clear()
    assert (via.stdout, via.returncode, via.timed_out) == (plain.stdout, plain.returncode, plain.timed_out) and via.stdout.strip() == "42"


def test_call_tool_without_the_layer_is_the_plain_call(tmp_path):
    oj_agent._TOOLS.clear()
    assert asyncio.run(oj_agent.call_tool("run_program", lambda code, ws: (code, ws), "x", tmp_path)) == ("x", tmp_path)


def test_a_broken_layer_falls_back_to_the_plain_call(tmp_path):
    class Broken:
        async def invoke(self, inputs):
            raise RuntimeError("framework")
    oj_agent._TOOLS["run_program"] = Broken()
    try:
        assert asyncio.run(oj_agent.call_tool("run_program", lambda code, ws: "plain", "x", tmp_path)) == "plain"
    finally:
        oj_agent._TOOLS.clear()


def test_the_ladder_runs_its_tests_through_the_tool_layer(tmp_path):
    """The Docker run of this pass: with two arguments the framework passed target=None and every ladder died."""
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_d(self):\n        self.assertEqual(m.double(2), 4)\n")
    ws = _ws(tmp_path, {"mylib.py": "def double(x):\n    return x + 1\n", "test_mylib.py": tests})
    task = Task("t", "coding", INS, 2.0, "unittest", tmp_path / "t", ws)
    oj_agent.build_agent(None)
    try:
        rep = asyncio.run(oj_agent.call_tool("run_tests", L._run_candidate, ws, "def double(x):\n    return x * 2\n"))
        assert rep.ok
        res = asyncio.run(L.fix_coding(task, Fake(["FUNC double\ndef double(x):\n    return x * 2\n"])))
    finally:
        oj_agent._TOOLS.clear()
    assert res.status == "solved" and "x * 2" in (ws / "mylib.py").read_text(encoding="utf-8")


def test_run_returns_none_when_the_layer_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_OJ_AGENT", "0")
    assert oj_agent.run(tmp_path, 100.0, None) is None


def test_run_never_solves_a_task_set_twice(tmp_path, monkeypatch):
    from harness import scheduler
    calls = []

    def fake_run(tasks_dir, budget, client):
        calls.append(1)
        return {"solved": 3}
    monkeypatch.setattr(scheduler, "run", fake_run)
    monkeypatch.delenv("HARNESS_OJ_AGENT", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        assert oj_agent.run(tmp_path, 100.0, None) == {"solved": 3} and calls == [1]
    finally:
        oj_agent._TOOLS.clear()

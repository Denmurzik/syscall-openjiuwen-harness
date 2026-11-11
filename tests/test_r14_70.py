"""Round 14, task 70 (expert 7, idea d): the coding ladder tells the model that the test run was stopped on time."""
import asyncio

from harness import llm_routes as L
from harness.llm import ChatResult
from harness.solvers.testrun import TestReport
from harness.tasks import Task

INS = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
       "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the "
       "word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
LIB = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_fib(self):\n        self.assertEqual(m.fib(60), 1548008755920)\n")
FIX = "FUNC fib\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost=6)


def _task(tmp_path, lib=LIB, tests=TESTS) -> Task:
    ws = tmp_path / "t" / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("t", "coding", INS, 3.0, "unittest", tmp_path / "t", ws)


def test_shown_output_adds_the_note_only_for_a_timeout():
    assert L.TIMEOUT_HINT in L._shown_output(TestReport(False, ["<timeout>"], ""))
    assert L._shown_output(TestReport(False, ["<timeout>"], "partial")).startswith("partial")
    assert L._shown_output(TestReport(False, ["test_a"], "FAIL: test_a")) == "FAIL: test_a"
    assert L._shown_output(TestReport(False, ["<unknown>"], "boom")) == "boom"


def test_the_ladder_prompt_says_the_run_was_stopped(tmp_path, monkeypatch):
    task = _task(tmp_path)

    def fake_run(ws_, text):   # the slow text "times out" at once, the fixed one passes
        return TestReport(False, ["<timeout>"], "") if "fib(n - 1)" in text else TestReport(True, [], "OK")
    monkeypatch.setattr(L, "_run_candidate", fake_run)
    client = Fake([FIX])
    res = asyncio.run(L.fix_coding(task, client))
    assert res.status == "solved" and L.TIMEOUT_HINT in client.calls[0][1]["content"]
    assert "a, b = b, a + b" in (task.workspace / "mylib.py").read_text(encoding="utf-8")


def test_no_note_without_a_timeout(tmp_path, monkeypatch):
    task = _task(tmp_path)

    def fake_run(ws_, text):
        return TestReport(False, ["test_fib"], "FAIL: test_fib") if "fib(n - 1)" in text else TestReport(True, [], "OK")
    monkeypatch.setattr(L, "_run_candidate", fake_run)
    client = Fake([FIX])
    assert asyncio.run(L.fix_coding(task, client)).status == "solved"
    assert L.TIMEOUT_HINT not in client.calls[0][1]["content"]


def test_a_candidate_that_hangs_does_not_mislead_the_next_rung(tmp_path, monkeypatch):
    # the base fails plainly; the first model's candidate hangs. The next rung is asked about the base again and
    # is told the base's own failure, not "the run was stopped by the time limit"
    task = _task(tmp_path)
    hang = "FUNC fib\ndef fib(n):\n    while True:\n        pass\n"

    def fake_run(ws_, text):
        if "while True" in text:
            return TestReport(False, ["<timeout>"], "")
        if "fib(n - 1)" in text:
            return TestReport(False, ["test_fib"], "FAIL: test_fib base")
        return TestReport(True, [], "OK")
    monkeypatch.setattr(L, "_run_candidate", fake_run)
    client = Fake([hang, FIX])
    assert asyncio.run(L.fix_coding(task, client)).status == "solved"
    second = client.calls[1][1]["content"]
    assert "FAIL: test_fib base" in second and L.TIMEOUT_HINT not in second

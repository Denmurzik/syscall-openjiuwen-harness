import asyncio
from pathlib import Path

from harness.arbitrate import arbitrate, parse_choice
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.result import SolveResult
from harness.solvers.coding_solver import solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task

LIB = "def count_even(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"
TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
         "    def test_count(self):\n        self.assertEqual(mylib.count_even([1, 2, 4, 6, 7]), 3)\n")


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra": extra})
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=1, cost=2)


def _solved(tmp_path):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(LIB, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(TESTS, encoding="utf-8")
    task = Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")
    res = solve(task)
    assert res.tentative and res.extra["ambiguous"]
    return task, res.extra


def run(c):
    return asyncio.run(c)


def _opt_with(extra, needle):
    return next(i for i, c in enumerate(extra["ambiguous"][0]["candidates"]) if needle in c) + 1


def test_picks_other_option_and_rewrites_mylib(tmp_path):
    task, extra = _solved(tmp_path)
    want = _opt_with(extra, "x % 2 == 0")
    c = Fake([str(want)])
    res = run(arbitrate(task, extra, c))
    assert res.status == "solved" and res.answer == "DONE" and not res.tentative
    assert len(c.calls) == 1 and c.calls[0]["max_tokens"] <= 8
    n = len(extra["ambiguous"][0]["candidates"])
    assert c.calls[0]["extra"] == {"extra_body": {"guided_regex": "[1-%d]" % n}}
    assert "x % 2 == 0" in (task.workspace / "mylib.py").read_text(encoding="utf-8")
    assert run_tests(task.workspace).ok


def test_picks_current_choice_leaves_file_untouched(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    res = run(arbitrate(task, extra, Fake([str(extra["ambiguous"][0]["chosen"] + 1)])))
    assert res.status == "solved" and not res.tentative
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_400_falls_back_without_structured_output(tmp_path):
    task, extra = _solved(tmp_path)
    want = _opt_with(extra, "x % 2 == 0")
    c = Fake([LLMUnavailable("BadRequestError: Error code: 400 - guided_regex"), f"Answer: {want}"])
    res = run(arbitrate(task, extra, c))
    assert "extra_body" in c.calls[0]["extra"] and c.calls[1]["extra"] == {}
    assert "x % 2 == 0" in (task.workspace / "mylib.py").read_text(encoding="utf-8")
    assert getattr(c, "structured_output_refused", False)
    assert res.status == "solved"


def test_garbage_then_garbage_changes_nothing(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    c = Fake(["???", "no idea"])
    res = run(arbitrate(task, extra, c))
    assert res.status == "solved" and len(c.calls) == 2
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_out_of_range_number_changes_nothing(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    run(arbitrate(task, extra, Fake(["9", "9"])))
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_option_failing_tests_is_not_written(tmp_path):
    task, extra = _solved(tmp_path)
    extra["ambiguous"][0]["candidates"].append("def count_even(xs):\n    return 0")
    before = (task.workspace / "mylib.py").read_bytes()
    n = len(extra["ambiguous"][0]["candidates"])
    res = run(arbitrate(task, extra, Fake([str(n)])))
    assert res.status == "solved"
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_unavailable_is_unparsed_and_budget_propagates(tmp_path):
    task, extra = _solved(tmp_path)
    assert run(arbitrate(task, extra, Fake([LLMUnavailable("HTTP 503")]))).status == "unparsed"
    try:
        run(arbitrate(task, extra, Fake([BudgetExhausted("x")])))
        assert False
    except BudgetExhausted:
        pass


def test_parse_choice():
    items = [{"candidates": ["a", "b"]}, {"candidates": ["a", "b", "c"]}]
    assert parse_choice("2,3", items) == [1, 2]
    assert parse_choice("1", items) is None
    assert parse_choice("1,4", items) is None

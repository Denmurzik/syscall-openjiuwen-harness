import asyncio
from pathlib import Path

import pytest

from harness.budget import TokenLedger
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.llm_routes import (apply_patch, extract_code, fix_coding, normalize_answer,
                                solve_with_program)
from harness.tasks import Task, copy_task

FIX = Path(__file__).parent / "fixtures" / "examples"


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                           "extra": extra})
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=10, cost=11)


def _task(tmp_path, check="exact_int", files=None, reward=3.0, ttype="data_analysis"):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    for n, c in (files or {}).items():
        (d / "workspace" / n).write_text(c, encoding="utf-8")
    return Task("t", ttype, "How many?", reward, check, d, d / "workspace")


def _fixture(name, tmp_path, reward=2.0):
    d = FIX / name
    return copy_task(Task(name, "coding", "fix", reward, "unittest", d, d / "workspace"), tmp_path)


def run(coro):
    return asyncio.run(coro)


def test_program_route_writes_last_line(tmp_path):
    c = FakeClient(["```python\nprint('x')\nprint(42)\n```"])
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "42"
    assert c.calls[0]["model"] == "qwen3.6-35b-a3b-noreason" and c.calls[0]["max_tokens"] == 400


def test_program_route_impossible(tmp_path):
    r = run(solve_with_program(_task(tmp_path), FakeClient(["print('IMPOSSIBLE')"])))
    assert r.status == "infeasible"


def test_program_route_crash_retries_then_unparsed(tmp_path):
    c = FakeClient(["import sys\nsys.stderr.write('boom_marker')\nraise SystemExit(3)"] * 2)
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "unparsed" and len(c.calls) == 2
    assert "boom_marker" in str(c.calls[1]["messages"])


def test_program_runs_in_workspace_copy(tmp_path):
    t = _task(tmp_path, files={"sales.csv": "a,b\n1,2\n3,4\n"})
    code = ("import csv\nrows=list(csv.DictReader(open('sales.csv')))\n"
            "open('junk.txt','w').write('x')\nprint(sum(int(r['b']) for r in rows))")
    c = FakeClient([code])
    r = run(solve_with_program(t, c))
    assert r.status == "solved" and r.answer == "6"
    assert not (t.workspace / "junk.txt").exists()
    assert "sales.csv" in str(c.calls[0]["messages"]) and "a,b" in str(c.calls[0]["messages"])


def test_normalize_answer():
    assert normalize_answer("389.0\n", "exact_int") == "389"
    assert normalize_answer("Answer: 'Osaka'", "exact_str") == "Osaka"
    assert normalize_answer("", "exact_int") is None
    assert normalize_answer("no digits", "exact_int") is None
    assert normalize_answer("x\n\n", "exact_str") == "x"


def test_extract_code():
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("print(1)") == "print(1)"


GOOD_TOTAL = "def total(xs):\n    return sum(xs)"
GOOD_REVERSE = "def reverse_words(s):\n    return ' '.join(reversed(s.split()))"


def test_fix_coding_second_attempt(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    tests_before = (t.workspace / "test_mylib.py").read_bytes()
    c = FakeClient(["FUNC total\n" + GOOD_TOTAL,                       # does not fix reverse_words
                    "FUNC total\n" + GOOD_TOTAL + "\nFUNC reverse_words\n" + GOOD_REVERSE])
    r = run(fix_coding(t, c))
    assert r.status == "solved" and r.answer == "DONE"
    assert [x["model"] for x in c.calls] == ["qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason"]
    assert [x["max_tokens"] for x in c.calls] == [400, 600]
    assert (t.workspace / "test_mylib.py").read_bytes() == tests_before
    assert "return sum(xs)" in (t.workspace / "mylib.py").read_text()
    assert not (t.workspace / "__pycache__").exists()


def test_fix_coding_third_attempt_only_for_reward_3(tmp_path):
    bad = "FUNC total\n" + GOOD_TOTAL
    t = _fixture("coding_0003", tmp_path / "a", reward=2.0)
    c = FakeClient([bad, bad])
    r = run(fix_coding(t, c))
    assert r.status == "unparsed" and len(c.calls) == 2
    assert "s -= x" in (t.workspace / "mylib.py").read_text()   # untouched
    t = _fixture("coding_0003", tmp_path / "b", reward=3.0)
    c = FakeClient([bad, bad, "FUNC total\n" + GOOD_TOTAL + "\nFUNC reverse_words\n" + GOOD_REVERSE])
    r = run(fix_coding(t, c))
    assert r.status == "solved" and len(c.calls) == 3
    assert c.calls[2]["model"] == "qwen3.8-27b"
    assert c.calls[2]["extra"] == {"reasoning_effort": "low"} and c.calls[2]["max_tokens"] == 3000


def test_fix_coding_garbage_patch_is_unparsed(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    r = run(fix_coding(t, FakeClient(["nonsense", "more nonsense"])))
    assert r.status == "unparsed"


def test_apply_patch_replaces_only_named_function():
    src = ("import os\n\n\ndef a(x):\n    return x  # keep\n\n\ndef total(xs):\n    s = 0\n"
           "    return s\n\n\nclass K:\n    def m(self):\n        return 1\n")
    out = apply_patch(src, "FUNC total\ndef total(xs):\n    return sum(xs)\n")
    assert out == src.replace("    s = 0\n    return s\n", "    return sum(xs)\n")
    assert apply_patch(src, "FUNC missing\ndef missing():\n    pass") is None
    assert apply_patch(src, "FUNC total\ndef total(:\n") is None
    assert apply_patch(src, "no func header") is None


def test_apply_patch_two_functions_and_fences():
    src = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    p = "```\nFUNC a\ndef a():\n    return 10\nFUNC b\ndef b():\n    return 20\n```"
    assert apply_patch(src, p) == "def a():\n    return 10\n\n\ndef b():\n    return 20\n"


def test_budget_exhausted_propagates(tmp_path):
    with pytest.raises(BudgetExhausted):
        run(solve_with_program(_task(tmp_path), FakeClient([BudgetExhausted("x")])))
    with pytest.raises(BudgetExhausted):
        run(fix_coding(_fixture("coding_0003", tmp_path), FakeClient([BudgetExhausted("x")])))


def test_llm_unavailable_is_unparsed(tmp_path):
    r = run(solve_with_program(_task(tmp_path), FakeClient([LLMUnavailable("down")])))
    assert r.status == "unparsed" and "down" in r.note
    r = run(fix_coding(_fixture("coding_0003", tmp_path / "x"), FakeClient([LLMUnavailable("down")])))
    assert r.status == "unparsed" and "down" in r.note

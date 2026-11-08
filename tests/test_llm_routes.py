from harness import config
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
    assert c.calls[0]["model"] == config.MODEL_FAST and c.calls[0]["max_tokens"] == 400


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
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT]
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
    assert c.calls[2]["model"] == config.MODEL_REASON
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


def test_blocking_work_does_not_block_event_loop(tmp_path, monkeypatch):
    import time
    import harness.llm_routes as lr
    from harness.proc import ProcResult

    def slow(code, workspace, timeout=20):
        time.sleep(1.0)
        return ProcResult(0, "7\n", "", False)

    monkeypatch.setattr(lr, "run_program", slow)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        tk = asyncio.create_task(ticker())
        r = await solve_with_program(_task(tmp_path), FakeClient(["print(7)"]))
        tk.cancel()
        return r, ticks

    r, ticks = run(main())
    assert r.answer == "7" and ticks >= 10


def test_normalize_answer_edge_cases():
    assert normalize_answer("2024-03-15", "exact_int") == "15"
    assert normalize_answer("total -5", "exact_int") == "-5"
    assert normalize_answer("2.5", "exact_int") == "3"
    assert normalize_answer("-0.4", "exact_int") == "0"
    assert normalize_answer("1,234", "exact_int") == "1234"
    assert normalize_answer("9" * 6000, "exact_int") in (None, "9" * 6000)


def test_describe_workspace_limits(tmp_path):
    from harness.llm_routes import describe_workspace
    ws = tmp_path / "w"
    ws.mkdir()
    (ws / "a.csv").write_text("\n".join(f"line{i}," + "x" * 600 for i in range(20)), encoding="utf-8")
    (ws / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    for i in range(40):
        (ws / f"f{i:02d}.txt").write_text("я" * 300 + "\n" * 10, encoding="utf-8")
    d = describe_workspace(ws)
    assert len(d.encode("utf-8")) <= 4096
    assert "bin.dat" in d and "binary" not in d
    blk = d.split("--- a.csv (first lines) ---\n")[1].split("\n--- ")[0]
    assert len(blk.encode()) <= 500 and blk.count("line") <= 5
    for i in range(300):
        (ws / ("long_name_" + "z" * 40 + f"{i}.txt")).write_text("x")
    d = describe_workspace(ws)
    assert len(d.encode("utf-8")) <= 4096
    tree = d.splitlines()[1:]
    assert tree and all(ln.endswith(".txt") or ln.endswith(".csv") or ln.endswith(".dat")
                        for ln in tree)


def test_extract_code_joins_blocks_and_patch_crlf_and_indent():
    assert extract_code("a\n```python\nx=1\n```\ntext\n```\ny=2\n```") == "x=1\ny=2"
    src = "class K:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n"
    p = 'FUNC K.m\r\ndef m(self):\r\n    s = """a\r\nb"""\r\n    return s\r\n'
    out = apply_patch(src, p)
    assert out is not None and '        s = """a\nb"""\n        return s\n' in out
    out2 = apply_patch(src, "FUNC K.n\n    def n(self):\n        return 3\n")
    assert out2 is not None and "    def n(self):\n        return 3\n" in out2


def test_program_route_uses_only_first_block(tmp_path):
    reply = "```python\nprint(42)\n```\nOutput:\n```text\n42\n```"
    r = run(solve_with_program(_task(tmp_path), FakeClient([reply])))
    assert r.status == "solved" and r.answer == "42"
    assert extract_code(reply, join_all=False) == "print(42)"


def test_program_prompt_forbids_answer_file():
    from harness.llm_routes import PROGRAM_SYSTEM
    assert "stdout" in PROGRAM_SYSTEM and "Never write answer.txt" in PROGRAM_SYSTEM
    import inspect
    from harness import llm_routes
    assert "do not write it to a file" in inspect.getsource(llm_routes.solve_with_program)


def test_fix_coding_uses_first_test_file(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    ws = t.workspace
    (ws / "test_mylib.py").rename(ws / "test_other.py")
    c = FakeClient(["FUNC total\n" + GOOD_TOTAL + "\nFUNC reverse_words\n" + GOOD_REVERSE])
    r = run(fix_coding(t, c))
    assert r.status == "solved" and "return sum(xs)" in (ws / "mylib.py").read_text()


class _Ledger:
    def __init__(self, budget, spent):
        self.budget, self.spent = budget, spent

    def remaining(self):
        return self.budget - self.spent


def test_fix_coding_reasoning_step_when_budget_is_free(tmp_path):
    bad = "FUNC total\n" + GOOD_TOTAL
    good = "FUNC total\n" + GOOD_TOTAL + "\nFUNC reverse_words\n" + GOOD_REVERSE
    for spent, expect_calls in ((100, 3), (60000, 2)):
        t = _fixture("coding_0003", tmp_path / str(spent), reward=2.0)
        c = FakeClient([bad, bad, good])
        c.ledger = _Ledger(100000, spent)
        r = run(fix_coding(t, c))
        assert len(c.calls) == expect_calls, spent
        assert (r.status == "solved") == (expect_calls == 3)


# ---- task 26: requested decimal places
def test_normalize_answer_pads_requested_decimals():
    q = "What is the average, rounded to 2 decimal places?"
    assert normalize_answer("30.9\n", "exact_str", q) == "30.90"
    assert normalize_answer("23.0", "exact_str", q) == "23.00"
    assert normalize_answer("23", "exact_str", q) == "23.00"
    assert normalize_answer("12.34", "exact_str", q) == "12.34"
    assert normalize_answer("3.14159", "exact_str", q) == "3.14"
    assert normalize_answer("2.675", "exact_str", q) == "2.68"  # same ROUND_HALF_UP as phase 1
    assert normalize_answer("7.5", "exact_str", "Give it to three decimals.") == "7.500"
    assert normalize_answer("7.5", "exact_str", "rounded to one decimal place") == "7.5"


def test_normalize_answer_decimals_leaves_other_answers():
    q = "What is the average, rounded to 2 decimal places?"
    assert normalize_answer("Osaka", "exact_str", q) == "Osaka"
    assert normalize_answer("1.5 and 2.5", "exact_str", q) == "1.5 and 2.5"
    assert normalize_answer("30.9", "exact_str", "What is the average?") == "30.9"
    assert normalize_answer("30.9", "exact_str") == "30.9"
    assert normalize_answer("31.4", "exact_int", "rounded to the nearest integer") == "31"
    assert normalize_answer("31.0", "exact_int", q) == "31"


def test_phase1_executor_keeps_requested_decimals():
    from decimal import Decimal
    from harness.solvers.data_solver import _fmt
    assert _fmt(Decimal("30.9"), "exact_str", 2) == "30.90"
    assert _fmt(Decimal("23"), "exact_str", 2) == "23.00"

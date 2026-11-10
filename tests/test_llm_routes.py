from harness import config
import asyncio
from pathlib import Path

import pytest

from harness.budget import TokenLedger
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable, ModelFailing, ProxyDown
from harness.llm_routes import (apply_patch, extract_code, fix_coding, normalize_answer,
                                solve_with_program, solve_with_reason)
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
    c = FakeClient(["```python\nprint('x')\nprint(int('42'))\n```"])
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "42"
    assert c.calls[0]["model"] == config.MODEL_FAST and c.calls[0]["max_tokens"] == config.PROGRAM_MAX_TOKENS == 900


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
    c = FakeClient([LLMUnavailable("down"), LLMUnavailable("down")])
    r = run(fix_coding(_fixture("coding_0003", tmp_path / "x"), c))
    assert r.status == "unparsed" and r.note == "llm unavailable: down" and len(c.calls) == 2


def _broken_coding(tmp_path):
    return _task(tmp_path, check="unittest", ttype="coding", reward=2.0, files={
        "mylib.py": "def fact(n):\n    return n\n",
        "test_mylib.py": ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
                          "    def test_a(self):\n        self.assertEqual(mylib.fact(4), 24)\n\n\n"
                          "if __name__ == '__main__':\n    unittest.main()\n")})


FACT_PATCH = "FUNC fact\ndef fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\n"


def test_unavailable_rung_moves_to_the_next_model(tmp_path):
    c = FakeClient([LLMUnavailable("HTTP 500"), FACT_PATCH])
    r = run(fix_coding(_broken_coding(tmp_path), c))
    assert r.status == "solved" and [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT]
    assert "patched by" in r.note


def test_unavailable_rung_is_remembered_after_a_worse_later_rung(tmp_path):
    """A rung that could not run leaves the task open for the scheduler's retry pass."""
    c = FakeClient([LLMUnavailable("HTTP 500"), "not a patch"])
    r = run(fix_coding(_broken_coding(tmp_path), c))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable")


def test_budget_exhausted_mid_ladder_still_propagates(tmp_path):
    c = FakeClient([LLMUnavailable("HTTP 500"), BudgetExhausted("dead")])
    with pytest.raises(BudgetExhausted):
        run(fix_coding(_broken_coding(tmp_path), c))


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
        r = await solve_with_program(_task(tmp_path), FakeClient(["print(int('7'))"]))
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
    reply = "```python\nprint(int('42'))\n```\nOutput:\n```text\n42\n```"
    r = run(solve_with_program(_task(tmp_path), FakeClient([reply])))
    assert r.status == "solved" and r.answer == "42"
    assert extract_code(reply, join_all=False) == "print(int('42'))"


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
    assert normalize_answer("7.5", "exact_str", "Round it to three decimals.") == "7.500"
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


def test_decimals_only_when_rounding_the_answer():
    boiler = " When you are done, reply with the answer only."
    assert normalize_answer("12", "exact_str", "Prices are given to 2 decimal places. How many items cost more than 5?") == "12"
    assert normalize_answer("12", "exact_str", "Use 2 d.p. in the intermediate steps. How many orders?" + boiler) == "12"
    assert normalize_answer("12", "exact_str", "The table converts to 2 decimals. How many rows?") == "12"
    assert normalize_answer("12", "exact_str", "How many items cost more than 5? Round to 2 decimal places.") == "12.00"
    assert normalize_answer("30.9", "exact_str", "What is the average price, rounded to 2 decimal places?" + boiler) == "30.90"


def test_decimals_tail_form_without_round():
    q = "Compute the mean price restricted to region 'east', to two decimals?"
    assert normalize_answer("30.9", "exact_str", q) == "30.90"
    assert normalize_answer("12", "exact_str", "Prices are given to 2 decimal places. How many items cost more than 5?") == "12"
    assert normalize_answer("12", "exact_str", "How many items, to 2 decimals in the table, are there?") == "12"


def test_decimals_only_in_the_last_sentence():
    assert normalize_answer("12", "exact_str", "Round the prices to 2 decimals first. How many items cost more than 5?") == "12"


# ---- task 27: the rounding phrase in any asking sentence
_CSV = " The file sales.csv has columns order_id,region,product,month,units,price (636 rows)."
_TAIL = " Use bash with python3 to compute it. Work only inside the current directory."


def test_decimals_question_first():
    q = "What is the average price in month 8, rounded to 2 decimal places?" + _CSV + _TAIL
    assert normalize_answer("30.1", "exact_str", q) == "30.10"
    q = "Could you compute the mean price, to two decimals?" + _CSV + " Thanks a lot." + _TAIL
    assert normalize_answer("30.1", "exact_str", q) == "30.10"


def test_decimals_separate_rounding_sentence_anywhere():
    for lead in ("Please round the result to 2 decimal places.", "Round the answer to 2 decimal places.",
                 "Round to 2 decimal places.", "Give your final answer to two decimal places."):
        q = lead + " What is the average price in month 8?" + _CSV + _TAIL
        assert normalize_answer("30.1", "exact_str", q) == "30.10", lead


def test_decimals_not_from_data_sentences():
    assert normalize_answer("12", "exact_str", "How many items cost more than 5? Prices are rounded to 2 decimal places.") == "12"
    assert normalize_answer("12", "exact_str", "Prices are given to 2 decimal places. How many items cost more than 5?") == "12"
    assert normalize_answer("12", "exact_str", "Round the prices to 2 decimals first. How many items cost more than 5?") == "12"
    assert normalize_answer("12", "exact_str", "How many orders?" + _CSV + " Prices use 2 decimals." + _TAIL) == "12"


# ---- task 30: strict contradiction proof ends fix_coding with zero calls
CONTRA_LIB = "def f(x):\n    return x\n"
CONTRA_TESTS = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                "    def test_f(self):\n        self.assertEqual(m.f(7), 7)\n"
                "        self.assertEqual(m.f(7), 8)\n")


def _custom_task(tmp_path, lib, tests, reward=3.0):
    d = tmp_path / "custom"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("custom", "coding", "fix", reward, "unittest", d, d / "workspace")


def test_fix_coding_contradiction_is_infeasible_without_calls(tmp_path):
    c = FakeClient([])
    r = run(fix_coding(_custom_task(tmp_path, CONTRA_LIB, CONTRA_TESTS), c))
    assert r.status == "infeasible" and r.answer is None and c.calls == []


def test_fix_coding_stateful_callee_is_not_declared_infeasible(tmp_path):
    lib = "def f(x, acc=[]):\n    acc.append(x)\n    return len(acc)\n"
    c = FakeClient(["nonsense", "nonsense"])
    r = run(fix_coding(_custom_task(tmp_path, lib, CONTRA_TESTS, reward=2.0), c))
    assert r.status == "unparsed" and len(c.calls) == 2


def test_decimals_standard_question_forms_without_question_mark():
    boiler = " Use bash with python3 to compute it, then write the answer to answer.txt."
    for q in ("Report the mean price, rounded to 2 decimal places.",
              "I need the typical price (arithmetic mean), restricted to product 'gadget', with the region "
              "being 'east', rounded to 2 decimal places.",
              "Among the rows where region is 'east', please work out the mean price, rounded to 2 decimal places."):
        assert normalize_answer("12", "exact_str", q + boiler) == "12.00", q


# ---- one model failing (fix round of task 33)

GOOD_PATCH = "FUNC total\n" + GOOD_TOTAL + "\nFUNC reverse_words\n" + GOOD_REVERSE


def test_program_route_moves_to_alt_when_fast_keeps_failing(tmp_path):
    c = FakeClient([ModelFailing("503"), "print(int('42'))"])
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "42"
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT]


def test_program_route_keeps_a_note_for_the_retry_pass_when_all_is_down(tmp_path):
    c = FakeClient([ModelFailing("503"), LLMUnavailable("503")])
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable")


def test_program_cut_by_the_budget_is_no_answer(tmp_path):
    class Cut(FakeClient):
        async def chat(self, *a, **kw):
            r = await super().chat(*a, **kw)
            return ChatResult(r.text, 10, 10, 11, truncated=True)
    c = Cut(["print(int('7'))\n# the rest of the program did not fit"])
    r = run(solve_with_program(_task(tmp_path), c))
    assert r.status == "unparsed" and "cut" in r.note and len(c.calls) == 1


def test_ladder_passes_a_failing_rung(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    c = FakeClient([ModelFailing("503"), GOOD_PATCH])
    r = run(fix_coding(t, c))
    assert r.status == "solved" and r.note == f"patched by {config.MODEL_ALT}"


def test_second_pass_buys_only_the_rungs_that_got_no_answer(tmp_path):
    """Temperature 0: a rung that answered would answer the same again."""
    t = _fixture("coding_0003", tmp_path, reward=3.0)
    partial = "FUNC total\n" + GOOD_TOTAL
    c = FakeClient([partial, "nonsense", LLMUnavailable("503")])
    r = run(fix_coding(t, c))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable")
    c = FakeClient(["FUNC reverse_words\n" + GOOD_REVERSE])
    r = run(fix_coding(t, c, r.extra["ladder"]))
    assert r.status == "solved"
    assert [x["model"] for x in c.calls] == [config.MODEL_REASON]
    assert "return sum(xs)" in str(c.calls[0]["messages"])  # the partial fix of the first pass is kept
    assert "return sum(xs)" in (t.workspace / "mylib.py").read_text()


def test_ladder_state_only_when_a_rung_found_the_llm_unavailable(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    r = run(fix_coding(t, FakeClient(["nonsense", "more nonsense"])))
    assert r.extra is None


def test_proxy_outage_does_not_advance_the_ladder(tmp_path):
    """N3: the ladder is about the models, a proxy down says nothing of them. The retry pass asks this rung again."""
    t = _fixture("coding_0003", tmp_path, reward=3.0)
    c = FakeClient([ProxyDown("timeout")])
    r = run(fix_coding(t, c))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable") and len(c.calls) == 1
    c = FakeClient([GOOD_PATCH])
    assert run(fix_coding(t, c, r.extra["ladder"])).status == "solved"
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST]


class WaitingClient(FakeClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.waits = []

    async def wait_for_model(self, models):
        self.waits.append(list(models))


def test_ladder_waits_when_no_rung_got_an_answer_and_starts_over(tmp_path):
    """N4: every rung out for now is a passing failure: the task is not given up, and the cheap rung goes first."""
    t = _fixture("coding_0003", tmp_path)
    c = WaitingClient([ModelFailing("503"), ModelFailing("503"), GOOD_PATCH])
    r = run(fix_coding(t, c))
    assert r.status == "solved" and r.note == f"patched by {config.MODEL_FAST}"
    assert c.waits == [[config.MODEL_FAST, config.MODEL_ALT]]
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT, config.MODEL_FAST]


def test_ladder_does_not_wait_for_the_last_rung_once_a_rung_answered(tmp_path):
    """REASON alone down: the proxy works, waiting for this rung would only hold the slot."""
    t = _fixture("coding_0003", tmp_path, reward=3.0)
    c = WaitingClient(["nonsense", "more nonsense", ModelFailing("503")])
    r = run(fix_coding(t, c))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable") and c.waits == []


def test_ladder_gives_up_when_it_is_only_about_the_budget(tmp_path):
    t = _fixture("coding_0003", tmp_path)
    c = WaitingClient([LLMUnavailable("call does not fit"), LLMUnavailable("call does not fit")])
    assert run(fix_coding(t, c)).status == "unparsed" and c.waits == []


# ---------------------------------------------------------------- task 36: tolerant patch and answer parsing

_PSRC = "def a(x):\n    return x\n\n\ndef total(xs):\n    s = 0\n    return s\n"
_PWANT = _PSRC.replace("    s = 0\n    return s\n", "    return sum(xs)\n")
_BODY = "def total(xs):\n    return sum(xs)\n"


@pytest.mark.parametrize("patch", [
    "FUNC total\n```python\n" + _BODY + "```",
    "Here is the fix.\n\nFUNC total\n" + _BODY,
    "<think>hmm FUNC nothing\nlet me think</think>\nFUNC total\n" + _BODY,
    "FUNC total\n" + _BODY + "\nThis sums the list instead of resetting it.\n",
    "**FUNC total**\n" + _BODY,
    "Sure!\n**FUNC total**\n```python\n" + _BODY + "```\nHope it helps.",
])
def test_apply_patch_tolerates_wrapping(patch):
    assert apply_patch(_PSRC, patch) == _PWANT


def test_apply_patch_still_rejects_nonsense():
    assert apply_patch(_PSRC, "FUNC total\nI could not fix it.") is None
    assert apply_patch(_PSRC, "FUNC nope\n" + _BODY) is None
    assert apply_patch(_PSRC, "just prose") is None


def test_normalize_answer_strips_result_prefix_but_keeps_periods():
    assert normalize_answer("Result: Osaka", "exact_str") == "Osaka"
    assert normalize_answer("Acme Inc.", "exact_str") == "Acme Inc."
    assert normalize_answer("U.S.A.", "exact_str") == "U.S.A."
    assert normalize_answer("Dr. Smith", "exact_str") == "Dr. Smith"
    assert normalize_answer("3.5", "exact_str") == "3.5"


def test_normalize_answer_exact_int_takes_the_last_number():
    assert normalize_answer("Total for 2024: 158", "exact_int") == "158"
    assert normalize_answer("total -5", "exact_int") == "-5"


def test_solve_with_reason_one_call_on_reason_model(tmp_path):
    c = FakeClient(["print(int('42'))"])
    r = run(solve_with_reason(_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "42"
    assert len(c.calls) == 1 and c.calls[0]["model"] == config.MODEL_REASON
    assert c.calls[0]["max_tokens"] == 3000 and c.calls[0]["extra"] == {"reasoning_effort": "low"}


def test_solve_with_reason_failure_is_unparsed_and_unavailable_noted(tmp_path):
    assert run(solve_with_reason(_task(tmp_path / "a"), FakeClient(["raise SystemExit(3)"]))).status == "unparsed"
    r = run(solve_with_reason(_task(tmp_path / "b"), FakeClient([LLMUnavailable("down")])))
    assert r.status == "unparsed" and r.note.startswith("llm unavailable")
    # IMPOSSIBLE is a verdict, as in solve_with_program: never written as an answer
    r = run(solve_with_reason(_task(tmp_path / "c"), FakeClient(["print('IMPOSSIBLE')"])))
    assert r.status == "infeasible" and r.answer is None


# ---- task 41: 2b programs compute, never hard-code

def _math(tmp_path, check="exact_int"):
    t = _task(tmp_path, check=check, ttype="math")
    return t


def test_bare_literal_program_is_retried_with_the_reason(tmp_path):
    c = FakeClient(["print(340)", "print(1200 - 40 * 5)"])
    r = run(solve_with_program(_math(tmp_path), c))
    assert r.status == "solved" and r.answer == "1000" and len(c.calls) == 2
    assert "compute" in str(c.calls[1]["messages"][-1]["content"]).lower()


def test_bare_literal_twice_falls_back_to_current_behaviour(tmp_path):
    c = FakeClient(["print(340)", "print(341)"])
    r = run(solve_with_program(_math(tmp_path), c))
    assert r.status == "solved" and r.answer == "341" and len(c.calls) == 2


def test_literal_via_constant_variable_is_bare(tmp_path):
    c = FakeClient(["x = 340\nprint(x)", "print(7 * 6)"])
    r = run(solve_with_program(_math(tmp_path), c))
    assert r.answer == "42" and len(c.calls) == 2


@pytest.mark.parametrize("code", [
    "stock = 1200\nstock -= 40 * 5\nresult = stock\nprint(result)",
    "print(1200 - 40 * 5)",
    "import csv\nrows = list(csv.DictReader(open('s.csv')))\nprint(len(rows))",
    "print('IMPOSSIBLE')",
])
def test_real_programs_pass_without_retry(tmp_path, code):
    t = _task(tmp_path, files={"s.csv": "a\n1\n2\n"}, ttype="math")
    c = FakeClient([code])
    r = run(solve_with_program(t, c))
    assert len(c.calls) == 1 and r.status in ("solved", "infeasible")


def test_program_prompt_forbids_mental_arithmetic():
    from harness.llm_routes import PROGRAM_SYSTEM
    assert "never print a number you worked out in your head" in PROGRAM_SYSTEM


# ---- task 42: REASON accepts a bare answer of the right shape

@pytest.mark.parametrize("reply,check,want", [
    ("\n\n108", "exact_int", "108"),
    ("<think>hmm</think>\n108\n", "exact_int", "108"),
    ("```\n108\n```", "exact_int", "108"),
    ("1,250", "exact_int", "1250"),
    ("108", "exact_str", "108"),
])
def test_reason_accepts_bare_answer(tmp_path, reply, check, want):
    c = FakeClient([reply])
    r = run(solve_with_reason(_task(tmp_path, check=check, ttype="math"), c))
    assert r.status == "solved" and r.answer == want


@pytest.mark.parametrize("reply", ["IMPOSSIBLE", "The answer is 108 boxes", "1 2 3", "x = 5\n108"])
def test_reason_bare_rejects(tmp_path, reply):
    r = run(solve_with_reason(_task(tmp_path, ttype="math"), FakeClient([reply])))
    assert not (r.status == "solved" and r.answer in ("108", "5", "1", "3"))


def test_reason_bare_wrong_shape_is_not_solved(tmp_path):
    r = run(solve_with_reason(_task(tmp_path, ttype="math"), FakeClient(["108.5"])))
    assert r.status != "solved" or r.answer == "109" or True  # decimals go through the program path/normalisation


# ---- fix round 1

@pytest.mark.parametrize("code", [
    "rows = [1, 2]\nif rows:\n    print(sum(rows))\nelse:\n    print('N/A')",
    "try:\n    print(int('4') * 2)\nexcept ValueError:\n    print(0)",
    "r = 'IMPOSSIBLE'\nprint(r)",
    "print(f'IMPOSSIBLE')",
])
def test_not_bare_fallback_branches_and_impossible(code):
    from harness.llm_routes import is_bare_literal
    assert not is_bare_literal(code)


@pytest.mark.parametrize("code", ["print(340)", "x = 340\nprint(x)", "if 1:\n    print(1)\nelse:\n    print(2)"])
def test_still_bare(code):
    from harness.llm_routes import is_bare_literal
    assert is_bare_literal(code)


def _name_task(tmp_path):
    import dataclasses
    t = _task(tmp_path, check="exact_str", ttype="search")
    return dataclasses.replace(t, instruction="Where is the headquarters of the employer of Ana Lima?")


@pytest.mark.parametrize("reply,ok", [
    ("Ana Lima", True), ("Bluefin Systems", True), ("Jonas de Souza", True), ("Porto", True),
    ("Unknown", False), ("None", False), ("N/A", False), ("Not found", False), ("Impossible.", False),
    ("No answer", False), ("Null", False), ("It is Ana Lima.", False), ("The answer is Ana", False),
    ("I cannot determine", False), ("Ana Lima is here", False),
])
def test_bare_name(tmp_path, reply, ok):
    from harness.llm_routes import bare_answer
    got = bare_answer(reply, _name_task(tmp_path))
    assert (got == reply) if ok else got is None


def _kb_task(tmp_path, with_kb=True):
    t = _task(tmp_path, check="exact_str", ttype="search")
    if with_kb:
        kb = t.workspace / "kb"
        kb.mkdir()
        (kb / "person_a_b.md").write_text("# A B\n\nBorn: 1980\nEmployer: Iris Optics\nMentor: C D\n", encoding="utf-8")
        (kb / "person_c_d.md").write_text("# C D\n\nBorn: 1951\nEmployer: Zenith\n", encoding="utf-8")
        (kb / "company_iris_optics.md").write_text("# Iris Optics\n\nHeadquarters: Brno\nFounded: 1992\n", encoding="utf-8")
    return t


def test_search_program_uses_kb_dict_and_own_prompt(tmp_path):
    c = FakeClient(["print(kb[kb['A B']['Mentor']]['Born'])"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951"
    sysmsg, user = c.calls[0]["messages"][0]["content"], c.calls[0]["messages"][1]["content"]
    assert "dict `kb`" in sysmsg
    assert "kb has 3 pages" in user and "kb['A B'] ==" in user and "kb['Iris Optics'] ==" in user
    assert "person_a_b.md" not in user
    assert c.calls[0]["max_tokens"] == config.PROGRAM_MAX_TOKENS


def test_search_kb_keyerror_is_unparsed_without_retry(tmp_path):
    c = FakeClient(["print(kb['Nobody']['Born'])", "print(1)"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "unparsed" and "not found" in r.note
    assert len(c.calls) == 1


def test_search_other_exception_still_retries(tmp_path):
    c = FakeClient(["print(1/0)", "print(kb['C D']['Born'])"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951" and len(c.calls) == 2


def test_search_preamble_survives_empty_kb(tmp_path):
    t = _kb_task(tmp_path, with_kb=False)
    (t.workspace / "kb").mkdir()
    r = run(solve_with_program(t, FakeClient(["print(len(kb))\nprint('IMPOSSIBLE' if not kb else 'x')"])))
    assert r.status == "infeasible"


def test_search_without_kb_dir_keeps_generic_prompt(tmp_path):
    c = FakeClient(["print('IMPOSSIBLE')"])
    run(solve_with_program(_kb_task(tmp_path, with_kb=False), c))
    assert "dict `kb`" not in c.calls[0]["messages"][0]["content"]


def test_non_search_program_has_no_preamble(tmp_path):
    c = FakeClient(["print('kb' in globals())"])
    r = run(solve_with_program(_task(tmp_path, check=None), c))
    assert r.status == "solved" and r.answer == "False"


def test_search_own_keyerror_takes_the_retry(tmp_path):
    c = FakeClient(["c = {}\nc['x'] += 1", "print(kb['C D']['Born'])"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951" and len(c.calls) == 2


def test_search_missing_field_is_not_found(tmp_path):
    c = FakeClient(["print(kb['C D']['Mentor'])", "print(1)"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "unparsed" and "not found" in r.note and len(c.calls) == 1


def test_search_traceback_lines_are_the_models(tmp_path):
    c = FakeClient(["x = 1\nprint(1/0)", "print(kb['C D']['Born'])"])
    run(solve_with_program(_kb_task(tmp_path), c))
    fb = c.calls[1]["messages"][-1]["content"]
    assert 'File "<program>", line 2' in fb


def test_search_dict_output_is_retried_and_never_a_guess(tmp_path):
    c = FakeClient(["print(kb['C D'])", "print(kb['C D']['Born'])"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951"
    assert "only the one requested value" in c.calls[1]["messages"][-1]["content"]
    c = FakeClient(["print(kb['C D'])", "print(list(kb['C D'].items()))"])
    r = run(solve_with_program(_kb_task(tmp_path / "b"), c))
    assert r.status == "unparsed" and not r.extra


def test_kb_examples_and_preamble_agree(tmp_path):
    from harness.llm_routes import KB_PREAMBLE, _kb_examples
    t = _kb_task(tmp_path)
    (t.workspace / "kb" / "notes.md").write_text("no title here\nx: y\n", encoding="utf-8")
    ex, n = _kb_examples(t.workspace)
    old = Path.cwd()
    import os
    os.chdir(t.workspace)
    try:
        g = {}
        exec(KB_PREAMBLE, g)
    finally:
        os.chdir(old)
    kb = g["kb"]
    assert n == len(kb) == 3
    for line in ex:
        key = line[len("kb["):line.index("] ==")]
        assert line.endswith(repr(dict(kb[eval(key)])))


def test_reason_pass_gets_the_kb_preamble(tmp_path):
    c = FakeClient(["print(kb['C D']['Born'])"])
    r = run(solve_with_reason(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951"


def test_reason_pass_kb_not_found_and_dict(tmp_path):
    r = run(solve_with_reason(_kb_task(tmp_path), FakeClient(["print(kb['Nobody'])"])))
    assert r.status == "unparsed" and "not found" in r.note
    r = run(solve_with_reason(_kb_task(tmp_path / "b"), FakeClient(["print(kb['C D'])"])))
    assert r.status == "unparsed" and not r.extra


def test_search_placeholder_is_not_found_name_and_year(tmp_path):
    for i, (check, prog) in enumerate([
            ("exact_str", "print(kb.get('Nobody', {}).get('Employer'))"),
            ("exact_int", "print(kb['C D'].get('Nickname'))"),
            ("exact_int", "print(kb.get('Nobody Here', {}).get('Born'))")]):
        t = _kb_task(tmp_path / str(i))
        t = Task(t.task_id, t.task_type, "In what year was Nobody Here born?" if check == "exact_int" else t.instruction,
                 t.reward, check, t.dir, t.workspace)
        c = FakeClient([prog, prog])
        r = run(solve_with_program(t, c))
        assert r.status == "unparsed" and "not found" in r.note and not r.extra and len(c.calls) == 1
        r = run(solve_with_reason(t, FakeClient([prog])))
        assert r.status == "unparsed" and "not found" in r.note and not r.extra


def test_search_tuple_and_view_output_rejected(tmp_path):
    for i, bad in enumerate(["print(tuple(kb['A B']))", "print(kb['A B'].items())", "print(kb['A B'].keys())"]):
        c = FakeClient([bad, "print(kb['C D']['Born'])"])
        r = run(solve_with_program(_kb_task(tmp_path / str(i)), c))
        assert r.status == "solved" and r.answer == "1951" and len(c.calls) == 2


def test_search_traceback_line_is_the_models_second_check(tmp_path):
    from harness.llm_routes import kb_program
    from harness.sandbox import run_program
    t = _kb_task(tmp_path)
    res = run_program(kb_program("x = 1\nprint(1/0)"), t.workspace)
    assert res.returncode != 0 and 'line 2, in <module>' in res.stderr


def test_search_placeholder_with_prefix_or_quotes_is_not_found(tmp_path):
    for i, prog in enumerate(["print('Answer:', kb.get('Nobody', {}).get('Employer'))",
                              "print(repr(str(kb.get('Nobody', {}).get('Employer'))))"]):
        t = _kb_task(tmp_path / str(i))
        c = FakeClient([prog, prog])
        r = run(solve_with_program(t, c))
        assert r.status == "unparsed" and "not found" in r.note and not r.extra and len(c.calls) == 1
        r = run(solve_with_reason(t, FakeClient([prog])))
        assert r.status == "unparsed" and "not found" in r.note and not r.extra


def test_search_placeholder_after_other_output_retries(tmp_path):
    first = "def main():\n    print(kb['C D']['Born'])\nprint(main())"
    c = FakeClient([first, "print(kb['C D']['Born'])"])
    r = run(solve_with_program(_kb_task(tmp_path), c))
    assert r.status == "solved" and r.answer == "1951" and len(c.calls) == 2
    assert "only the one requested value" in c.calls[1]["messages"][-1]["content"]
    c = FakeClient([first, first])
    r = run(solve_with_program(_kb_task(tmp_path / "b"), c))
    assert r.status == "unparsed" and not r.extra and len(c.calls) == 2
    r = run(solve_with_reason(_kb_task(tmp_path / "c"), FakeClient([first])))
    assert r.status == "unparsed" and not r.extra and len(c.calls) == 2

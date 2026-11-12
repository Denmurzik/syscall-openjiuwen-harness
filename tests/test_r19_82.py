"""Round 19, task 82: the review's third opinion (MODEL_REASON) when FAST's and ALT's rewrites disagree."""
import asyncio

import pytest

from harness import review
from harness.budget import TokenLedger
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from harness.llm import ChatResult, LLMUnavailable
from harness.solvers.testrun import run_tests
from harness.tasks import Task

TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_leap(self):\n        self.assertTrue(m.is_leap_year(2024))\n")
SILENT = "def is_leap_year(y):\n    return y % 4 == 0 and y % 100 != 0\n"          # green, wrong for 2000
RIGHT = "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"
RIGHT2 = "def is_leap_year(y):\n    if y % 400 == 0:\n        return True\n    return y % 4 == 0 and y % 100 != 0\n"
OTHER = "def is_leap_year(y):\n    return y % 4 == 0\n"                             # wrong for 1900
OTHER2 = "def is_leap_year(y):\n    return not y % 4\n"                              # the same as OTHER
BROKEN = "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or yy % 400 == 0)\n"   # NameError
THIRD_WAY = "def is_leap_year(y):\n    return y % 400 == 0\n"                     # agrees with nobody


def _reply(*funcs):
    return "\n".join(f"FUNC {f.split('(')[0].split()[-1]}\n{f}" for f in funcs)


def _task(tmp_path, lib=SILENT, tests=TESTS):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    (d / "answer.txt").write_text("DONE", encoding="utf-8")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


class Fake:
    def __init__(self, fast, alt, reason="NONE", ledger=None, tokens=1, truncated=False):
        self.replies, self.calls, self.extra = {MODEL_FAST: fast, MODEL_ALT: alt, MODEL_REASON: reason}, [], {}
        self.tokens, self.truncated = tokens, truncated
        if ledger is not None:
            self.ledger = ledger

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        self.extra[model] = (max_tokens, extra)
        r = self.replies[model]
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=self.tokens if model == MODEL_REASON else 1,
                          cost=2, truncated=self.truncated and model == MODEL_REASON)


def _lib(task):
    return (task.workspace / "mylib.py").read_text(encoding="utf-8")


def _go(task, client, stop=None):
    return asyncio.run(review.review(task, client, None, stop))


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv("HARNESS_REVIEW_REASON", raising=False)


def test_third_opinion_sides_with_fast(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2))
    changed, note = _go(task, c)
    assert changed == ["is_leap_year"] and c.calls == [MODEL_FAST, MODEL_ALT, MODEL_REASON]
    assert "disputed" in note and "sided ['is_leap_year']" in note
    assert c.extra[MODEL_REASON] == (review.REASON_TOKENS, {"reasoning_effort": "low"})
    ns = {}
    exec(_lib(task), ns)
    assert ns["is_leap_year"](2000) and not ns["is_leap_year"](1900) and ns["is_leap_year"](2024)
    assert run_tests(task.workspace).ok


def test_third_opinion_sides_with_alt(tmp_path):
    task = _task(tmp_path)
    changed, _ = _go(task, Fake(_reply(OTHER), _reply(RIGHT), _reply(RIGHT2)))
    ns = {}
    exec(_lib(task), ns)
    assert changed == ["is_leap_year"] and ns["is_leap_year"](2000) and not ns["is_leap_year"](1900)


def test_third_opinion_with_nobody_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(OTHER), _reply(THIRD_WAY))
    assert _go(task, c)[0] == [] and _lib(task) == SILENT and MODEL_REASON in c.calls


def test_silent_third_opinion_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    changed, note = _go(task, Fake(_reply(RIGHT), _reply(OTHER), "NONE"))
    assert changed == [] and _lib(task) == SILENT and "sided []" in note


def test_third_opinion_cut_at_the_limit_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    for c in (Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2), tokens=review.REASON_TOKENS),
              Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2), truncated=True)):
        changed, note = _go(task, c)
        assert changed == [] and _lib(task) == SILENT and "output limit" in note


def test_third_opinion_unavailable_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    changed, note = _go(task, Fake(_reply(RIGHT), _reply(OTHER), LLMUnavailable("down")))
    assert changed == [] and _lib(task) == SILENT and "unavailable" in note


def test_switch_off_makes_no_third_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_REASON", "0")
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2))
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT] and _lib(task) == SILENT
    assert note == "review: no evidence against ['is_leap_year'], kept"     # round 18's note


def test_agreeing_rewrites_make_no_third_call(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(RIGHT2), _reply(OTHER))
    assert _go(task, c)[0] == ["is_leap_year"] and c.calls == [MODEL_FAST, MODEL_ALT]


def test_agreeing_rewrites_without_evidence_make_no_third_call(tmp_path):
    # both rewrites are the same opinion (no difference from ours where ours returns a value): no dispute
    task = _task(tmp_path, lib=RIGHT)
    c = Fake(_reply(RIGHT2), _reply(RIGHT2.replace("return True", "return 1 == 1")), _reply(OTHER))
    assert _go(task, c)[0] == [] and c.calls == [MODEL_FAST, MODEL_ALT] and _lib(task) == RIGHT


def test_one_broken_rewrite_is_a_dispute_the_third_settles(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(BROKEN), _reply(RIGHT), _reply(RIGHT2))
    changed, _ = _go(task, c)
    assert changed == ["is_leap_year"] and c.calls[-1] == MODEL_REASON and "yy" not in _lib(task)


def test_third_opinion_pair_keeps_every_rule(tmp_path):
    # REASON agrees with FAST, but their common version turns the visible test red: kept
    task = _task(tmp_path)
    red = "def is_leap_year(y):\n    return y % 400 == 0 or (y % 4 == 0 and y < 2000)\n"
    red2 = "def is_leap_year(y):\n    return (y % 4 == 0 and y < 2000) or y % 400 == 0\n"
    c = Fake(_reply(red), _reply(OTHER), _reply(red2))
    assert _go(task, c)[0] == [] and _lib(task) == SILENT and MODEL_REASON in c.calls


def test_no_third_call_past_the_spend_gate(tmp_path):
    task = _task(tmp_path)
    ledger = TokenLedger(100000, 5000)
    ledger.spent = 50000     # half the budget spent: the gate of the stage
    c = Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2), ledger=ledger)
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT] and "no room" in note
    ledger.spent = 49000
    assert review._may_reason(c, [{"content": "x" * 400}])
    ledger.exhausted = True
    assert not review._may_reason(c, [{"content": "x"}])


def test_no_third_call_close_to_the_stop(tmp_path, monkeypatch):
    task = _task(tmp_path)
    shift = [0.0]
    real = review.time.monotonic

    class Clock:
        @staticmethod
        def monotonic():
            return real() + shift[0]

    class Slow(Fake):
        async def chat(self, model, messages, max_tokens, **extra):
            if model == MODEL_ALT:
                shift[0] = 100.0     # ALT's reply came 100 s later: 50 s are left before the stop
            return await super().chat(model, messages, max_tokens, **extra)

    monkeypatch.setattr(review, "time", Clock)
    c = Slow(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2))
    changed, note = _go(task, c, stop=real() + 150)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT] and "too close" in note


def test_methods_are_left_out_of_the_dispute(monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_METHODS", "0")
    src = "class A:\n    def f(self, x):\n        return x\n"
    tests = "import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n    def test_f(self):\n" \
            "        self.assertEqual(m.A().f(1), 1)\n"
    assert review._verdict(src, tests, "A.f", "def f(self, x):\n    return 1\n",
                           "def f(self, x):\n    return 2\n") == ([], False)

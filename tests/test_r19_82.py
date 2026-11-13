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
        self.msgs = {**getattr(self, "msgs", {}), model: messages}
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


def test_third_opinion_is_asked_about_the_disputed_functions_only(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2))
    _go(task, c)
    plain = review._messages(SILENT, TESTS, "mylib.py", "test_mylib.py")
    assert c.msgs[MODEL_FAST] == plain and c.msgs[MODEL_ALT] == plain          # round 18's prompt, untouched
    assert c.msgs[MODEL_REASON][0] == plain[0]
    assert c.msgs[MODEL_REASON][1]["content"] == plain[1]["content"] + (
        "\n\nLook only at is_leap_year: rewrite it if it is wrong, else reply NONE.")


# fix round 1 (review of round 19): I1 nondeterminism, M3 the side in the note, M4 the run's cap, M5 decided fixes kept
RND = "import random\n\n\ndef pick(xs):\n    \"\"\"A random element of xs.\"\"\"\n    return random.choice(xs)\n"
RND_TESTS = TESTS.replace("m.is_leap_year(2024)", "m.pick([3]) == 3")
RND_WRONG = "def pick(xs):\n    \"\"\"A random element of xs.\"\"\"\n    return random.choice(xs[:-1] or xs)\n"


@pytest.mark.parametrize("off", [False, True])
def test_identical_rewrites_of_a_random_function_are_no_evidence(monkeypatch, off):
    if off:   # every round-19 switch off: round 18 (two runs of the rewrites, one of ours)
        for k in ("HARNESS_REVIEW_REASON", "HARNESS_REVIEW_METHODS", "HARNESS_REVIEW_NAMED"):
            monkeypatch.setenv(k, "0")
    for _ in range(2):
        texts, split = review._verdict(RND, RND_TESTS, "pick", RND_WRONG, RND_WRONG)
        assert texts == [] and (off or not split)    # on: ours differs from itself, no dispute either


def test_identical_texts_are_two_runs_and_ours_runs_twice(monkeypatch):
    calls = []
    real = review.behaviour
    monkeypatch.setattr(review, "behaviour", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    trial = review._Trial(SILENT, TESTS, "is_leap_year")
    review._verdict(SILENT, TESTS, "is_leap_year", RIGHT, RIGHT, None, trial)
    assert len(calls) == 4 and calls[0] == calls[1] == SILENT and calls[2] == calls[3]
    monkeypatch.setenv("HARNESS_REVIEW_REASON", "0")
    monkeypatch.setenv("HARNESS_REVIEW_METHODS", "0")
    calls.clear()
    review._verdict(SILENT, TESTS, "is_leap_year", RIGHT, RIGHT)
    assert len(calls) == 3 and calls[0] == SILENT                    # round 18: ours once


def test_random_function_costs_no_third_opinion(tmp_path):
    task = _task(tmp_path, lib=RND, tests=RND_TESTS)
    other = "def pick(xs):\n    \"\"\"A random element of xs.\"\"\"\n    return xs[random.randrange(len(xs))]\n"
    c = Fake(_reply(RND_WRONG), _reply(other), _reply(RND_WRONG))
    assert _go(task, c)[0] == [] and c.calls == [MODEL_FAST, MODEL_ALT] and _lib(task) == RND


def test_note_names_the_side_taken(tmp_path):
    task = _task(tmp_path)
    assert _go(task, Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2)))[1].endswith("sided ['is_leap_year'] with FAST")
    task = _task(tmp_path / "b")
    assert _go(task, Fake(_reply(OTHER), _reply(RIGHT), _reply(RIGHT2)))[1].endswith("sided ['is_leap_year'] with ALT")


def test_third_opinions_have_a_cap_a_run(tmp_path):
    c = Fake(_reply(RIGHT), _reply(OTHER), "NONE")
    for i in range(review.REASON_MAX_CALLS):
        assert review._may_reason(c, [{"content": "x"}])
    assert not review._may_reason(c, [{"content": "x"}])
    c2 = Fake(_reply(RIGHT), _reply(OTHER), "NONE")
    c2._review_reason = (1, review.REASON_MAX_UNITS)
    task = _task(tmp_path)
    changed, note = _go(task, c2)
    assert changed == [] and MODEL_REASON not in c2.calls and "no room" in note
    c3 = Fake(_reply(RIGHT), _reply(OTHER), _reply(RIGHT2))
    _go(_task(tmp_path / "c"), c3)
    assert c3._review_reason == (1, 2)                               # one call, its cost


def test_a_failed_third_opinion_keeps_the_agreed_fix(tmp_path):
    from harness.llm import BudgetExhausted
    clamp = "\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
    tests = TESTS.replace("    def test_leap", "    def test_clamp(self):\n"
                                                "        self.assertEqual(m.clamp(5, 0, 9), 5)"
                                                "\n\n    def test_leap")
    task = _task(tmp_path, lib=SILENT + clamp, tests=tests)
    c1 = "def clamp(x, lo, hi):\n    return min(hi, max(x, lo)) if lo <= hi else lo\n"
    c2 = "def clamp(x, lo, hi):\n    return min(hi, max(x, lo)) if lo <= hi else hi\n"
    c = Fake(_reply(RIGHT, c1), _reply(RIGHT2, c2), BudgetExhausted("out"))
    with pytest.raises(BudgetExhausted):
        _go(task, c)
    assert c.calls == [MODEL_FAST, MODEL_ALT, MODEL_REASON] and "y % 400 == 0" in _lib(task)


# task 87: a function only one of FAST and ALT rewrote (one-sided) goes to the third opinion too (trap1 coding_7024)
SQ_LIB = ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n\n\n"
          "def sum_squares(xs):\n    return sum(x * x for x in xs if x > 0)\n\n\n"
          "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n"
          "    return best\n")
SQ_TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
            "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
            "    def test_sum_squares(self):\n        self.assertEqual(m.sum_squares([1, 2, 3]), 14)\n\n"
            "    def test_minimum(self):\n        self.assertEqual(m.minimum([4, -1, 7]), -1)\n")
SQ_FIX = "def sum_squares(xs):\n    return sum(x * x for x in xs)\n"
SQ_FIX2 = "def sum_squares(xs):\n    return sum(x ** 2 for x in xs)\n"
MIN_GUARD = ("def minimum(xs):\n    if not xs:\n        raise ValueError('empty')\n    best = xs[0]\n    for x in xs:\n"
             "        if x < best:\n            best = x\n    return best\n")      # guard-only: no case


def test_one_sided_rewrite_the_third_opinion_agrees_with_is_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_LONE", "1")   # off by default since round 24
    task = _task(tmp_path, lib=SQ_LIB, tests=SQ_TESTS)
    c = Fake(_reply(SQ_FIX), _reply(MIN_GUARD), _reply(SQ_FIX2))
    changed, note = _go(task, c)
    assert changed == ["sum_squares"] and c.calls == [MODEL_FAST, MODEL_ALT, MODEL_REASON]
    assert "['sum_squares'] one-sided" in note and note.endswith("sided ['sum_squares'] with FAST")
    assert note.startswith("review: replaced ['sum_squares']")
    assert c.msgs[MODEL_REASON][1]["content"].endswith("Look only at sum_squares: rewrite it if it is wrong, "
                                                       "else reply NONE.")          # not the guard-only minimum
    ns = {}
    exec(_lib(task), ns)
    assert ns["sum_squares"]([-2, 3]) == 13 and "best = xs[0]\n" in _lib(task) and "raise" not in _lib(task)


def test_one_sided_rewrite_with_a_silent_third_opinion_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_LONE", "1")   # off by default since round 24
    task = _task(tmp_path, lib=SQ_LIB, tests=SQ_TESTS)
    c = Fake(_reply(SQ_FIX), "NONE", "NONE")
    changed, note = _go(task, c)
    assert changed == [] and _lib(task) == SQ_LIB and c.calls == [MODEL_FAST, MODEL_ALT, MODEL_REASON]
    assert note.startswith("review: no agreement (['sum_squares'] / [])") and "sided []" in note


def test_one_sided_rewrite_the_third_opinion_differs_from_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_LONE", "1")   # off by default since round 24
    task = _task(tmp_path, lib=SQ_LIB, tests=SQ_TESTS)
    other = "def sum_squares(xs):\n    return sum(abs(x) * x for x in xs)\n"
    assert _go(task, Fake(_reply(SQ_FIX), "NONE", _reply(other)))[0] == [] and _lib(task) == SQ_LIB


def test_guard_only_or_known_one_sided_rewrites_make_no_third_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_LONE", "1")   # off by default since round 24
    task = _task(tmp_path, lib=SQ_LIB, tests=SQ_TESTS)
    c = Fake(_reply(MIN_GUARD), "NONE", _reply(SQ_FIX))
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT]
    assert note == "review: no agreement (['minimum'] / []), kept"
    wrong_avg = "def average(xs):\n    return sum(xs) // len(xs)\n"           # the examples' known text
    lib = "def average(xs):\n    return sum(xs) / len(xs)\n"
    tests = SQ_TESTS.split("    def test_total")[0] + (
        "    def test_a(self):\n        self.assertEqual(m.average([2, 4]), 3)\n")
    c = Fake(_reply(wrong_avg), "NONE", _reply(wrong_avg))
    assert _go(_task(tmp_path / "k", lib=lib, tests=tests), c)[0] == [] and MODEL_REASON not in c.calls


def test_one_sided_switch_off_is_round_18(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_REASON", "0")
    task = _task(tmp_path, lib=SQ_LIB, tests=SQ_TESTS)
    c = Fake(_reply(SQ_FIX), _reply(MIN_GUARD), _reply(SQ_FIX2))
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT]
    assert note == "review: no agreement (['sum_squares'] / ['minimum']), kept"


def test_one_sided_only_files_leave_room_for_disputes():
    c = Fake("NONE", "NONE")
    c._review_reason = (review.REASON_MAX_CALLS - review.REASON_DISPUTE_CALLS, 0.0)
    assert not review._may_reason(c, [{"content": "x"}], lone_only=True)
    assert review._may_reason(c, [{"content": "x"}])
    c._review_reason = (0, review.REASON_MAX_UNITS - review.REASON_DISPUTE_UNITS)
    assert not review._may_reason(c, [{"content": "x"}], lone_only=True) and review._may_reason(c, [{"content": "x"}])


def test_one_sided_path_is_off_by_default():
    from harness import review
    assert review.REVIEW_LONE is False

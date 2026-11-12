"""Review of answered coding files for bugs the visible tests do not catch (HARNESS_REVIEW_CODING, expert 9)."""
import asyncio

from harness import config, review, scheduler
from harness.budget import TokenLedger
from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import ChatResult, LLMUnavailable
from harness.solvers.testrun import run_tests
from harness.tasks import Task
from harness.trajectory import Trajectory

TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_leap(self):\n        self.assertTrue(m.is_leap_year(2024))\n\n"
         "    def test_average(self):\n        self.assertEqual(m.average([2, 4, 6]), 4)\n")
SILENT = "def is_leap_year(y):\n    return y % 4 == 0 and y % 100 != 0\n"          # green, wrong for 2000
RIGHT = "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"
RIGHT2 = "def is_leap_year(y):\n    if y % 400 == 0:\n        return True\n    return y % 4 == 0 and y % 100 != 0\n"
OTHER = "def is_leap_year(y):\n    return y % 4 == 0\n"                             # another opinion
RED = "def is_leap_year(y):\n    return False\n"
AVERAGE = "\n\ndef average(xs):\n    return sum(xs) / len(xs)\n"                     # the examples' text
GUARDED = "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n"
MEAN = "\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n"                           # the same under an unknown name
MEAN_GUARDED = "def mean(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n"
MEAN_TESTS = TESTS.replace("m.average(", "m.mean(")


def _reply(*funcs):
    return "\n".join(f"FUNC {f.split('(')[0].split()[-1]}\n{f}" for f in funcs)


def _task(tmp_path, lib=SILENT + AVERAGE, tests=TESTS):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    (d / "answer.txt").write_text("DONE", encoding="utf-8")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


class Fake:
    def __init__(self, fast, alt="NONE"):
        self.replies, self.calls = {MODEL_FAST: fast, MODEL_ALT: alt}, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        r = self.replies[model]
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=1, cost=2)


def _lib(task):
    return (task.workspace / "mylib.py").read_text(encoding="utf-8")


def _go(task, client):
    return asyncio.run(review.review(task, client))


def test_gate_known_texts_only():
    assert not review.needs_review(AVERAGE)                 # every function is an example's text: no call
    assert review.needs_review(SILENT + AVERAGE)            # a function outside the known texts
    assert review.needs_review(AVERAGE.replace("len(xs)", "(len(xs) - 1)"))   # a known name in another form
    assert not review.needs_review("x = 1\n") and not review.needs_review("def f(:\n")


def test_two_agreeing_rewrites_replace_the_silent_bug(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), _reply(RIGHT2))
    changed, note = _go(task, c)
    assert changed == ["is_leap_year"] and c.calls == [MODEL_FAST, MODEL_ALT]
    assert "y % 400 == 0" in _lib(task) and AVERAGE.strip() in _lib(task)
    assert run_tests(task.workspace).ok


def test_none_is_one_call_and_no_change(tmp_path):
    task = _task(tmp_path)
    c = Fake("NONE")
    assert _go(task, c)[0] == [] and c.calls == [MODEL_FAST] and _lib(task) == SILENT + AVERAGE


def test_second_model_silent_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    assert _go(task, Fake(_reply(RIGHT), "NONE"))[0] == [] and _lib(task) == SILENT + AVERAGE


def test_rewrites_that_differ_keep_the_file(tmp_path):
    task = _task(tmp_path)
    assert _go(task, Fake(_reply(RIGHT), _reply(OTHER)))[0] == [] and _lib(task) == SILENT + AVERAGE


def test_rewrite_red_on_visible_tests_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    assert _go(task, Fake(_reply(RED), _reply(RED)))[0] == [] and _lib(task) == SILENT + AVERAGE


def test_guard_only_rewrite_is_no_evidence(tmp_path):
    task = _task(tmp_path, lib=RIGHT + MEAN, tests=MEAN_TESTS)
    assert _go(task, Fake(_reply(MEAN_GUARDED), _reply(MEAN_GUARDED)))[0] == [] and _lib(task) == RIGHT + MEAN


def test_known_text_is_never_replaced(tmp_path):
    wrong = "def average(xs):\n    return sum(xs) // len(xs)\n"
    task = _task(tmp_path)
    assert _go(task, Fake(_reply(wrong), _reply(wrong)))[0] == [] and _lib(task) == SILENT + AVERAGE


def test_method_is_never_replaced(tmp_path):
    lib = "class A:\n    def f(self, x):\n        return x + 1\n"
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.A().f(1), 2)\n")
    task = _task(tmp_path, lib=lib, tests=tests)
    new = "FUNC A.f\ndef f(self, x):\n    return x + 1 if x else 1\n"
    assert _go(task, Fake(new, new))[0] == [] and _lib(task) == lib


def test_unavailable_model_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    changed, note = _go(task, Fake(_reply(RIGHT), LLMUnavailable("down")))
    assert changed == [] and "unavailable" in note and _lib(task) == SILENT + AVERAGE


def test_red_file_is_not_reviewed(tmp_path):
    task = _task(tmp_path, lib=RED + AVERAGE)
    c = Fake(_reply(RIGHT), _reply(RIGHT))
    assert _go(task, c)[0] == [] and c.calls == []


def test_untested_function_is_probed_by_common_shapes(tmp_path):
    lib = RIGHT + "\n\ndef total_of(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"
    fix = "def total_of(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
    fix2 = "def total_of(xs):\n    return sum(xs)\n"
    task = _task(tmp_path, lib=lib, tests=TESTS.split("    def test_average")[0])
    assert _go(task, Fake(_reply(fix), _reply(fix2)))[0] == ["total_of"]


def test_stage_runs_on_answered_tasks_and_knob_turns_it_off(tmp_path, monkeypatch):
    task = _task(tmp_path)
    monkeypatch.setenv("HARNESS_REVIEW_CODING", "0")
    assert scheduler._review_tasks([task]) == []
    monkeypatch.delenv("HARNESS_REVIEW_CODING")
    assert config.review_coding() and scheduler._review_tasks([task]) == [task]
    ledger = TokenLedger(1000, 50)
    traj = Trajectory(tmp_path)
    c = Fake(_reply(RIGHT), _reply(RIGHT2))
    asyncio.run(scheduler._review_stage([task], c, ledger, __import__("time").monotonic(), 3300, {"t": "solved"},
                                        traj, {"stop": False, "traj": traj}))
    traj.close()
    assert "y % 400 == 0" in _lib(task)
    assert "review:t" in (tmp_path / "trajectory.jsonl").read_text(encoding="utf-8")


def test_stage_skips_unanswered_and_known_files(tmp_path):
    task = _task(tmp_path, lib=AVERAGE, tests=TESTS.replace("        self.assertTrue(m.is_leap_year(2024))", "        pass"))
    ledger = TokenLedger(1000, 50)
    traj = Trajectory(tmp_path)
    c = Fake(_reply(RIGHT), _reply(RIGHT))
    asyncio.run(scheduler._review_stage([task], c, ledger, __import__("time").monotonic(), 3300, {"t": "solved"},
                                        traj, {"stop": False, "traj": traj}))
    asyncio.run(scheduler._review_stage([_task(tmp_path / "u")], c, ledger, __import__("time").monotonic(), 3300,
                                        {"t": "unparsed"}, traj, {"stop": False, "traj": traj}))
    traj.close()
    assert c.calls == []


# round 18 review, C1: a broken rewrite is no evidence
LEAP_TESTS = TESTS.split("    def test_average")[0]
ROOT = ("\n\ndef int_sqrt(n):\n    \"\"\"Floor of the square root of n >= 0.\"\"\"\n    r = int(n ** 0.5)\n"
        "    while r * r > n:\n        r -= 1\n    while (r + 1) * (r + 1) <= n:\n        r += 1\n    return r\n")
ISQRT = "def int_sqrt(n):\n    \"\"\"Floor of the square root of n >= 0.\"\"\"\n    return math.isqrt(n)\n"


def test_rewrite_whose_import_is_dropped_keeps_the_file(tmp_path):
    task = _task(tmp_path, lib=RIGHT + ROOT, tests=LEAP_TESTS)
    fast = "FUNC int_sqrt\nimport math\n\n" + ISQRT       # the import line is dropped by apply_patch
    alt = "FUNC int_sqrt\n" + ISQRT                        # forgets the import
    assert _go(task, Fake(fast, alt))[0] == [] and _lib(task) == RIGHT + ROOT


def test_rewrite_whose_helper_is_dropped_keeps_the_file(tmp_path):
    task = _task(tmp_path, lib=RIGHT + ROOT, tests=LEAP_TESTS)
    block = ("FUNC int_sqrt\ndef int_sqrt(n):\n    \"\"\"Floor of the square root of n >= 0.\"\"\"\n"
             "    return _newton(n)\n\n"
             "def _newton(n):\n    x = n\n    y = (x + 1) // 2\n    while y < x:\n        x, y = y, (y + n // y) // 2\n"
             "    return x\n")
    assert review._rewrites(RIGHT + ROOT, block) == {}
    assert _go(task, Fake(block, block))[0] == [] and _lib(task) == RIGHT + ROOT


def test_import_the_file_has_is_no_loss():
    src = "import math\n\n\n" + RIGHT + ROOT
    block = "FUNC int_sqrt\nimport math\n\n" + ISQRT
    assert set(review._rewrites(src, block)) == {"int_sqrt"}


def test_timeout_or_name_error_of_both_rewrites_is_no_evidence(monkeypatch):
    src = RIGHT + ROOT
    n = len(review._battery(None, 1))
    fa = "def int_sqrt(n):\n    return 0\n"
    fb = "def int_sqrt(n):\n    return 1\n"
    for exc in ("<exc:TimeoutError>", "<exc:NameError>", "<exc:MemoryError>", "<exc:Timeout>"):
        seq = iter([["1"] * n, ["1"] * (n - 2) + [exc] * 2, ["1"] * (n - 2) + [exc] * 2])
        monkeypatch.setattr(review, "behaviour", lambda *a, **k: next(seq))
        assert review._evidence(src, LEAP_TESTS, "int_sqrt", fa, fb) == [], exc
    # the same values with no exception: evidence, both rewrites may be stored
    seq = iter([["1"] * n, ["1"] * (n - 2) + ["2"] * 2, ["1"] * (n - 2) + ["2"] * 2])
    monkeypatch.setattr(review, "behaviour", lambda *a, **k: next(seq))
    assert review._evidence(src, LEAP_TESTS, "int_sqrt", fa, fb) == [fa, fb]


# I2: of two agreeing rewrites, only one that raises nowhere ours returns a value is stored
PROFIT = ("def max_profit(prices):\n"
          "    \"\"\"Best profit from one buy then one later sell; 0 when no profit is possible.\"\"\"\n"
          "    if not prices:\n        return 0\n    best = 0\n    lo = prices[0]\n    for p in prices:\n"
          "        best = max(best, p - lo)\n        lo = max(lo, p)\n    return best\n")
PROFIT_NOGUARD = PROFIT.replace("    if not prices:\n        return 0\n", "").replace("lo = max(lo, p)", "lo = min(lo, p)")
PROFIT_FIXED = PROFIT.replace("lo = max(lo, p)", "lo = min(lo, p)")
PROFIT_TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                "    def test_p(self):\n        self.assertEqual(m.max_profit([3, 5]), 2)\n")


def _profit(task):
    ns = {}
    exec(_lib(task), ns)
    return ns["max_profit"]


def test_rewrite_that_drops_our_guard_is_not_stored(tmp_path):
    task = _task(tmp_path, lib=PROFIT, tests=PROFIT_TESTS)
    changed, _ = _go(task, Fake("FUNC max_profit\n" + PROFIT_NOGUARD, "FUNC max_profit\n" + PROFIT_FIXED))
    assert changed == ["max_profit"] and _profit(task)([5, 1, 4]) == 3 and _profit(task)([]) == 0


def test_both_rewrites_drop_our_guard_keeps_the_file(tmp_path):
    task = _task(tmp_path, lib=PROFIT, tests=PROFIT_TESTS)
    reply = "FUNC max_profit\n" + PROFIT_NOGUARD
    assert _go(task, Fake(reply, reply.replace("best = 0", "best = 0 * 1")))[0] == [] and _lib(task) == PROFIT

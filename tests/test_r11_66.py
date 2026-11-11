"""Final review of round 11: a mixed red test on mined texts proves nothing; cut mining makes no text final; the
retry pass does not buy an answered restart again."""
import asyncio
import time

import pytest

from harness.llm import ChatResult, ProxyDown
from harness.llm_routes import MODEL_FAST, fix_coding
from harness.solvers import coding_solver
from harness.tasks import Task

LIB = "def inc(x):\n    return x + 2\n\n\ndef dbl(x):\n    return x * 2\n\n\ndef neg(x):\n    return x\n"
GREEN = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
         "    def test_inc(self):\n        self.assertTrue(mylib.inc(1) > 0)\n\n"
         "    def test_dbl(self):\n        self.assertEqual(mylib.dbl(3), 6)\n\n"
         "    def test_neg(self):\n        self.assertEqual(mylib.neg(%d), %d)\n")
Y_HEAD = "import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
MIXED = Y_HEAD + "    def test_both(self):\n        self.assertEqual(mylib.dbl(mylib.inc(1)), 4)\n"
ALONE = Y_HEAD + "    def test_inc(self):\n        self.assertEqual(mylib.inc(1), 2)\n"


@pytest.fixture
def learned():
    """Names installed by a test leave the module tables afterwards."""
    tables = (coding_solver._CANON, coding_solver._CANON_NODES, coding_solver._CANON_TEXT)
    before = [dict(t) for t in tables]
    final = set(coding_solver._FINAL)
    yield
    for table, old in zip(tables, before):
        table.clear()
        table.update(old)
    coding_solver._FINAL.clear()
    coding_solver._FINAL.update(final)


def _ws(root, name, lib, tests):
    ws = root / name / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8", newline="\n")
    return ws


def _set(root, y_tests):
    """The reviewer's probe: A, B, C hold `inc` with an uncaught bug (x + 2) and `dbl`, all green; Y holds the same
    texts and `y_tests`, feasible with `inc` -> x + 1."""
    wss = [_ws(root, n, LIB, GREEN % (0, 0) if i == 0 else GREEN % (i, -i)) for i, n in enumerate("abc")]
    return wss + [_ws(root, "y", LIB, y_tests)]


def _task(ws):
    return Task(ws.parent.name, "coding", "Fix the bugs in mylib.py", 1.0, "unittest", ws.parent, ws)


# ---------------------------------------------------------------- I1: a mixed red test on mined texts
def test_red_test_mixing_two_mined_final_texts_is_not_infeasible(tmp_path, learned):
    wss = _set(tmp_path, MIXED)
    texts, final = coding_solver.mine_known_texts(wss)
    assert sorted(texts) == ["dbl", "inc"] and final == {"dbl", "inc"}   # the mixed test was never cross-checked
    coding_solver.install_known_texts(texts, final)
    res = coding_solver.solve(_task(wss[-1]), time_limit=60)
    assert res.status == "solved" and res.status != "infeasible"
    assert "x + 1" in (wss[-1] / "mylib.py").read_text(encoding="utf-8")


def test_control_red_test_of_inc_alone_is_cross_checked_as_before(tmp_path, learned):
    wss = _set(tmp_path, ALONE)
    texts, final = coding_solver.mine_known_texts(wss)
    assert sorted(texts) == ["dbl"] and final == {"dbl"}   # the cross-check rejects inc
    coding_solver.install_known_texts(texts, final)
    res = coding_solver.solve(_task(wss[-1]), time_limit=60)
    assert res.status == "solved"
    assert "x + 1" in (wss[-1] / "mylib.py").read_text(encoding="utf-8")


def test_known_text_fails_judges_one_mined_function_but_not_two(learned):
    inc_ok = "def inc(x):\n    return x + 1\n"
    dbl = "def dbl(x):\n    return x * 2\n"
    coding_solver.install_known_texts({"inc": inc_ok.rstrip("\n"), "dbl": dbl.rstrip("\n")}, {"inc", "dbl"})
    lib = inc_ok + "\n\n" + dbl
    one = Y_HEAD + "    def test_inc(self):\n        self.assertEqual(mylib.inc(1), 5)\n"
    two = Y_HEAD + "    def test_both(self):\n        self.assertEqual(mylib.dbl(mylib.inc(1)), 5)\n"
    assert coding_solver.known_text_fails(one, lib, ["test_inc"], {"test_inc": {"inc"}})
    assert not coding_solver.known_text_fails(two, lib, ["test_both"], {"test_both": {"inc", "dbl"}})


# ---------------------------------------------------------------- M1: cut mining makes no text final
MIN_OK = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"
M_HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"


def _min_set(root, n=3):
    return [_ws(root, f"g{i}", MIN_OK, M_HEAD + f"    def test_minimum(self):\n"
                f"        self.assertEqual(m.minimum([{3 + i}, 9, 2]), 2)\n") for i in range(n)]


def test_full_mining_with_short_budget_still_makes_final_texts(tmp_path):
    # every run's timeout is cut by the budget, but each run finishes: the evidence is whole
    texts, final = coding_solver.mine_known_texts(_min_set(tmp_path), budget_s=8.0, run_timeout=10.0)
    assert sorted(texts) == ["minimum"] and final == {"minimum"}


def test_run_lost_to_a_cut_timeout_leaves_texts_weak(tmp_path):
    slow = _ws(tmp_path, "slow", MIN_OK, M_HEAD + "    def test_minimum(self):\n        import time\n"
               "        time.sleep(60)\n        self.assertEqual(m.minimum([1, 2]), 1)\n")
    texts, final = coding_solver.mine_known_texts(_min_set(tmp_path) + [slow], budget_s=6.0, run_timeout=30.0)
    assert sorted(texts) == ["minimum"] and final == set()


def test_loop_cut_by_the_budget_leaves_texts_weak(tmp_path, monkeypatch):
    real, calls = coding_solver._mine_run, []

    def slow_third(ws, timeout):
        run = real(ws, timeout)
        calls.append(ws)
        if len(calls) == 3:
            time.sleep(timeout)   # the budget is spent: the fourth workspace is never run
        return run

    monkeypatch.setattr(coding_solver, "_mine_run", slow_third)
    texts, final = coding_solver.mine_known_texts(_min_set(tmp_path, 4), budget_s=4.0, run_timeout=10.0)
    assert len(calls) == 3 and sorted(texts) == ["minimum"] and final == set()


# ---------------------------------------------------------------- M2: an answered restart is not bought again
MIN_BUG = MIN_OK.replace("x < best", "x > best")
LAST_BUG = "def last_n(xs, n):\n    return xs[:n]\n"
TWO = (M_HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n\n"
       "    def test_last_n(self):\n        self.assertEqual(m.last_n([5, 6, 7, 8], 3), [6, 7, 8])\n")
LAST_FIX = "FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "user": messages[-1]["content"]})
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=10, cost=11)


def test_retry_pass_does_not_repeat_an_answered_restart(tmp_path):
    lib = MIN_BUG + "\n\n" + LAST_BUG
    ws = _ws(tmp_path, "c", lib, TWO)
    t = _task(ws)
    res = coding_solver.solve(t, time_limit=60)
    assert res.extra == {"restart": lib}
    c = FakeClient(["nonsense", ProxyDown("dead")])   # the restart answers, then the proxy dies on the ladder
    out = asyncio.run(fix_coding(t, c, restart=res.extra["restart"]))
    assert out.status == "unparsed" and out.note.startswith("llm unavailable") and len(c.calls) == 2
    assert out.extra["ladder"]["restarted"] is True and out.extra["ladder"]["answered"] == []
    c2 = FakeClient([LAST_FIX])
    again = asyncio.run(fix_coding(t, c2, out.extra["ladder"], restart=res.extra["restart"]))   # as the scheduler
    assert again.status == "solved" and "before mutations" not in again.note
    assert len(c2.calls) == 1 and c2.calls[0]["model"] == MODEL_FAST and "x < best" in c2.calls[0]["user"]


def test_ladder_without_a_restart_call_has_no_restarted_mark(tmp_path):
    lib = MIN_BUG + "\n\n" + LAST_BUG
    t = _task(_ws(tmp_path, "c", lib, TWO))
    coding_solver.solve(t, time_limit=60)
    out = asyncio.run(fix_coding(t, FakeClient([ProxyDown("dead")])))
    assert out.extra == {"ladder": {"answered": [], "base": (t.workspace / "mylib.py").read_text(encoding="utf-8")}}

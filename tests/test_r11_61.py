"""Task 61: function texts mined from the coding tasks of the run join the known texts before phase 1."""
import queue

import pytest

from harness import scheduler
from harness.solvers import coding_solver
from harness.tasks import Task

HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
MAIN = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
TOTAL = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
MIN_BUG = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
MIN_OK = MIN_BUG.replace("x > best", "x < best")
SQ = "def square(x):\n    return x * x\n"
T_MIN = "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n"
T_SQ = "    def test_square(self):\n        self.assertEqual(m.square(3), 9)\n"


def _ws(tmp_path, name, lib, tests):
    ws = tmp_path / name / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (ws / "test_mylib.py").write_text(HEAD + tests + MAIN, encoding="utf-8", newline="\n")
    return ws


def _task(ws, name=None):
    return Task(name or ws.parent.name, "coding", "Fix the bugs in mylib.py", 2.0, "unittest", ws.parent, ws)


@pytest.fixture
def learned():
    """Names installed by a test leave the module tables afterwards."""
    tables = (coding_solver._CANON, coding_solver._CANON_NODES, coding_solver._CANON_TEXT)
    before = [dict(t) for t in tables]
    yield
    for table, old in zip(tables, before):
        table.clear()
        table.update(old)


# ---------------------------------------------------------------- mining
def test_text_held_green_by_two_tasks_is_learned_and_repairs_a_third(tmp_path, learned):
    a = _ws(tmp_path, "a", MIN_OK + "\n\n" + SQ, T_MIN + "\n" + T_SQ)
    b = _ws(tmp_path, "b", SQ + "\n\n" + MIN_OK, T_SQ + "\n" + T_MIN)
    c = _ws(tmp_path, "c", MIN_BUG + "\n\n" + SQ.replace("x * x", "x + x"), T_MIN + "\n" + T_SQ)
    texts = coding_solver.mine_known_texts([a, b, c])
    assert texts == {"minimum": MIN_OK.rstrip("\n"), "square": SQ.rstrip("\n")}
    assert sorted(coding_solver.install_known_texts(texts)) == ["minimum", "square"]
    res = coding_solver.solve(_task(c), time_limit=60)
    assert res.status == "solved" and res.note == "repaired by mutation, canonical" and not res.tentative
    assert (c / "mylib.py").read_text(encoding="utf-8") == MIN_OK + "\n\n" + SQ
    for ws in (a, b, c):   # the runs went to throw-away copies
        assert not (ws / "__pycache__").exists()


def test_one_vote_or_two_equal_forms_are_not_learned(tmp_path):
    ok = _ws(tmp_path, "a", MIN_OK, T_MIN)
    red = _ws(tmp_path, "b", MIN_BUG, T_MIN)                       # its test is red: no vote
    assert coding_solver.mine_known_texts([ok, red]) == {}          # one vote is not enough
    other = "def minimum(xs):\n    return min(xs)\n"                # a second green form with as many votes
    oks = [ok] + [_ws(tmp_path, f"c{i}", MIN_OK + "\n" * (i + 1), T_MIN) for i in range(3)]
    others = [_ws(tmp_path, f"o{i}", other + "\n" * i, T_MIN) for i in range(2)]
    assert coding_solver.mine_known_texts(oks[:2] + others) == {}
    assert coding_solver.mine_known_texts(oks[:3] + others) == {}      # 3 against 2: not twice
    # four against two: twice the other green forms together
    assert coding_solver.mine_known_texts(oks + others) == {"minimum": MIN_OK.rstrip("\n")}


def test_identical_workspaces_count_once(tmp_path):
    a = _ws(tmp_path, "a", MIN_OK, T_MIN)
    b = _ws(tmp_path, "b", MIN_OK, T_MIN)    # a copy of a: not a second witness
    assert coding_solver.mine_known_texts([a, b]) == {}
    c = _ws(tmp_path, "c", MIN_OK, T_MIN.replace("test_minimum", "test_min"))
    assert coding_solver.mine_known_texts([a, b, c]) == {"minimum": MIN_OK.rstrip("\n")}


def test_a_test_red_on_the_very_text_weighs_against_it(tmp_path):
    green = [_ws(tmp_path, f"g{i}", MIN_OK + "\n" * (i + 1), T_MIN) for i in range(4)]
    # the same text red alone in a test: the tests of that task want something else
    red = _ws(tmp_path, "r", MIN_OK, T_MIN.replace("), 2)", "), 3)"))
    assert coding_solver.mine_known_texts(green[:2] + [red]) == {}
    assert coding_solver.mine_known_texts(green[:3] + [red]) == {}
    assert coding_solver.mine_known_texts(green + [red]) == {"minimum": MIN_OK.rstrip("\n")}
    # red in a test that also uses another function is no evidence against minimum
    both = _ws(tmp_path, "b", MIN_OK + "\n\n" + SQ.replace("x * x", "x + x"),
               "    def test_both(self):\n        self.assertEqual(m.minimum([3, 9, 2]) + m.square(3), 11)\n")
    assert coding_solver.mine_known_texts(green[:2] + [both]) == {"minimum": MIN_OK.rstrip("\n")}


def test_functions_that_lean_on_their_module_are_not_learned(tmp_path):
    lean = SQ + "\n\ndef fourth(x):\n    return square(square(x))\n"
    t4 = "    def test_fourth(self):\n        self.assertEqual(m.fourth(2), 16)\n"
    got = coding_solver.mine_known_texts([_ws(tmp_path, "f", lean + "\n\n" + TOTAL, t4),
                                          _ws(tmp_path, "g", lean + "\n\n" + TOTAL + "\n", t4)])
    assert got == {}                                                # square is untested here
    const = "LIMIT = 10\n\n\ndef capped(x):\n    return min(x, LIMIT)\n"
    tc = "    def test_capped(self):\n        self.assertEqual(m.capped(30), 10)\n"
    assert coding_solver.mine_known_texts([_ws(tmp_path, "h", const, tc), _ws(tmp_path, "i", const + "\n", tc)]) == {}
    deco = "import functools\n\n\n@functools.lru_cache()\ndef square(x):\n    return x * x\n"
    assert coding_solver.mine_known_texts([_ws(tmp_path, "j", deco, T_SQ), _ws(tmp_path, "k", deco + "\n", T_SQ)]) == {}


def test_example_names_keep_their_example_texts(tmp_path, learned):
    other_total = "def total(xs):\n    return sum(xs)\n"
    tt = "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
    assert coding_solver.mine_known_texts([_ws(tmp_path, "a", other_total, tt),
                                           _ws(tmp_path, "b", other_total + "\n", tt)]) == {}
    before = coding_solver._CANON["total"]
    assert coding_solver.install_known_texts({"total": other_total, "x": "x = 1\n", "y": "def z():\n    pass\n",
                                              "w": "@d\ndef w():\n    pass\n", "v": "def v(:\n"}) == []
    assert coding_solver._CANON["total"] == before


def test_runs_that_fail_or_time_out_give_no_vote(tmp_path):
    hang = "def minimum(xs):\n    while True:\n        pass\n\n\n" + SQ
    broken = "import nosuchmodule_xyz\n\n\n" + SQ
    ws = [_ws(tmp_path, "a", hang, T_SQ + T_MIN), _ws(tmp_path, "b", broken, T_SQ),
          _ws(tmp_path, "c", SQ, T_SQ)]
    assert coding_solver.mine_known_texts(ws, run_timeout=3) == {}   # square has one vote only, from c


def test_mining_keeps_to_its_time_budget(tmp_path, monkeypatch):
    from harness.solvers import testrun
    calls = []
    real = testrun.run_tests
    monkeypatch.setattr(testrun, "run_tests", lambda *a, **k: calls.append(k["timeout"]) or real(*a, **k))
    ws = [_ws(tmp_path, n, MIN_OK + "\n" * i, T_MIN) for i, n in enumerate("abc")]
    assert coding_solver.mine_known_texts(ws, budget_s=0.1) == {} and calls == []
    coding_solver.mine_known_texts(ws, budget_s=30, run_timeout=7)
    assert len(calls) == 3 and all(t <= 7 for t in calls)


def test_mined_text_is_kept_only_when_it_turns_red_green_and_breaks_nothing(tmp_path, learned):
    coding_solver.install_known_texts({"minimum": MIN_OK})
    # accepted: red turns green, nothing breaks
    ws = _ws(tmp_path, "a", MIN_BUG, T_MIN + "    def test_two(self):\n        self.assertEqual(m.minimum([4, 4]), 4)\n")
    res = coding_solver.solve(_task(ws), time_limit=60)
    assert res.status == "solved" and res.note == "repaired by mutation, canonical"
    # refused: the substitution turns the green test_other red (it also uses square, so it proves nothing)
    lib = MIN_BUG + "\n\n" + SQ
    tests = T_MIN + "    def test_other(self):\n        self.assertEqual(m.minimum([1, 5]) + m.square(1), 6)\n"
    ws = _ws(tmp_path, "c", lib, tests)
    res = coding_solver.solve(_task(ws), time_limit=60)
    assert "canonical" not in (res.note or "") and res.status != "infeasible"
    assert MIN_OK not in (ws / "mylib.py").read_text(encoding="utf-8")
    # another shape (a guard) is never replaced by the mined text
    guarded = "def minimum(xs):\n    return min(xs) if xs else 1\n"
    ws = _ws(tmp_path, "b", guarded, T_MIN.replace("[3, 9, 2]), 2", "[]), 0"))
    res = coding_solver.solve(_task(ws), time_limit=60)
    assert "canonical" not in (res.note or "")
    assert (ws / "mylib.py").read_text(encoding="utf-8") != MIN_OK


# ---------------------------------------------------------------- scheduler
def test_mine_texts_switch_and_minimum_task_count(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: calls.append((len(ws), budget_s))
                        or {"minimum": MIN_OK})
    tasks = [_task(_ws(tmp_path, n, MIN_OK, T_MIN)) for n in "abc"]
    assert scheduler._mine_texts(tasks[:2]) == {} and calls == []     # fewer than MINE_MIN_TASKS
    assert scheduler._mine_texts(tasks) == {"minimum": MIN_OK} and calls == [(3, coding_solver.MINE_BUDGET_S)]
    import time
    scheduler._mine_texts(tasks, time.monotonic(), 100.0)              # a short run: a tenth of the time left
    assert calls[-1][1] <= 10.0
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "0")
    assert scheduler._mine_texts(tasks) == {} and len(calls) == 2
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: 1 / 0)
    monkeypatch.delenv("HARNESS_MINE_TEXTS")
    assert scheduler._mine_texts(tasks) == {}                         # a failure never stops the run


def test_switched_off_the_phase1_call_is_round_10(tmp_path, monkeypatch):
    seen = {}

    def fake_phase1(tasks, workers, t0, deadline_s, on_result, **kw):
        seen["kw"] = kw
        seen["n"] = len(tasks)

    mined = []
    monkeypatch.setattr(scheduler, "_phase1", fake_phase1)
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: mined.append(1) or {"minimum": MIN_OK})
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "log"))
    tasks_dir = tmp_path / "tasks"
    for n in "abc":
        _ws(tasks_dir, n, MIN_OK, T_MIN)
        (tasks_dir / n / "task.json").write_text(
            '{"task_id": "%s", "task_type": "coding", "instruction": "Fix", "reward": 2, "check": "unittest"}' % n,
            encoding="utf-8")
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "0")
    scheduler.run(tasks_dir, 1000, None)
    assert seen == {"kw": {}, "n": 3} and mined == []
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "1")
    scheduler.run(tasks_dir, 1000, None)
    assert seen == {"kw": {"known_texts": {"minimum": MIN_OK}}, "n": 3} and mined == [1]


def test_phase1_child_installs_the_mined_texts(tmp_path, learned):
    ws = _ws(tmp_path, "c", MIN_BUG, T_MIN)
    q = queue.Queue()
    scheduler._phase1_target(q, _task(ws), None, {"minimum": MIN_OK})
    tid, res = q.get_nowait()
    assert tid == "c" and res.status == "solved" and res.note == "repaired by mutation, canonical"
    assert coding_solver._CANON_TEXT["minimum"] == MIN_OK


def test_run_mines_then_repairs_through_spawned_children(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "log"))
    tasks_dir = tmp_path / "tasks"
    libs = {"a": MIN_OK + "\n\n" + SQ, "b": SQ + "\n\n" + MIN_OK, "c": MIN_BUG + "\n\n" + SQ}
    for n, lib in libs.items():
        _ws(tasks_dir, n, lib, T_MIN + "\n" + T_SQ)
        (tasks_dir / n / "task.json").write_text(
            '{"task_id": "%s", "task_type": "coding", "instruction": "Fix", "reward": 2, "check": "unittest"}' % n,
            encoding="utf-8")
    counts = scheduler.run(tasks_dir, 1000, None)
    assert counts["solved"] == 3 and counts["spent"] == 0
    assert (tasks_dir / "c" / "workspace" / "mylib.py").read_text(encoding="utf-8") == MIN_OK + "\n\n" + SQ
    import json
    rows = [json.loads(x) for x in (tmp_path / "log" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r.get("task_id") == "c" and r.get("note") == "repaired by mutation, canonical" for r in rows)

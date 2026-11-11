"""Task 61: function texts mined from the coding tasks of the run join the known texts before phase 1."""
import json
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
JSON = '{"task_id": "%s", "task_type": "coding", "instruction": "Fix", "reward": 2, "check": "unittest"}'


def t_min(i):
    """A test of minimum that differs (as code) from the others: distinct tasks are not copies."""
    return f"    def test_minimum(self):\n        self.assertEqual(m.minimum([{3 + i}, 9, 2]), 2)\n"


def cp(op):
    return f"def count_pos(xs):\n    n = 0\n    for x in xs:\n        if x {op} 0:\n            n += 1\n    return n\n"


def _ws(tmp_path, name, lib, tests):
    ws = tmp_path / name / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (ws / "test_mylib.py").write_text(HEAD + tests + MAIN, encoding="utf-8", newline="\n")
    return ws


def _task(ws, name=None):
    return Task(name or ws.parent.name, "coding", "Fix the bugs in mylib.py", 2.0, "unittest", ws.parent, ws)


def mined(*a, **k):
    return coding_solver.mine_known_texts(*a, **k)[0]


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


# ---------------------------------------------------------------- mining
def test_text_held_green_by_three_tasks_is_final_and_repairs_a_fourth(tmp_path, learned):
    a = _ws(tmp_path, "a", MIN_OK + "\n\n" + SQ, T_MIN + "\n" + T_SQ)
    b = _ws(tmp_path, "b", SQ + "\n\n" + MIN_OK, T_SQ + "\n" + T_MIN)
    d = _ws(tmp_path, "d", MIN_OK + "\n\n" + SQ, t_min(1) + "\n" + T_SQ)
    c = _ws(tmp_path, "c", MIN_BUG + "\n\n" + SQ.replace("x * x", "x + x"), T_MIN + "\n" + T_SQ)
    texts, final = coding_solver.mine_known_texts([a, b, c, d])
    assert texts == {"minimum": MIN_OK.rstrip("\n"), "square": SQ.rstrip("\n")} and final == {"minimum", "square"}
    assert sorted(coding_solver.install_known_texts(texts, final)) == ["minimum", "square"]
    res = coding_solver.solve(_task(c), time_limit=60)
    assert res.status == "solved" and res.note == "repaired by mutation, canonical" and not res.tentative
    assert (c / "mylib.py").read_text(encoding="utf-8") == MIN_OK + "\n\n" + SQ
    for ws in (a, b, c, d):   # the runs went to throw-away copies
        assert not (ws / "__pycache__").exists()


def test_two_votes_or_a_rival_form_substitute_but_are_not_final(tmp_path, monkeypatch, learned):
    two = [_ws(tmp_path, f"g{i}", MIN_OK, t_min(i)) for i in range(2)]
    assert coding_solver.mine_known_texts(two) == ({"minimum": MIN_OK.rstrip("\n")}, set())
    other = "def minimum(xs):\n    return min(xs)\n"
    four = [_ws(tmp_path, f"h{i}", MIN_OK, t_min(i)) for i in range(4)] + [_ws(tmp_path, "o", other, T_MIN)]
    assert coding_solver.mine_known_texts(four) == ({"minimum": MIN_OK.rstrip("\n")}, set())   # 4 against 1
    # a weak text repairs, but the repair is checked: the model sees the real buggy text
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    coding_solver.install_known_texts({"minimum": MIN_OK})
    ws = _ws(tmp_path, "c", MIN_BUG, T_MIN)
    res = coding_solver.solve(_task(ws), time_limit=60)
    assert res.status == "solved" and res.tentative and "canonical" not in res.note
    hint = res.extra.get("verify") or res.extra.get("ambiguous")
    assert [h["func"] for h in hint] == ["minimum"] and hint[0]["buggy"] == MIN_BUG.rstrip("\n")
    assert "x < best" in (ws / "mylib.py").read_text(encoding="utf-8")


def test_one_vote_or_two_equal_forms_are_not_learned(tmp_path):
    ok = _ws(tmp_path, "a", MIN_OK, T_MIN)
    red = _ws(tmp_path, "b", MIN_BUG, t_min(1))                    # its test is red: no vote
    assert mined([ok]) == {}                                        # one vote is not enough
    other = "def minimum(xs):\n    return min(xs)\n"                # a second green form with as many votes
    oks = [ok] + [_ws(tmp_path, f"c{i}", MIN_OK, t_min(i + 1)) for i in range(3)]
    others = [_ws(tmp_path, f"o{i}", other, t_min(i + 1)) for i in range(2)]
    assert mined(oks[:2] + others) == {}
    assert mined(oks[:3] + others) == {}      # 3 against 2: not twice
    assert mined(oks + others) == {"minimum": MIN_OK.rstrip("\n")}   # four against two
    assert mined(oks + [red]) == {"minimum": MIN_OK.rstrip("\n")}    # the red one passes the cross-check


def test_copies_of_a_task_count_once(tmp_path):
    a = _ws(tmp_path, "a", MIN_OK, T_MIN)
    b = _ws(tmp_path, "b", MIN_OK, T_MIN)    # a byte copy of a: not a second witness
    assert mined([a, b]) == {}
    # nor a copy that differs only in layout, comments or blank lines
    c = _ws(tmp_path, "c", "# copy\n" + MIN_OK.replace("best = xs[0]", "best = xs[0]  # first") + "\n\n",
            "\n" + T_MIN.replace("[3, 9, 2]", "[3,9,2]"))
    assert mined([a, c]) == {}
    d = _ws(tmp_path, "d", MIN_OK, t_min(1))
    assert mined([a, b, c, d]) == {"minimum": MIN_OK.rstrip("\n")}


def test_a_test_red_on_the_text_refuses_it(tmp_path):
    green = [_ws(tmp_path, f"g{i}", MIN_OK, t_min(i)) for i in range(4)]
    # the same text red alone in a test: the tests of that task want something else
    red = _ws(tmp_path, "r", MIN_OK, T_MIN.replace("), 2)", "), 3)"))
    assert mined(green + [red]) == {}
    # red alone with another bug, and still red with the candidate: refused too
    bug = _ws(tmp_path, "s", MIN_BUG, T_MIN.replace("), 2)", "), 3)"))
    assert mined(green + [bug]) == {}
    # red in a test that also uses another function is no evidence against minimum
    both = _ws(tmp_path, "b", MIN_OK + "\n\n" + SQ.replace("x * x", "x + x"),
               "    def test_both(self):\n        self.assertEqual(m.minimum([3, 9, 2]) + m.square(3), 11)\n")
    assert mined(green[:2] + [both]) == {"minimum": MIN_OK.rstrip("\n")}


def _count_pos_set(tmp_path, d_test):
    """The reviewer's case: A and B hold the same uncaught bug (>=), C the right text, D another bug (<)."""
    t_cp = "    def test_cp(self):\n        self.assertEqual(m.count_pos([1, 2, -3]), 2)\n"
    a = _ws(tmp_path, "a", cp(">=") + "\n\n" + SQ.replace("x * x", "x + x + 1"), t_cp + "\n" + T_SQ)
    b = _ws(tmp_path, "b", SQ.replace("x * x", "x * 2") + "\n\n" + cp(">="), T_SQ + "\n" + t_cp)
    c = _ws(tmp_path, "c", cp(">") + "\n\n" + SQ, t_cp + "\n" + T_SQ)
    d = _ws(tmp_path, "d", cp("<") + "\n\n" + SQ, d_test + "\n" + T_SQ.replace("(3), 9", "(4), 16"))
    return [a, b, c, d]


@pytest.mark.parametrize("d_input", ["[5, 6, -1]", "[0, 5, 6, -1]"])
def test_an_uncaught_bug_in_two_tasks_never_gives_a_final_wrong_fix(tmp_path, learned, d_input):
    d_test = f"    def test_cp(self):\n        self.assertEqual(m.count_pos({d_input}), 2)\n"
    sets = _count_pos_set(tmp_path, d_test)
    texts, final = coding_solver.mine_known_texts(sets)
    assert "count_pos" not in final
    if "0" in d_input.split(",")[0]:
        assert "count_pos" not in texts      # D's test stays red with `>=`: refused by the cross-check
    coding_solver.install_known_texts(texts, final)
    res = coding_solver.solve(_task(sets[3]), time_limit=60)
    assert res.status == "solved" and "canonical" not in res.note
    text = (sets[3] / "mylib.py").read_text(encoding="utf-8")
    assert res.tentative or ">= 0" not in text   # never final with the wrong text


def test_functions_that_lean_on_their_module_are_not_learned(tmp_path):
    lean = SQ + "\n\ndef fourth(x):\n    return square(square(x))\n"
    t4 = "    def test_fourth(self):\n        self.assertEqual(m.fourth(2), 16)\n"
    got = mined([_ws(tmp_path, "f", lean + "\n\n" + TOTAL, t4),
                 _ws(tmp_path, "g", lean + "\n\n" + TOTAL, t4.replace("(2), 16", "(1), 1"))])
    assert got == {}                                                # square is untested here
    const = "LIMIT = 10\n\n\ndef capped(x):\n    return min(x, LIMIT)\n"
    tc = "    def test_capped(self):\n        self.assertEqual(m.capped(30), 10)\n"
    assert mined([_ws(tmp_path, "h", const, tc), _ws(tmp_path, "i", const, tc.replace("30", "31"))]) == {}
    deco = "import functools\n\n\n@functools.lru_cache()\ndef square(x):\n    return x * x\n"
    assert mined([_ws(tmp_path, "j", deco, T_SQ), _ws(tmp_path, "k", deco, T_SQ.replace("(3), 9", "(2), 4"))]) == {}


def test_a_function_used_only_by_skipped_tests_gets_no_vote(tmp_path):
    skip = "    @unittest.skip('later')\n" + T_MIN
    assert mined([_ws(tmp_path, "a", MIN_BUG, skip), _ws(tmp_path, "b", MIN_BUG, skip.replace("[3,", "[4,"))]) == {}
    xfail = "    @unittest.expectedFailure\n" + T_MIN
    assert mined([_ws(tmp_path, "c", MIN_BUG, xfail), _ws(tmp_path, "d", MIN_BUG, xfail.replace("[3,", "[4,"))]) == {}
    inner = T_MIN.replace("        self.assertEqual", "        self.skipTest('x')\n        self.assertEqual")
    assert mined([_ws(tmp_path, "e", MIN_BUG, inner), _ws(tmp_path, "f", MIN_BUG, inner.replace("[3,", "[4,"))]) == {}
    raised = T_MIN.replace("        self.assertEqual", "        raise unittest.SkipTest('x')\n        self.assertEqual")
    assert mined([_ws(tmp_path, "g", MIN_BUG, raised), _ws(tmp_path, "h", MIN_BUG, raised.replace("[3,", "[4,"))]) == {}


def test_cross_check_covers_tests_mixed_with_final_known_texts(tmp_path):
    green = [_ws(tmp_path, f"g{i}", MIN_OK, t_min(i)) for i in range(3)]
    mix = "    def test_mix(self):\n        self.assertEqual(m.minimum([3, 9, 2]) + m.total([1]), {})\n"
    # minimum with another bug next to the known total as it stands; with the candidate the mixed test stays red
    red = _ws(tmp_path, "r", MIN_BUG + "\n\n" + TOTAL, mix.format(4))
    assert mined(green + [red]) == {}
    # ... and turns green: the candidate passes
    fixed = _ws(tmp_path, "f", MIN_BUG + "\n\n" + TOTAL, mix.format(3))
    assert coding_solver.mine_known_texts(green + [fixed]) == ({"minimum": MIN_OK.rstrip("\n")}, {"minimum"})
    # a mixed test with a function that is not a known text is no evidence either way
    other = _ws(tmp_path, "o", MIN_BUG + "\n\n" + SQ,
                "    def test_mix(self):\n        self.assertEqual(m.minimum([3, 9, 2]) + m.square(1), 4)\n")
    assert mined(green + [other]) == {"minimum": MIN_OK.rstrip("\n")}


def test_example_names_keep_their_example_texts(tmp_path, learned):
    other_total = "def total(xs):\n    return sum(xs)\n"
    tt = "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
    assert mined([_ws(tmp_path, "a", other_total, tt), _ws(tmp_path, "b", other_total, tt.replace("3]), 6", "4]), 7"))]) == {}
    before = coding_solver._CANON["total"]
    assert coding_solver.install_known_texts({"total": other_total, "x": "x = 1\n", "y": "def z():\n    pass\n",
                                              "w": "@d\ndef w():\n    pass\n", "v": "def v(:\n"}, ["total"]) == []
    assert coding_solver._CANON["total"] == before and "total" in coding_solver._FINAL


def test_runs_that_fail_or_time_out_give_no_vote(tmp_path):
    hang = "def minimum(xs):\n    while True:\n        pass\n\n\n" + SQ
    broken = "import nosuchmodule_xyz\n\n\n" + SQ
    ws = [_ws(tmp_path, "a", hang, T_SQ + T_MIN), _ws(tmp_path, "b", broken, T_SQ),
          _ws(tmp_path, "c", SQ, T_SQ)]
    assert mined(ws, run_timeout=3) == {}   # square has one vote only, from c


def test_mining_keeps_to_its_time_budget(tmp_path, monkeypatch):
    from harness.solvers import testrun
    calls = []
    real = testrun.run_tests
    monkeypatch.setattr(testrun, "run_tests", lambda *a, **k: calls.append(k["timeout"]) or real(*a, **k))
    ws = [_ws(tmp_path, n, MIN_OK, t_min(i)) for i, n in enumerate("abc")]
    assert mined(ws, budget_s=0.1) == {} and calls == []
    mined(ws, budget_s=30, run_timeout=7)
    assert len(calls) == 3 and all(t <= 7 for t in calls)


def test_cross_check_out_of_time_refuses(tmp_path, monkeypatch):
    green = [_ws(tmp_path, f"g{i}", MIN_OK, t_min(i)) for i in range(3)]
    red = _ws(tmp_path, "r", MIN_BUG, T_MIN)
    real = coding_solver._cross_check
    monkeypatch.setattr(coding_solver, "_cross_check", lambda n, t, runs, stop, rt: real(n, t, runs, 0.0, rt))
    assert mined(green + [red]) == {}
    monkeypatch.setattr(coding_solver, "_cross_check", real)
    assert coding_solver.mine_known_texts(green + [red]) == ({"minimum": MIN_OK.rstrip("\n")}, {"minimum"})


def test_mined_text_is_kept_only_when_it_turns_red_green_and_breaks_nothing(tmp_path, learned):
    coding_solver.install_known_texts({"minimum": MIN_OK}, ["minimum"])
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
PAIR = ({"minimum": MIN_OK}, {"minimum"})


def test_mine_texts_switch_and_minimum_task_count(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: calls.append((len(ws), budget_s))
                        or PAIR)
    tasks = [_task(_ws(tmp_path, n, MIN_OK, T_MIN)) for n in "abc"]
    assert scheduler._mine_texts(tasks[:2]) == {} and calls == []     # fewer than MINE_MIN_TASKS
    assert scheduler._mine_texts(tasks) == {"texts": {"minimum": MIN_OK}, "final": ["minimum"]}
    assert calls == [(3, coding_solver.MINE_BUDGET_S)]
    import time
    scheduler._mine_texts(tasks, time.monotonic(), 100.0)              # a short run: a tenth of the time left
    assert calls[-1][1] <= 10.0
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: ({}, set()))
    assert scheduler._mine_texts(tasks) == {}                         # nothing mined: phase 1 as in round 10
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

    calls = []
    monkeypatch.setattr(scheduler, "_phase1", fake_phase1)
    monkeypatch.setattr(coding_solver, "mine_known_texts", lambda ws, budget_s: calls.append(1) or PAIR)
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "log"))
    tasks_dir = tmp_path / "tasks"
    for n in "abc":
        _ws(tasks_dir, n, MIN_OK, T_MIN)
        (tasks_dir / n / "task.json").write_text(JSON % n, encoding="utf-8")
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "0")
    scheduler.run(tasks_dir, 1000, None)
    assert seen == {"kw": {}, "n": 3} and calls == []
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "1")
    scheduler.run(tasks_dir, 1000, None)
    assert seen == {"kw": {"known_texts": {"texts": {"minimum": MIN_OK}, "final": ["minimum"]}}, "n": 3}
    assert calls == [1]


@pytest.mark.parametrize("final", [["minimum"], []])
def test_phase1_child_installs_the_mined_texts(tmp_path, learned, monkeypatch, final):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    ws = _ws(tmp_path, "c", MIN_BUG, T_MIN)
    q = queue.Queue()
    scheduler._phase1_target(q, _task(ws), None, {"texts": {"minimum": MIN_OK}, "final": final})
    tid, res = q.get_nowait()
    assert tid == "c" and res.status == "solved"
    assert (res.note == "repaired by mutation, canonical") == bool(final)
    assert coding_solver._CANON_TEXT["minimum"] == MIN_OK and ("minimum" in coding_solver._FINAL) == bool(final)


def test_run_mines_then_repairs_through_spawned_children(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "log"))
    tasks_dir = tmp_path / "tasks"
    libs = {"a": (MIN_OK + "\n\n" + SQ, T_MIN), "b": (SQ + "\n\n" + MIN_OK, t_min(1)),
            "d": (MIN_OK + "\n\n" + SQ, t_min(2)), "c": (MIN_BUG + "\n\n" + SQ, T_MIN)}
    for n, (lib, tm) in libs.items():
        _ws(tasks_dir, n, lib, tm + "\n" + T_SQ)
        (tasks_dir / n / "task.json").write_text(JSON % n, encoding="utf-8")
    counts = scheduler.run(tasks_dir, 1000, None)
    assert counts["solved"] == 4 and counts["spent"] == 0
    assert (tasks_dir / "c" / "workspace" / "mylib.py").read_text(encoding="utf-8") == MIN_OK + "\n\n" + SQ
    rows = [json.loads(x) for x in (tmp_path / "log" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r.get("task_id") == "c" and r.get("note") == "repaired by mutation, canonical" for r in rows)

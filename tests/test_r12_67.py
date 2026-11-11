"""Round 12, task 67 (expert pass 5): a function of the organizers' examples that the visible tests leave green but
that is not the known text takes the known text (`coding_solver._restore_known`): stronger hidden tests would catch
the slip."""
from pathlib import Path

import pytest

from harness.solvers import coding_solver
from harness.solvers.coding_solver import solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task

HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TAIL = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
T_MAX = "    def test_maximum(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 9)\n"
T_AVG = "    def test_average(self):\n        self.assertEqual(m.average([2, 4, 6]), 4)\n"
T_TOTAL = "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
T_FIRST = "    def test_first_n(self):\n        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])\n"
MAX_OK = "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
MAX_ZERO = MAX_OK.replace("best = xs[0]", "best = 0")          # green on [3, 9, 2], wrong on negatives
AVG_OK = "def average(xs):\n    return sum(xs) / len(xs)\n"
AVG_FLOOR = "def average(xs):\n    return sum(xs) // len(xs)\n"    # green on [2, 4, 6], wrong on [1, 2]
TOTAL_OK = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
TOTAL_BUG = TOTAL_OK.replace("+=", "-=")


def _task(tmp_path: Path, lib: str, tests: str) -> Task:
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (d / "workspace" / "test_mylib.py").write_text(HEAD + tests + TAIL, encoding="utf-8", newline="\n")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


def _lib(t: Task) -> str:
    return (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_green_file_with_a_hidden_slip_takes_the_known_text(tmp_path):
    t = _task(tmp_path, MAX_ZERO + "\n\n" + AVG_OK, T_MAX + "\n" + T_AVG)
    res = solve(t)
    assert res.status == "solved" and res.answer == "DONE" and not res.tentative
    assert "known texts restored" in res.note and "maximum" in res.note
    assert "best = xs[0]" in _lib(t) and run_tests(t.workspace).ok


def test_repair_also_restores_the_green_slip_and_stays_canonical(tmp_path):
    t = _task(tmp_path, AVG_FLOOR + "\n\n" + TOTAL_BUG + "\n\n" + MAX_OK, T_AVG + "\n" + T_TOTAL + "\n" + T_MAX)
    res = solve(t)
    assert res.status == "solved" and "canonical" in res.note and not res.tentative and res.extra is None
    lib = _lib(t)
    assert "sum(xs) / len(xs)" in lib and "s += x" in lib and run_tests(t.workspace).ok


def test_green_file_with_the_known_texts_is_untouched(tmp_path):
    t = _task(tmp_path, MAX_OK + "\n\n" + AVG_OK, T_MAX + "\n" + T_AVG)
    before = _lib(t)
    res = solve(t)
    assert res.note == "tests already pass" and _lib(t) == before


def test_a_guarded_function_is_not_replaced(tmp_path):
    guarded = "def average(xs):\n    return sum(xs) / len(xs) if xs else 0\n"
    t = _task(tmp_path, guarded + "\n\n" + MAX_OK, T_AVG + "\n" + T_MAX)
    res = solve(t)
    assert res.note == "tests already pass" and "if xs else 0" in _lib(t)


def test_another_shape_under_a_known_name_is_not_replaced(tmp_path):
    other = "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) // len(xs)\n"
    t = _task(tmp_path, other + "\n\n" + TOTAL_BUG, T_AVG + "\n" + T_TOTAL)
    res = solve(t)
    assert res.status == "solved" and "if not xs" in _lib(t) and "sum(xs) // len(xs)" in _lib(t)


def test_known_text_that_breaks_a_visible_test_is_not_taken(tmp_path):
    # the visible test wants four items of first_n(xs, 3): the file's own text stays, the known text would turn it red
    first_plus = "def first_n(xs, n):\n    return xs[:n + 1]\n"
    tests = "    def test_first_n(self):\n        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7, 8])\n"
    t = _task(tmp_path, first_plus + "\n\n" + MAX_OK, tests + "\n" + T_MAX)
    res = solve(t)
    assert res.note == "tests already pass" and "xs[:n + 1]" in _lib(t)


# ---------------------------------------------------------------- round 12 additions
DOUBLE_BUG = "def double(x):\n    return x * 3\n"
T_DOUBLE = "    def test_double(self):\n        self.assertEqual(m.double(2), 4)\n"
MIN_OK = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"
T_MIN = "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n"


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


def test_a_mined_text_never_restores_a_green_function(tmp_path, learned):
    # a final mined text of minimum is installed; the file's minimum differs (an extra - 0 that changes nothing on
    # the visible test) and stays: only the examples' texts restore a green function
    coding_solver.install_known_texts({"minimum": MIN_OK}, ["minimum"])
    other = MIN_OK.replace("best = xs[0]", "best = xs[0] - 0")
    t = _task(tmp_path, other + "\n\n" + MAX_OK, T_MIN + "\n" + T_MAX)
    res = solve(t)
    assert res.note == "tests already pass" and "xs[0] - 0" in _lib(t)


def test_decorated_or_other_parameters_are_not_replaced(tmp_path):
    deco = "import functools\n\n\n@functools.lru_cache(None)\ndef average(xs):\n    return sum(xs) // len(xs)\n"
    t = _task(tmp_path, deco, T_AVG.replace("[2, 4, 6]", "(2, 4, 6)"))
    assert solve(t).note == "tests already pass" and "//" in _lib(t)
    params = "def average(ys):\n    return sum(ys) // len(ys)\n"
    t2 = _task(tmp_path / "b", params, T_AVG)
    assert solve(t2).note == "tests already pass" and "//" in _lib(t2)


def test_restored_names_are_no_repair_for_the_2d_hint(tmp_path, monkeypatch):
    # double is repaired by the search, average (a green slip) takes the known text: only double goes to 2d,
    # with its real buggy text; the restored average is not shown as changed
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    t = _task(tmp_path, AVG_FLOOR + "\n\n" + DOUBLE_BUG, T_AVG + "\n" + T_DOUBLE)
    res = solve(t)
    assert res.status == "solved" and "canonical" not in res.note
    lib = _lib(t)
    assert "sum(xs) / len(xs)" in lib and "x * 3" not in lib and run_tests(t.workspace).ok
    hint = res.extra.get("verify") or res.extra.get("ambiguous")
    assert [h["func"] for h in hint] == ["double"] and hint[0]["buggy"] == DOUBLE_BUG.rstrip("\n")


class _FakeRun:
    """Green on every source for `runs` runs, then out of time."""
    verify_timeout = 1.0

    def __init__(self, runs):
        self.left, self.seen = runs, []

    def run(self, src, only, timeout):
        if self.left == 0:
            raise coding_solver._OutOfTime()
        self.left -= 1
        self.seen.append(src)
        return type("R", (), {"ok": True})()


def test_out_of_time_keeps_what_is_restored():
    src = AVG_FLOOR + "\n\n" + MAX_ZERO
    out, names = coding_solver._restore_known(_FakeRun(1), src)
    assert names == ["average"] and out == AVG_OK + "\n\n" + MAX_ZERO
    out, names = coding_solver._restore_known(_FakeRun(5), src)
    assert names == ["average", "maximum"] and out == AVG_OK + "\n\n" + MAX_OK


def test_known_texts_in_place_cost_no_run():
    fake = _FakeRun(0)   # any run would raise
    src = MAX_OK + "\n\n" + AVG_OK + "\n\n" + TOTAL_OK + "\n\n" + DOUBLE_BUG
    assert coding_solver._restore_known(fake, src) == (src, []) and not fake.seen



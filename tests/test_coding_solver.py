import shutil
from pathlib import Path

from harness.solvers.coding_solver import solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task, copy_task

FIX = Path(__file__).parent / "fixtures" / "examples"


def _fixture_task(name: str, tmp_path: Path) -> Task:
    d = FIX / name
    src = Task(name, "coding", "fix", 2.0, "unittest", d, d / "workspace")
    return copy_task(src, tmp_path)


def _custom(tmp_path: Path, mylib: str, tests: str) -> Task:
    d = tmp_path / "custom"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(mylib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("custom", "coding", "fix", 2.0, "unittest", d, d / "workspace")


def test_run_tests_reports_failures(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    before = sorted(p.name for p in t.workspace.iterdir())
    rep = run_tests(t.workspace)
    assert not rep.ok and sorted(rep.failed) == ["test_reverse_words", "test_total"]
    assert sorted(p.name for p in t.workspace.iterdir()) == before
    rep = run_tests(t.workspace, only=["test_total"])
    assert rep.failed == ["test_total"]


def test_run_tests_timeout(tmp_path):
    t = _custom(tmp_path, "def f():\n    while True: pass\n",
                "import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                "    def test_f(self):\n        m.f()\n")
    rep = run_tests(t.workspace, timeout=1)
    assert not rep.ok and rep.failed == ["<timeout>"]


def test_fixes_both_examples(tmp_path):
    for name in ("coding_0003", "coding_0007"):
        t = _fixture_task(name, tmp_path)
        tests_before = (t.workspace / "test_mylib.py").read_bytes()
        res = solve(t)
        assert res.status == "solved" and res.answer == "DONE", res.note
        assert run_tests(t.workspace).ok
        assert (t.workspace / "test_mylib.py").read_bytes() == tests_before


def test_fix_restores_original_semantics(tmp_path):
    t = _fixture_task("coding_0007", tmp_path)
    assert solve(t).status == "solved"
    text = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "x % 2 == 0" in text and "x > best" in text


def test_infinite_loop_mutant_does_not_hang(tmp_path):
    mylib = ("def f(n):\n    i = 0\n    while i <= n:\n        i += 1\n    return i\n")
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(5), 5)\n")
    t = _custom(tmp_path, mylib, tests)
    res = solve(t, time_limit=60)
    assert res.status == "solved", res.note
    assert "while i < n" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_unfixable_is_unparsed(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    tp = t.workspace / "test_mylib.py"
    tp.write_text(tp.read_text(encoding="utf-8").replace("m.total([1, 2, 3]), 6", "m.total([1, 2, 3]), 1000"),
                  encoding="utf-8")
    before = (t.workspace / "mylib.py").read_bytes()
    res = solve(t, time_limit=60)
    assert res.status == "unparsed" and res.note
    assert (t.workspace / "mylib.py").read_bytes() == before


def test_no_pycache_left(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    assert solve(t).status == "solved"
    assert not (t.workspace / "__pycache__").exists()


def test_bug_in_helper(tmp_path):
    mylib = ("def g(x):\n    return x + 1\n\n\ndef f(x):\n    return g(x) * 2\n")
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(3), 8)\n"
             "    def test_g(self):\n        self.assertEqual(m.g(3), 4)\n")
    t = _custom(tmp_path, mylib.replace("x + 1", "x - 1"), tests)
    res = solve(t)
    assert res.status == "solved", res.note
    assert "x + 1" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_bug_in_helper_only_caller_tested(tmp_path):
    mylib = ("def g(x):\n    return x - 1\n\n\ndef f(x):\n    return g(x) * 2\n")
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(3), 8)\n")
    t = _custom(tmp_path, mylib, tests)
    assert solve(t).status == "solved"


def test_already_green_is_done(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    assert solve(t).status == "solved"
    assert solve(t).answer == "DONE"


def test_missing_mylib_is_unparsed(tmp_path):
    t = _custom(tmp_path, "", "")
    (t.workspace / "mylib.py").unlink()
    assert solve(t).status == "unparsed"


def _hidden_check(tmp_path, mylib, name, visible, hidden):
    head = "import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n    def test_%s(self):\n" % name
    line = "        self.assertEqual(m.%s(%s), %r)\n"
    t = _custom(tmp_path, mylib, head + line % ((name,) + visible))
    assert solve(t).status == "solved"
    (t.workspace / "test_mylib.py").write_text(
        head + "".join(line % ((name, a, e)) for a, e in [visible] + hidden), encoding="utf-8")
    assert run_tests(t.workspace).ok


def test_regression_gcd_remainder_constant(tmp_path):
    bug = ("def gcd(a, b):\n    d = min(a, b)\n    while d > 1:\n"
           "        if a % d == 0 and b % d == 1:\n            return d\n        d -= 1\n    return 1\n")
    _hidden_check(tmp_path, bug, "gcd", ("20, 5", 5),
                  [("12, 18", 6), ("7, 13", 1), ("9, 27", 9)])


def test_regression_total_initial_value(tmp_path):
    bug = "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"
    _hidden_check(tmp_path, bug, "total", ("[1, 2, 3]", 6), [("[]", 0), ("[5]", 5), ("[2, 2]", 4)])


def test_regression_first_n_slice_bound(tmp_path):
    bug = "def first_n(xs, n):\n    return xs[:n - 1]\n"
    _hidden_check(tmp_path, bug, "first_n", ("[4, 3, 2, 1], 4", [4, 3, 2, 1]),
                  [("[1, 2, 3], 2", [1, 2]), ("[], 3", []), ("[7, 8], 1", [7])])


def test_contradictory_visible_tests_are_infeasible(tmp_path):
    lib = "def f(x):\n    return x\n"
    tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(7), 7)\n        self.assertEqual(m.f(7), 8)\n")
    r = solve(_custom(tmp_path, lib, tests))
    assert r.status == "infeasible" and r.answer is None


def _cd(body: str, header: str = "import mylib as m\n") -> bool:
    from harness.solvers.coding_solver import contradictory_tests
    return contradictory_tests(header + body)


def test_same_call_same_value_is_not_contradictory():
    ok = ("class T:\n    def test_a(self):\n        self.assertEqual(m.f(1), 2)\n"
          "        self.assertEqual(m.f(1), 2)\n        self.assertEqual(m.f(2), 3)\n")
    assert not _cd(ok)
    assert _cd("class T:\n    def test_a(self):\n        self.assertEqual(m.f(1), 2)\n"
               "        self.assertEqual(m.f(1), 3)\n")
    assert _cd("def test_a():\n    self.assertEqual(f(1), 2)\n    self.assertEqual(f(1), 3)\n",
               "from mylib import f\n")
    assert not _cd("def broken(:")


def test_feasible_patterns_are_not_contradictory():
    # stateful method on an instance
    assert not _cd("class T:\n    def test_a(self):\n        s = m.Stack()\n"
                   "        self.assertEqual(s.pop(), 2)\n        self.assertEqual(s.pop(), 1)\n")
    # argument mutated between asserts
    assert not _cd("class T:\n    def test_a(self):\n        xs = [1]\n"
                   "        self.assertEqual(m.total(xs), 1)\n        xs.append(2)\n"
                   "        self.assertEqual(m.total(xs), 3)\n")
    # rebound variable
    assert not _cd("class T:\n    def test_a(self):\n        x = 1\n"
                   "        self.assertEqual(m.f(x), 1)\n        x = 5\n        self.assertEqual(m.f(x), 5)\n")
    # same test name in different classes
    assert not _cd("class A:\n    def test_x(self):\n        self.assertEqual(m.f(1), 1)\n"
                   "class B:\n    def test_x(self):\n        self.assertEqual(m.f(1), 2)\n")
    # 1 and 1.0 are equal for assertEqual
    assert not _cd("class T:\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n"
                   "        self.assertEqual(m.f(1), 1.0)\n")
    # unhashable literal does not crash
    assert not _cd("class T:\n    def test_a(self):\n        self.assertEqual(m.f(1), {[1]: 2})\n")
    # stateful no-arg module function
    assert not _cd("class T:\n    def test_a(self):\n        self.assertEqual(m.next_id(), 1)\n"
                   "        self.assertEqual(m.next_id(), 2)\n")


def test_contradiction_never_preempts_a_repair(tmp_path):
    lib = "def f(x):\n    return x - 1\n"
    tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(7), 8)\n")
    assert solve(_custom(tmp_path, lib, tests)).status == "solved"

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

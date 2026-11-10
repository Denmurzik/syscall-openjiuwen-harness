"""Known organizer functions: a repair that restores their correct text skips 2c and 2d."""
from pathlib import Path

import pytest

from harness.solvers.coding_solver import _CANON, _all_canonical, solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task, copy_task

FIX = Path(__file__).parent / "fixtures" / "examples"

TOTAL_BUG = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
TOTAL_OK = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
EVEN_BUG = "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n"
EVEN_OK_UNPARSED = "def count_even(xs):\n    return sum((1 for x in xs if x % 2 == 0))\n"  # ast.unparse form
OTHER = "\n\ndef double(x):\n    return x * 2\n"


def _fixture_task(name: str, tmp_path: Path) -> Task:
    d = FIX / name
    return copy_task(Task(name, "coding", "fix", 2.0, "unittest", d, d / "workspace"), tmp_path)


def test_canon_has_the_six_example_functions():
    assert sorted(_CANON) == ["average", "count_even", "first_n", "maximum", "reverse_words", "total"]


@pytest.mark.parametrize("name", ["coding_0003", "coding_0007"])
def test_organizer_examples_are_final_without_model_checks(tmp_path, monkeypatch, name):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")   # the check would otherwise run (2d)
    t = _fixture_task(name, tmp_path / name)
    res = solve(t)
    assert res.status == "solved" and res.answer == "DONE", res.note
    assert not res.tentative and res.extra is None and "canonical" in res.note
    assert run_tests(t.workspace).ok


def test_unknown_function_still_goes_to_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    d = tmp_path / "custom"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text("def has(xs, v):\n    return v not in xs\n", encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(
        "import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
        "    def test_has(self):\n        self.assertEqual(mylib.has([1, 2], 2), True)\n\n"
        "    def test_has_more(self):\n        self.assertEqual(mylib.has([3], 3), True)\n", encoding="utf-8")
    res = solve(Task("custom", "coding", "fix", 2.0, "unittest", d, d / "workspace"))
    assert res.status == "solved" and res.tentative and "canonical" not in res.note


def test_all_canonical_accepts_restored_known_functions():
    assert _all_canonical(TOTAL_BUG + OTHER, TOTAL_OK + OTHER)
    assert _all_canonical(TOTAL_BUG + "\n\n" + EVEN_BUG, TOTAL_OK + "\n\n" + EVEN_OK_UNPARSED)


def test_known_name_with_another_body_is_not_canonical():
    assert not _all_canonical(TOTAL_BUG, "def total(xs):\n    return sum(xs)\n")
    assert not _all_canonical(TOTAL_BUG, TOTAL_OK.replace("(xs)", "(ys)").replace("in xs", "in ys"))


def test_one_unknown_changed_function_makes_it_not_canonical():
    assert not _all_canonical(TOTAL_BUG + OTHER, TOTAL_OK + OTHER.replace("x * 2", "x + x"))


def test_nothing_changed_or_layout_changed_is_not_canonical():
    assert not _all_canonical(TOTAL_OK, TOTAL_OK)
    assert not _all_canonical(TOTAL_BUG, TOTAL_OK + OTHER)               # a function was added
    assert not _all_canonical(TOTAL_BUG + OTHER, TOTAL_OK)               # a function was removed
    assert not _all_canonical("K = 1\n\n" + TOTAL_BUG, "K = 2\n\n" + TOTAL_OK)   # a statement changed


def test_changed_class_or_async_def_is_not_canonical():
    cls_bug = "class C:\n    def total(self, xs):\n        return -sum(xs)\n"
    cls_ok = "class C:\n    def total(self, xs):\n        return sum(xs)\n"
    assert not _all_canonical(cls_bug, cls_ok)
    assert not _all_canonical("async " + TOTAL_BUG, "async " + TOTAL_OK)


def test_parse_failure_is_not_canonical():
    assert not _all_canonical(TOTAL_BUG, "def total(xs:\n")
    assert not _all_canonical("def total(xs:\n", TOTAL_OK)

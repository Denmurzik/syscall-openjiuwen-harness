"""Task 57: a red assertion that is false whatever mylib does proves the coding task infeasible."""
import json
from pathlib import Path

import pytest

from harness.solvers.coding_solver import provable_contradiction, solve
from harness.tasks import Task

EX = Path(__file__).parent / "fixtures" / "examples"
HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TAIL = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
GOOD = {
    "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
}
ASSERT = {
    "total": "self.assertEqual(m.total([1, 2, 3]), 6)",
    "maximum": "self.assertEqual(m.maximum([3, 9, 2]), 9)",
}
LIB = GOOD["total"] + "\n\n" + GOOD["maximum"]


def _tests(names, extra=""):
    return HEAD + "\n".join(f"    def test_{n}(self):\n        {ASSERT[n]}\n" for n in names) + extra + TAIL


def _task(tmp_path, lib, tests) -> Task:
    ws = tmp_path / "t" / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("coding_x", "coding", "fix", 2.0, "unittest", tmp_path / "t", ws)


@pytest.mark.parametrize("extra", [
    "    def test_impossible(self):\n        self.assertEqual(1, 2)\n",
    "    def test_impossible(self):\n        self.assertTrue(False)\n",
    "    def test_impossible(self):\n        self.assertNotEqual(m.total([1, 2, 3]), m.total([1, 2, 3]))\n",
    "    def test_small(self):\n        self.assertLess(m.maximum([3, 9, 2]), 5)\n",
    "    def test_big(self):\n        self.assertGreater(m.total([1, 2, 3]), 10)\n",
    "    def test_not(self):\n        self.assertNotEqual(m.total([1, 2, 3]), 6)\n",
    "    def test_false(self):\n        self.assertFalse(m.total([1, 2, 3]))\n",
    "    def test_le(self):\n        self.assertLessEqual(m.maximum([3, 9, 2]), m.total([1, 2, 3]))\n",
    "    def test_ge(self):\n        self.assertGreaterEqual(m.total([1, 2, 3]), m.maximum([3, 9, 2]))\n",
])
def test_unsatisfiable_assertion_is_a_proof(extra):
    assert provable_contradiction(_tests(["total", "maximum"], extra), LIB)


@pytest.mark.parametrize("extra", [
    "",
    "    def test_ok(self):\n        self.assertEqual(2, 2)\n",
    "    def test_other_input(self):\n        self.assertLess(m.maximum([1, 2]), 5)\n",          # not the pinned call
    "    def test_small(self):\n        self.assertLess(m.maximum([3, 9, 2]), 50)\n",             # holds for the pinned value
    "    def test_true(self):\n        self.assertTrue(m.total([1, 2, 3]))\n",
    "    def test_cmp(self):\n        self.assertTrue(m.total([1, 2, 3]) == 7)\n",                # not a plain value: no proof
    "    def test_msg(self):\n        self.assertEqual(1, 2, 'why')\n",                           # three arguments: left alone
    "    def test_kw(self):\n        self.assertLess(m.total(xs=[1, 2, 3]), 5)\n",               # another call form: not pinned
    "    def test_type(self):\n        self.assertLess(m.total([1, 2, 3]), 'a')\n",              # no verdict on 6 < 'a'
    "    def test_in(self):\n        self.assertIn(m.total([1, 2, 3]), [1, 2])\n",              # not one of the eight kinds
])
def test_no_proof_without_certainty(extra):
    assert not provable_contradiction(_tests(["total", "maximum"], extra), LIB)


def test_no_proof_when_the_callee_is_not_pure():
    lib = "K = 0\n\n\ndef total(xs):\n    return sum(xs) + K\n\n\n" + GOOD["maximum"]
    tests = _tests(["total", "maximum"], "    def test_big(self):\n        self.assertGreater(m.total([1, 2, 3]), 10)\n")
    assert not provable_contradiction(tests, lib)


def test_no_proof_when_the_pin_is_in_a_duplicated_test():
    # the pin sits in a test whose name occurs twice: it may never run, so it pins nothing
    tests = (HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
             "    def test_big(self):\n        self.assertGreater(m.total([1, 2, 3]), 10)\n\n\n"
             "class U(unittest.TestCase):\n    def test_total(self):\n        self.assertEqual(m.total([1]), 1)\n" + TAIL)
    assert not provable_contradiction(tests, LIB)


def test_no_proof_outside_the_strict_file_shape():
    tests = _tests(["total"], "    def test_x(self):\n        if True:\n            self.assertEqual(1, 2)\n")
    assert not provable_contradiction(tests, LIB)
    tests = "import os\n" + _tests(["total"], "    def test_x(self):\n        self.assertEqual(1, 2)\n")
    assert not provable_contradiction(tests, LIB)


def test_unsatisfiable_task_is_infeasible_only_after_the_repair_fails(tmp_path):
    lib = GOOD["total"].replace("s += x", "s -= x") + "\n\n" + GOOD["maximum"]
    t = _task(tmp_path, lib, _tests(["total", "maximum"], "    def test_impossible(self):\n        self.assertEqual(1, 2)\n"))
    res = solve(t, time_limit=60)
    assert res.status == "infeasible" and res.answer is None
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == lib   # nothing written for an infeasible task
    t2 = _task(tmp_path / "b", lib, _tests(["total", "maximum"]))
    assert solve(t2).status == "solved"


def test_pinned_comparison_task_is_infeasible(tmp_path):
    lib = GOOD["total"] + "\n\n" + GOOD["maximum"]
    tests = _tests(["total", "maximum"], "    def test_small(self):\n        self.assertLess(m.maximum([3, 9, 2]), 5)\n")
    assert solve(_task(tmp_path, lib, tests), time_limit=60).status == "infeasible"


def test_examples_are_never_a_proof():
    for d in sorted(EX.glob("coding_*")):
        ws = d / "workspace"
        assert not provable_contradiction((ws / "test_mylib.py").read_text(encoding="utf-8"),
                                          (ws / "mylib.py").read_text(encoding="utf-8")), d.name

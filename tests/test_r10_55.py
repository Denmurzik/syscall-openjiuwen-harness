"""Task 55: the known text of the organizers' functions is tried before the mutation search."""
import json
import shutil
from pathlib import Path

import pytest

from harness.solvers import coding_solver
from harness.solvers.coding_solver import _canonical_substitute, solve
from harness.tasks import Task

EX = Path(__file__).parent / "fixtures" / "examples"
HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TAIL = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
GOOD = {
    "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    "average": "def average(xs):\n    return sum(xs) / len(xs)\n",
    "first_n": "def first_n(xs, n):\n    return xs[:n]\n",
    "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
    "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
}
ASSERT = {
    "total": "self.assertEqual(m.total([1, 2, 3]), 6)",
    "average": "self.assertEqual(m.average([2, 4, 6]), 4)",
    "first_n": "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])",
    "maximum": "self.assertEqual(m.maximum([3, 9, 2]), 9)",
    "reverse_words": "self.assertEqual(m.reverse_words('c a b'), 'b a c')",
    "count_even": "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)",
}


def _tests(names, extra=""):
    return HEAD + "\n".join(f"    def test_{n}(self):\n        {ASSERT[n]}\n" for n in names) + extra + TAIL


def _task(tmp_path, lib, tests) -> Task:
    ws = tmp_path / "t" / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("coding_x", "coding", "fix", 2.0, "unittest", tmp_path / "t", ws)


@pytest.mark.parametrize("name,bug", [
    ("total", "def total(xs):\n    s = 0\n    for x in xs:\n        s = x\n    return s\n"),          # no single mutation
    ("average", "def average(xs):\n    return sum(xs) / 2\n"),                                        # mutation overfits: / 3
    ("first_n", "def first_n(xs, n):\n    return xs[-n:]\n"),
    ("maximum", "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return x\n"),
    ("reverse_words", "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n"),
    ("count_even", "def count_even(xs):\n    return sum(1 for x in xs)\n"),
])
def test_known_function_gets_the_known_text(tmp_path, name, bug):
    names = [name] + [n for n in GOOD if n != name]
    lib = "\n\n".join((bug if n == name else GOOD[n]).rstrip("\n") for n in names) + "\n"
    t = _task(tmp_path, lib, _tests(names))
    res = solve(t)
    assert res.status == "solved" and not res.tentative and res.note.endswith("canonical")
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == "\n\n".join(GOOD[n].rstrip("\n") for n in names) + "\n"


def test_known_name_with_another_meaning_is_left_to_the_search(tmp_path):
    # `total` here counts the elements: the known text fails its test, so it is not kept
    lib = "def total(xs):\n    return len(xs) + 1\n"
    tests = HEAD + "    def test_total(self):\n        self.assertEqual(m.total([5, 5, 5]), 3)\n" + TAIL
    t = _task(tmp_path, lib, tests)
    res = solve(t)
    assert res.status == "solved" and "canonical" not in res.note
    assert "len(xs)" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_known_function_with_passing_test_is_not_touched(tmp_path):
    # average is written differently and its test passes: only total (failing) is replaced
    odd_average = "def average(xs):\n    return sum(xs) / len(xs) if xs else 0\n"
    lib = GOOD["total"].replace("s += x", "s = x") + "\n\n" + odd_average
    t = _task(tmp_path, lib, _tests(["total", "average"]))
    assert solve(t).status == "solved"
    assert odd_average in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_known_text_that_breaks_a_passing_test_is_not_kept(tmp_path):
    # the known text turns test_empty green but test_shifted red: not a strict improvement, never kept
    lib = "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"
    tests = (HEAD + "    def test_shifted(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n\n"
             "    def test_empty(self):\n        self.assertEqual(m.total([]), 0)\n" + TAIL)
    t = _task(tmp_path, lib, tests)
    assert _canonical_substitute(lib, {"total"}) == GOOD["total"]   # a candidate, so the rule is what refuses it
    res = solve(t)
    assert res.status == "unparsed" and "canonical" not in res.note   # no single mutation fixes both either
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == lib   # the known text is not on disk


def test_substitute_needs_the_known_signature():
    assert _canonical_substitute("def total(xs, start):\n    return start\n", {"total"}) is None
    assert _canonical_substitute(GOOD["total"], {"total"}) is None            # already the known text
    assert _canonical_substitute(GOOD["total"].replace("s += x", "s -= x"), set()) is None
    assert _canonical_substitute("@staticmethod\n" + GOOD["total"].replace("s += x", "s -= x"), {"total"}) is None


def test_substitute_replaces_only_the_named_function():
    lib = GOOD["total"].replace("s += x", "s -= x") + "\n\ndef average(xs):\n    return sum(xs) / 2\n"
    out = _canonical_substitute(lib, {"total"})
    assert out == GOOD["total"] + "\n\ndef average(xs):\n    return sum(xs) / 2\n"


def test_unknown_function_still_goes_the_old_way(tmp_path):
    lib = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
    tests = HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n" + TAIL
    res = solve(_task(tmp_path, lib, tests))
    assert res.status == "solved" and "canonical" not in res.note


def test_out_of_time_keeps_the_known_text_accepted_so_far(tmp_path, monkeypatch):
    # average is replaced first (sorted order); the run for total runs out of time: the partial repair stays
    lib = "def average(xs):\n    return sum(xs) / 2\n\n\n" + GOOD["total"].replace("s += x", "s = x")
    t = _task(tmp_path, lib, _tests(["average", "total"]))
    real = coding_solver._Run.run
    calls = []

    def run(self, src, only, timeout):
        calls.append(src)
        if len(calls) == 3:
            raise coding_solver._OutOfTime()
        return real(self, src, only, timeout)
    monkeypatch.setattr(coding_solver._Run, "run", run)
    res = solve(t)
    assert res.status == "solved" and res.tentative and "partial" in res.note
    written = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert GOOD["average"] in written and "s = x" in written


def test_examples_still_solved(tmp_path):
    for name in ("coding_0003", "coding_0007"):
        d = tmp_path / name
        shutil.copytree(EX / name, d)
        meta = json.loads((d / "task.json").read_text())
        res = solve(Task(name, "coding", meta["instruction"], 2.0, "unittest", d, d / "workspace"))
        assert res.status == "solved" and res.note == "repaired by mutation, canonical"


# ------------------------------------------------------------ same name, more meaning: only the known shape is replaced

MORE_MEANING = {   # the visible test cannot tell these from the known text, the hidden tests may
    "guard": ("average", "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) / 2\n", "if not xs"),
    "docstring": ("total", 'def total(xs):\n    """Sum of xs; None entries are skipped."""\n    s = 0\n    for x in xs:\n'
                  '        if x is None:\n            continue\n        s = x\n    return s\n', "x is None"),
    "abs": ("maximum", "def maximum(xs):\n    best = abs(xs[0])\n    for x in xs:\n        if abs(x) < best:\n"
            "            best = abs(x)\n    return best\n", "abs("),
    "negative_n": ("first_n", "def first_n(xs, n):\n    if n < 0:\n        return []\n    return xs[:n + 1]\n", "n < 0"),
    "round": ("average", "def average(xs):\n    return round(sum(xs) / 2)\n", "round("),
}


@pytest.mark.parametrize("case", sorted(MORE_MEANING))
def test_function_with_more_meaning_is_not_replaced(tmp_path, case):
    name, lib, mark = MORE_MEANING[case]
    assert _canonical_substitute(lib, {name}) is None
    t = _task(tmp_path, lib, _tests([name]))
    res = solve(t)
    assert "canonical" not in res.note                     # never final without a second opinion
    assert mark in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_same_shape_with_another_call_argument_is_replaced():
    # accepted risk: split(' ') keeps the known shape and calls, so the known split() takes its place
    lib = "def reverse_words(s):\n    return ' '.join(s.split(' '))\n"
    assert _canonical_substitute(lib, {"reverse_words"}) == GOOD["reverse_words"]


def test_assign_for_augassign_is_the_same_shape():
    lib = GOOD["total"].replace("s += x", "s = x")
    assert _canonical_substitute(lib, {"total"}) == GOOD["total"]


MINIMUM = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"


def test_known_text_and_a_search_fix_of_an_unknown_function(tmp_path, monkeypatch):
    # total gets the known text, minimum is fixed by the search: only minimum, with its real text, goes to 2d
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = GOOD["total"].replace("s += x", "s -= x") + "\n\n" + MINIMUM
    tests = HEAD + ("    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
                    "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n") + TAIL
    t = _task(tmp_path, lib, tests)
    res = solve(t)
    assert res.status == "solved" and res.tentative and "canonical" not in res.note
    hint = res.extra.get("verify") or res.extra.get("ambiguous")
    assert [h["func"] for h in hint] == ["minimum"] and hint[0]["buggy"] == MINIMUM.rstrip("\n")
    written = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert written.startswith(GOOD["total"]) and "x < best" in written


def test_known_text_changed_again_by_the_search_shows_the_real_original(tmp_path, monkeypatch):
    # the known total turns test_sum green; the search then changes it (s = 1) for the odd test_seven:
    # the check must show the model the original `s = x`, not the known text, as the buggy function
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = GOOD["total"].replace("s += x", "s = x")
    tests = HEAD + ("    def test_seven(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n\n"
                    "    def test_sum(self):\n        self.assertTrue(m.total([1, 1]) >= 2)\n") + TAIL
    t = _task(tmp_path, lib, tests)
    res = solve(t)
    assert res.status == "solved" and res.note == "repaired by mutation, unchecked"
    [item] = res.extra["verify"]
    assert item["func"] == "total" and item["buggy"] == lib.rstrip("\n")

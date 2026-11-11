"""Task 62: a test red on the known correct text, or a test file that does not parse, is infeasible at 0 units."""
import ast

import pytest

from harness.solvers import coding_solver
from harness.tasks import Task

HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
MAIN = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
TOTAL = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
MAXIMUM = "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
MIN_BUG = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
MIN_OK = MIN_BUG.replace("x > best", "x < best")


def _coding(tmp_path, lib, tests, reward=2.0, raw=None):
    d = tmp_path / "c"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    if raw is not None:
        (d / "workspace" / "test_mylib.py").write_bytes(raw)
    else:
        (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8", newline="\n")
    return Task("c", "coding", "Fix the bugs in mylib.py", reward, "unittest", d, d / "workspace")


@pytest.fixture
def learned():
    tables = (coding_solver._CANON, coding_solver._CANON_NODES, coding_solver._CANON_TEXT)
    before = [dict(t) for t in tables]
    final = set(coding_solver._FINAL)
    yield
    for table, old in zip(tables, before):
        table.clear()
        table.update(old)
    coding_solver._FINAL.clear()
    coding_solver._FINAL.update(final)


# ---------------------------------------------------------------- red on the known text
def test_wrong_expected_value_for_a_known_function_is_infeasible(tmp_path):
    # the buggy total gets the known text; test_total still asks for 7: no ladder, no partial answer
    tests = (HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n\n"
             "    def test_total_small(self):\n        self.assertEqual(m.total([1, 2]), 3)\n" + MAIN)
    t = _coding(tmp_path, TOTAL.replace("+=", "-="), tests)
    res = coding_solver.solve(t, time_limit=60)
    assert res.status == "infeasible" and "known correct text" in res.note
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == TOTAL.replace("+=", "-=")   # nothing written


def test_known_text_already_in_the_file_with_a_red_test_is_infeasible(tmp_path):
    tests = HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n" + MAIN
    res = coding_solver.solve(_coding(tmp_path, TOTAL, tests), time_limit=60)
    assert res.status == "infeasible" and "known correct text" in res.note


def test_known_text_that_turns_a_green_test_red_is_infeasible(tmp_path):
    # maximum must be the minimum on another input: with the known text test_other goes red
    tests = (HEAD + "    def test_maximum(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 9)\n\n"
             "    def test_other(self):\n        self.assertEqual(m.maximum([3, 9, 2, 1]), 1)\n" + MAIN)
    res = coding_solver.solve(_coding(tmp_path, MAXIMUM.replace("x > best", "x < best"), tests), time_limit=60)
    assert res.status == "infeasible"


def test_unknown_function_and_guarded_variant_are_not_judged(tmp_path):
    lib = "def minimum(xs):\n    return 0\n"
    tests = HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 1000)\n" + MAIN
    assert coding_solver.solve(_coding(tmp_path / "a", lib, tests), time_limit=60).status != "infeasible"
    # an average with a guard is not the known text and is not replaced (shape): its odd test is the model's business
    lib = "def average(xs):\n    return sum(xs) / len(xs) if xs else 1\n"
    tests = HEAD + "    def test_average(self):\n        self.assertEqual(m.average([]), 0)\n" + MAIN
    assert coding_solver.solve(_coding(tmp_path / "b", lib, tests), time_limit=60).status != "infeasible"
    # a test that also uses a function without a known text is not judged either
    lib = TOTAL + "\n\ndef twice(x):\n    return x\n"
    tests = HEAD + "    def test_both(self):\n        self.assertEqual(m.total([1, 2]) + m.twice(2), 7)\n" + MAIN
    assert coding_solver.solve(_coding(tmp_path / "c", lib, tests), time_limit=60).status != "infeasible"


def test_red_test_that_also_needs_an_unknown_buggy_function_is_not_judged(tmp_path):
    # the known total is right; the red test needs the buggy unknown `double` too: the search repairs double
    lib = TOTAL.replace("+=", "-=") + "\n\ndef double(x):\n    return x + 2\n"
    tests = (HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
             "    def test_mix(self):\n        self.assertEqual(m.double(m.total([1, 2])), 6)\n\n"
             "    def test_double(self):\n        self.assertEqual(m.double(4), 8)\n" + MAIN)
    t = _coding(tmp_path, lib, tests)
    res = coding_solver.solve(t, time_limit=60)
    assert res.status == "solved"
    text = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "s += x" in text and ("x * 2" in text or "x + x" in text)


def test_known_text_fails_needs_the_plain_file_shape():
    tests = HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n" + MAIN
    refs = coding_solver._test_refs(tests)
    assert coding_solver.known_text_fails(tests, TOTAL, ["test_total"], refs)
    assert not coding_solver.known_text_fails(tests, TOTAL, [], refs)
    assert not coding_solver.known_text_fails(tests, TOTAL, ["<timeout>"], refs)
    assert not coding_solver.known_text_fails(tests, TOTAL + "\ntotal = max\n", ["test_total"], refs)
    assert not coding_solver.known_text_fails(tests, "import os\n" + TOTAL, ["test_total"], refs)
    assert not coding_solver.known_text_fails(tests, TOTAL + "\n\ndef total(xs):\n    return 7\n", ["test_total"], refs)
    assert not coding_solver.known_text_fails(tests, "def total(xs):\n    return sum(xs)\n", ["test_total"], refs)
    patched = tests.replace("class T", "m.total = max\n\n\nclass T")
    assert not coding_solver.known_text_fails(patched, TOTAL, ["test_total"], coding_solver._test_refs(patched))
    setup = tests.replace("    def test_total", "    def setUp(self):\n        pass\n\n    def test_total")
    assert not coding_solver.known_text_fails(setup, TOTAL, ["test_total"], coding_solver._test_refs(setup))
    twice = tests.replace("class T(", "class U(unittest.TestCase):\n    def test_total(self):\n        pass\n\n\nclass T(")
    assert not coding_solver.known_text_fails(twice, TOTAL, ["test_total"], coding_solver._test_refs(twice))
    assert not coding_solver.known_text_fails("def broken(:\n", TOTAL, ["test_total"], refs)


def test_known_text_that_leans_on_another_function_proves_nothing(learned):
    lib = TOTAL.replace("+=", "-=") + "\n\ndef average(xs):\n    return total(xs) / len(xs)\n"
    tests = HEAD + "    def test_average(self):\n        self.assertEqual(m.average([2, 4, 6]), 4)\n" + MAIN
    refs = coding_solver._test_refs(tests)
    coding_solver._CANON["average"] = ast.dump(ast.parse(lib).body[1])   # as if this text were the known one
    assert not coding_solver.known_text_fails(tests, lib, ["test_average"], refs)


def test_a_final_mined_text_counts_as_known(tmp_path, learned):
    coding_solver.install_known_texts({"minimum": MIN_OK}, ["minimum"])
    tests = HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 3)\n" + MAIN
    res = coding_solver.solve(_coding(tmp_path, MIN_BUG, tests), time_limit=60)
    assert res.status == "infeasible" and "known correct text" in res.note


def test_a_mined_text_with_weak_evidence_never_proves_infeasibility(tmp_path, learned):
    coding_solver.install_known_texts({"minimum": MIN_OK})
    tests = HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 3)\n" + MAIN
    assert coding_solver.solve(_coding(tmp_path / "a", MIN_BUG, tests), time_limit=60).status != "infeasible"
    assert coding_solver.solve(_coding(tmp_path / "b", MIN_OK, tests), time_limit=60).status != "infeasible"


def test_without_the_mined_text_the_same_task_is_not_judged(tmp_path):
    tests = HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 3)\n" + MAIN
    assert coding_solver.solve(_coding(tmp_path, MIN_BUG, tests), time_limit=60).status != "infeasible"


# ---------------------------------------------------------------- only a one-edit slip of the known text is judged
def test_variants_with_a_meaning_of_their_own_are_not_judged(tmp_path):
    # the reviewer's probes: the shape guard lets each through, but each is more than one slip of the known text
    cases = {
        "float_total": ("def total(xs):\n    s = 0.0\n    for x in xs:\n        s -= x\n    return s\n",
                        "    def test_t(self):\n        self.assertEqual(m.total([1, 2]), 3)\n\n"
                        "    def test_f(self):\n        self.assertIsInstance(m.total([1]), float)\n"),
        "split_space": ("def reverse_words(s):\n    return ' '.join(sorted(s.split(' ')))\n",
                        "    def test_rw(self):\n        self.assertEqual(m.reverse_words('c a b'), 'b a c')\n\n"
                        "    def test_sp(self):\n        self.assertEqual(m.reverse_words('a  b'), 'b  a')\n"),
        "clamped_n": ("def first_n(xs, n):\n    return xs[:max(n, 1)]\n",
                      "    def test_neg(self):\n        self.assertEqual(m.first_n([1, 2], -1), [])\n\n"
                      "    def test_zero(self):\n        self.assertEqual(m.first_n([1, 2], 0), [])\n\n"
                      "    def test_two(self):\n        self.assertEqual(m.first_n([1, 2, 3], 2), [1, 2])\n"),
    }
    for name, (lib, tests) in cases.items():
        res = coding_solver.solve(_coding(tmp_path / name, lib, HEAD + tests + MAIN), time_limit=60)
        assert res.status != "infeasible", name


def test_a_known_function_green_on_its_own_is_not_judged(tmp_path):
    # the reviewer's probes: a lawful one-edit variant of a known function stands untouched; the bug is in the
    # unknown `double`, which shares the red test_mix with it; its own test is green on the original text
    double = "\n\ndef double(x):\n    return x + 2\n"
    t_double = "    def test_double(self):\n        self.assertEqual(m.double(4), 8)\n"
    cases = {
        "total_plus_one": ("def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n",
                           "self.assertEqual(m.total([1, 2]), 4)", "m.double(m.total([2]))", 6),
        "split_space": ("def reverse_words(s):\n    return ' '.join(reversed(s.split(' ')))\n",
                        "self.assertEqual(m.reverse_words('a  b'), 'b  a')", "len(m.reverse_words('a b')) + m.double(1)", 5),
        "floor_average": ("def average(xs):\n    return sum(xs) // len(xs)\n",
                          "self.assertEqual(m.average([1, 2]), 1)", "m.double(m.average([1, 2]))", 2),
    }
    for name, (lib, own, mix, want) in cases.items():
        tests = (HEAD + f"    def test_own(self):\n        {own}\n\n"
                 f"    def test_mix(self):\n        self.assertEqual({mix}, {want})\n\n" + t_double + MAIN)
        t = _coding(tmp_path / name, lib + double, tests)
        res = coding_solver.solve(t, time_limit=60)
        assert res.status != "infeasible", name
        assert lib in (t.workspace / "mylib.py").read_text(encoding="utf-8"), name   # the variant is untouched


def test_one_edit_detection():
    assert coding_solver._one_edit(TOTAL.replace("+=", "-="), "total")
    assert coding_solver._one_edit(MAXIMUM.replace("x > best", "x < best"), "maximum")
    assert not coding_solver._one_edit(TOTAL.replace("+=", "-=").replace("s = 0", "s = 0.0"), "total")
    assert not coding_solver._one_edit("def first_n(xs, n):\n    return xs[:max(n, 1)]\n", "first_n")
    assert not coding_solver._one_edit("def reverse_words(s):\n    return ' '.join(sorted(s.split(' ')))\n",
                                       "reverse_words")
    assert not coding_solver._one_edit(TOTAL, "nosuch")


def test_a_function_that_rebinds_the_known_one_proves_nothing(tmp_path):
    lib = "def aa(x):\n    global total\n    total = lambda xs: 7\n    return x\n\n\n" + TOTAL
    tests = (HEAD + "    def test_a(self):\n        self.assertEqual(m.aa(1), 2)\n\n"
             "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n" + MAIN)
    assert coding_solver.solve(_coding(tmp_path, lib, tests), time_limit=60).status != "infeasible"
    refs = coding_solver._test_refs(tests)
    assert not coding_solver.known_text_fails(tests, lib, ["test_total"], refs)
    for dyn in ("globals()['total'] = max", "exec('pass')", "eval('1')", "vars()"):
        lib = "def aa(x):\n    " + dyn + "\n    return x\n\n\n" + TOTAL
        assert not coding_solver.known_text_fails(tests, lib, ["test_total"], refs), dyn


# ---------------------------------------------------------------- the test file does not parse
def test_test_file_that_does_not_parse_is_infeasible(tmp_path):
    tests = HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6\n" + MAIN
    res = coding_solver.solve(_coding(tmp_path, TOTAL.replace("+=", "-="), tests), time_limit=60)
    assert res.status == "infeasible" and "does not parse" in res.note


def test_test_file_with_a_bom_or_an_import_error_is_not_infeasible(tmp_path):
    tests = HEAD + "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n" + MAIN
    # Python reads a UTF-8 BOM fine: the file parses, whatever the solver then makes of it
    res = coding_solver.solve(_coding(tmp_path / "a", TOTAL.replace("+=", "-="), None,
                                      raw=b"\xef\xbb\xbf" + tests.encode()), time_limit=60)
    assert res.status != "infeasible"
    # an import that fails is not a file that does not parse
    bad = tests.replace("import mylib as m", "import mylib as m\nimport nosuchmodule_xyz")
    res = coding_solver.solve(_coding(tmp_path / "b", TOTAL.replace("+=", "-="), bad), time_limit=60)
    assert res.status != "infeasible"
    # a broken mylib.py is not the tests' fault either
    res = coding_solver.solve(_coding(tmp_path / "c", "def total(xs:\n    return 0\n", tests), time_limit=60)
    assert res.status != "infeasible"

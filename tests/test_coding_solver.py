import ast
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
    mylib = "def total(xs):\n    return 0\n"
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 1000)\n")
    t = _custom(tmp_path, mylib, tests)
    before = (t.workspace / "mylib.py").read_bytes()
    res = solve(t, time_limit=60)
    assert res.status == "unparsed" and res.note
    assert (t.workspace / "mylib.py").read_bytes() == before


def test_unfixable_function_leaves_partial_tentative(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    tp = t.workspace / "test_mylib.py"
    tp.write_text(tp.read_text(encoding="utf-8").replace("m.total([1, 2, 3]), 6", "m.total([1, 2, 3]), 1000"),
                  encoding="utf-8")
    res = solve(t, time_limit=60)
    assert res.status == "solved" and res.tentative and "partial" in res.note


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


def test_regression_second_largest_domain_limit_is_not_a_crash(tmp_path):
    bug = "def second_largest(xs):\n    vals = sorted(set(xs))\n    return vals[-1]\n"
    _hidden_check(tmp_path, bug, "second_largest", ("[10, 20]", 10),
                  [("[4, 7, 5]", 5), ("[1, 2, 3, 4]", 3)])


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


# ---------------------------------------------------------------- candidates and tentative answers

AMBIG_LIB = "def count_evens(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"
AMBIG_TEST = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
              "    def test_count(self):\n        self.assertEqual(mylib.count_evens([1, 2, 4, 6, 7]), 3)\n")


def test_ambiguous_fix_returns_clusters(tmp_path):
    t = _custom(tmp_path, AMBIG_LIB, AMBIG_TEST)
    res = solve(t)
    assert res.status == "solved" and res.answer == "DONE" and res.tentative
    amb = res.extra["ambiguous"]
    assert len(amb) == 1 and amb[0]["func"] == "count_evens"
    cands = amb[0]["candidates"]
    assert len(cands) >= 2 and len(set(cands)) == len(cands)
    written = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    import ast
    assert ast.unparse(ast.parse(cands[amb[0]["chosen"]])) == ast.unparse(ast.parse(written))
    assert any("x % 2 == 0" in c for c in cands)
    assert any("count_evens" in a for a in res.extra["visible_asserts"])
    assert run_tests(t.workspace).ok


def test_unambiguous_fix_has_no_extra(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "0")   # the check would leave it tentative
    mylib = "def f(n):\n    i = 0\n    while i <= n:\n        i += 1\n    return i\n"
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(5), 5)\n")
    res = solve(_custom(tmp_path, mylib, tests))
    assert res.status == "solved" and not res.tentative and res.extra is None


def test_partial_repair_is_tentative_and_written(tmp_path):
    lib = ("def a(x):\n    return x + 2\n\n\ndef b(x):\n    return x * 3\n")
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_a(self):\n        self.assertEqual(mylib.a(5), 6)\n"
             "    def test_b(self):\n        self.assertEqual(mylib.b(5), 100)\n")
    t = _custom(tmp_path, lib, tests)
    res = solve(t)
    assert res.status == "solved" and res.answer == "DONE" and res.tentative
    assert "partial" in res.note
    text = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "x + 1" in text and "x * 3" in text  # a fixed, b untouched
    assert (t.workspace / "test_mylib.py").read_text(encoding="utf-8") == tests


def test_partial_repair_leaves_tests_byte_identical(tmp_path):
    t = _fixture_task("coding_0003", tmp_path)
    tp = t.workspace / "test_mylib.py"
    tp.write_text(tp.read_text(encoding="utf-8").replace("m.total([1, 2, 3]), 6", "m.total([1, 2, 3]), 1000"),
                  encoding="utf-8")
    before = tp.read_bytes()
    res = solve(t, time_limit=60)
    assert res.tentative and tp.read_bytes() == before


# ---------------------------------------------------------------- tie preference among green fixes

from harness.solvers.coding_solver import _edit_rank  # noqa: E402


def _r(buggy_body, cand_body):
    return _edit_rank(f"def f(x, y):\n    {buggy_body}\n", f"def f(x, y):\n    {cand_body}\n")


def test_edit_rank_comparison_direction_then_strictness_then_negation():
    flip, strict, neg = _r("return x < y", "return x > y"), _r("return x > y", "return x >= y"), \
        _r("return x > y", "return x <= y")
    assert flip < strict < neg
    assert _r("return x == y", "return x != y") > flip


def test_edit_rank_plus_minus_beats_other_operators():
    assert _r("return x + y", "return x - y") < _r("return x + y", "return x % y")
    assert _r("return x / y", "return x // y") < _r("return x / y", "return x % y")
    assert _r("x += y\n    return x", "x -= y\n    return x") < _r("x += y\n    return x", "x *= y\n    return x")


def test_edit_rank_constant_towards_smaller_magnitude():
    assert _r("return x - 2", "return x - 1") < _r("return x - 2", "return x - 3")
    assert _r("return sum(2 for v in x)", "return sum(1 for v in x)") < \
        _r("return sum(1 for v in x)", "return sum(2 for v in x)")


def test_edit_rank_constant_beats_flip_only_through_the_mutant_order():
    # equal rank: the solver falls back to the mutant order (constants first)
    assert _r("return sum(2 for v in x if v > y)", "return sum(1 for v in x if v > y)") == \
        _r("return sum(2 for v in x if v > y)", "return sum(2 for v in x if v < y)")


def test_edit_rank_degenerate_code_removed_or_created():
    assert _r("return x % 1 == y", "return x % 2 == y") < _r("return x % 1 == y", "return x % 1 != y")
    assert _r("return x[::-0]", "return x[::-1]") < _r("return x[::-0]", "return x[::--1]")
    assert _r("return x[-1]", "return x[-2]") < _r("return x[-1]", "return x[-0]")
    assert _r("return x % y == -1", "return x % y == -0") < _r("return x % y == -1", "return x % y != -1")


TIE_TESTS = "import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n    def test_v(self):\n"


def _chosen_src(res):
    it = res.extra["ambiguous"][0]
    return it["candidates"][it["chosen"]], it


def test_tie_constant_fix_wins_over_flip_by_mutant_order(tmp_path):
    # `_edit_rank` ties the two fixes (see test_edit_rank_constant_beats_flip_only_through_the_mutant_order):
    # the constant wins only because constants come first in the mutant order
    lib = "def count_positive(xs):\n    return sum(2 for x in xs if x > 0)\n"
    t = _custom(tmp_path, lib, TIE_TESTS + "        self.assertEqual(mylib.count_positive([0, 0, 4, -4, 9]), 2)\n")
    res = solve(t)
    src, it = _chosen_src(res)
    assert "sum(1 for x in xs if x > 0)" in src.replace("((", "(").replace("))", ")")
    assert it["chosen"] == 0   # the preferred fix is option 1 for the arbitration
    text = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "1 for x in xs if x > 0" in text and run_tests(t.workspace).ok


def test_tie_prefers_keeping_comparison_direction(tmp_path):
    lib = "def filter_values(d, t):\n    return {k: v for k, v in d.items() if v > t}\n"
    t = _custom(tmp_path, lib, TIE_TESTS + "        self.assertEqual(mylib.filter_values({'a': 2, 'b': 2}, 2), {'a': 2, 'b': 2})\n")
    res = solve(t)
    src, it = _chosen_src(res)
    assert "v >= t" in src and it["chosen"] == 0
    assert "v >= t" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_tie_keeps_strict_comparison_first_occurrence(tmp_path):
    lib = ("def longest_word(s):\n    best = ''\n    for w in s.split():\n"
           "        if len(w) < len(best):\n            best = w\n    return best\n")
    t = _custom(tmp_path, lib, TIE_TESTS + "        self.assertEqual(mylib.longest_word('a bbb cc'), 'bbb')\n")
    res = solve(t)
    src, it = _chosen_src(res)
    assert "len(w) > len(best)" in src and it["chosen"] == 0
    assert "len(w) > len(best)" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- strict contradiction proof
_H = ("import unittest\nimport mylib as m\nfrom unittest import mock\n\n"
      "class T(unittest.TestCase):\n    def test_a(self):\n")


def _t(body: str, header: str = _H.replace("from unittest import mock" + chr(10), "")) -> str:
    return header + "".join("        " + ln + "\n" for ln in body.split("\n"))


_TWO = "self.assertEqual(m.f(1), 1)\nself.assertEqual(m.f(1), 2)"
_PURE = "def f(x):\n    return x\n"


def _pc(tests, lib):
    from harness.solvers.coding_solver import provable_contradiction
    return provable_contradiction(tests, lib)


def test_proof_true_contradiction():
    assert _pc(_t("self.assertEqual(m.f(7), 7)\nself.assertEqual(m.f(7), 8)"), _PURE)
    assert _pc(_t("self.assertEqual(m.f([1]), 2)\nself.assertEqual(m.f([1]), 1)"),
               "def f(l):\n    l.append(1)\n    return len(l)\n")
    assert _pc(_t("self.assertEqual(m.f({'a':1}), 1)\nself.assertEqual(m.f({'a':1}), 2)"),
               "def f(d):\n    return sorted(d)[0]\n")


def test_proof_python_equality_means_no_proof():
    for a, b in (("1", "1.0"), ("True", "1"), ("'a'", '"a"'), ("0", "False"), ("[1]", "[1.0]")):
        assert not _pc(_t(f"self.assertEqual(m.f(7), {a})\nself.assertEqual(m.f(7), {b})"), _PURE)


def test_proof_other_assertions_and_contexts():
    assert not _pc(_t("self.assertAlmostEqual(m.f(7), 1.0)\nself.assertAlmostEqual(m.f(7), 2.0)"), _PURE)
    assert not _pc(_t("self.assertEqual(m.f(7), 7)\nself.assertNotEqual(m.f(7), 8)"), _PURE)
    assert not _pc(_t("self.assertEqual(m.f(7), 7)\nself.assertIn(m.f(7), [8])"), _PURE)
    assert not _pc(_t("self.assertCountEqual(m.f([1]), [1])\nself.assertCountEqual(m.f([1]), [2])"), _PURE)
    assert not _pc(_t("if True:\n    self.assertEqual(m.f(7), 7)\nelse:\n    self.assertEqual(m.f(7), 8)"), _PURE)
    assert not _pc(_t("try:\n    self.assertEqual(m.f(7), 8)\nexcept AssertionError:\n"
                      "    self.assertEqual(m.f(7), 7)"), _PURE)
    assert not _pc(_t("for e in (1, 2):\n    with self.subTest(e=e):\n        self.assertEqual(m.f(1), e)"), _PURE)
    # two classes with the same method name
    two_cls = ("import unittest\nimport mylib as m\nclass A(unittest.TestCase):\n    def test_a(self):\n"
               "        self.assertEqual(m.f(1), 1)\nclass B(unittest.TestCase):\n    def test_a(self):\n"
               "        self.assertEqual(m.f(1), 2)\n")
    assert not _pc(two_cls, _PURE)


def test_proof_mocking_and_monkeypatching():
    lib = "def g(x):\n    return x\ndef f(x):\n    return g(x)\n"
    assert not _pc(_t("self.assertEqual(m.f(7), 7)\nwith mock.patch('mylib.g', return_value=9):\n"
                      "    self.assertEqual(m.f(7), 9)"), lib)
    assert not _pc(_t("self.assertEqual(m.f(7), 7)\nm.g = lambda x: 9\nself.assertEqual(m.f(7), 9)"), lib)
    two = "self.assertEqual(m.f(7), 7)" + chr(10) + "self.assertEqual(m.f(7), 8)"
    assert not _pc(_t(two, _H), lib)  # imports mock
    setup = _H.replace("    def test_a", "    def setUp(self):\n        pass\n    def test_a")
    assert not _pc(setup + "        self.assertEqual(m.f(7), 7)\n        self.assertEqual(m.f(7), 8)\n", lib)


def test_proof_stateful_callee_means_no_proof():
    for lib in (
        "def f(x, acc=[]):\n    acc.append(x)\n    return len(acc)\n",
        "class C:\n    n = 0\ndef f(x):\n    C.n += 1\n    return C.n\n",
        "from random import randint\ndef f(x):\n    return randint(1, 2)\n",
        "from helper import c\ndef f(x):\n    return c()\n",
        "from pathlib import Path\ndef f(x):\n    p = Path('c.txt')\n    p.write_text('1')\n    return 1\n",
        "n = 0\ndef f(x):\n    global n\n    n += 1\n    return n\n",
        "c = []\ndef f(x):\n    c.append(x)\n    return len(c)\n",
        "def f(x):\n    return open('c.txt').read()\n",
        "class f:\n    def __init__(self, x): pass\n",
        "def f(x):\n    return g(x)\ndef g(x):\n    return id(x)\n",
        "def f(x):\n    return x\nf = print\n",
        "import functools\n@functools.lru_cache\ndef f(x):\n    return x\n",
        "class A:\n    def __eq__(s, o): return True\ndef f(x):\n    return A()\n",
    ):
        assert not _pc(_t(_TWO), lib), lib


def test_stateful_no_arg_and_missing_function():
    assert not _pc(_t(_TWO), "def h(x):\n    return x\n")
    assert not _pc(_t("self.assertEqual(m.f(), 1)\nself.assertEqual(m.f(), 2)"), "def f():\n    return 1\n")


def test_proof_decorators_returns_and_reflection():
    two = "self.assertEqual(m.f(7), 7)" + chr(10) + "self.assertEqual(m.f(7), 8)"
    assert _pc(_t(two), _PURE)
    for deco in ("@unittest.skip('x')", "@unittest.expectedFailure", "@some.thing"):
        t = _t(two).replace("    def test_a", "    " + deco + chr(10) + "    def test_a")
        assert not _pc(t, _PURE), deco
    t = _t(two).replace("class T(", "@unittest.skip('x')" + chr(10) + "class T(")
    assert not _pc(t, _PURE)
    assert not _pc(_t("return" + chr(10) + two), _PURE)
    assert not _pc(_t("raise RuntimeError()" + chr(10) + two), _PURE)
    assert not _pc(_t("self.skipTest('x')" + chr(10) + two), _PURE)
    assert not _pc(_t("if self.id():" + chr(10) + "    return" + chr(10) + two), _PURE)
    for hook in ("setattr(m, 'f', len)", "delattr(m, 'f')", "importlib.reload(m)", "m.__dict__.update({})"):
        assert not _pc(_t(hook + chr(10) + two), _PURE), hook


def test_phase1_branching_contradiction_is_not_infeasible(tmp_path):
    # asserts under a branch are not a proof: only provable_contradiction may declare infeasible
    lib = "def f(x):\n    return x\n"
    tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        if m.f(0) == 0:\n"
             "            self.assertEqual(m.f(7), 7)\n            self.assertEqual(m.f(7), 8)\n")
    r = solve(_custom(tmp_path, lib, tests))
    assert r.status != "infeasible"


def test_proof_redefinition_in_nested_module_block_means_no_proof():
    lib = "import sys\nif sys.platform:\n    def f(x):\n        return x\nelse:\n    def f(x):\n        return -x\n"
    assert not _pc(_t("self.assertEqual(m.f(7), 7)\nself.assertEqual(m.f(7), 8)"), lib)


# ---------------------------------------------------------------- round 7: file-wide proof
_FH = "import unittest\nimport mylib as m\n\n"


def _two_methods(deco_a="", deco_b="", cls_deco="", extra_b="", name_b="test_b"):
    return (_FH + cls_deco + "class T(unittest.TestCase):\n" + deco_a +
            "    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n" + deco_b +
            f"    def {name_b}(self):\n{extra_b}        self.assertEqual(m.f(1), 2)\n")


def test_proof_across_methods_and_classes():
    assert _pc(_two_methods(), _PURE)
    two_cls = (_FH + "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n\n"
               "class U(unittest.TestCase):\n    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n")
    assert _pc(two_cls, _PURE)


def test_proof_across_methods_keeps_every_guard():
    assert not _pc(_two_methods(deco_b="    @unittest.skip('x')\n"), _PURE)
    assert not _pc(_two_methods(deco_a="    @unittest.skipIf(True, 'x')\n"), _PURE)
    assert not _pc(_two_methods(cls_deco="@unittest.skip('x')\n"), _PURE)
    assert not _pc(_two_methods(extra_b="        return\n"), _PURE)
    assert not _pc(_two_methods(extra_b="        self.skipTest('x')\n"), _PURE)
    assert not _pc(_two_methods(name_b="helper"), _PURE)       # never run by unittest
    assert not _pc(_two_methods(name_b="test_a"), _PURE)       # redefinition: only the last runs
    assert not _pc(_two_methods(), "def f(x):\n    return x + len(m)\n")
    assert not _pc(_two_methods().replace("class T(", "class T(object, ").replace("(object, unittest.TestCase)", "(object)"), _PURE)
    assert not _pc(_two_methods().replace("    def test_b", "    def setUp(self):\n        pass\n    def test_b"), _PURE)
    stateful = "n = 0\ndef f(x):\n    global n\n    n += 1\n    return n\n"
    assert not _pc(_two_methods(), stateful)
    assert not _pc(_two_methods().replace("m.f(1), 2", "m.f(2), 2"), _PURE)   # different args


def test_proof_conditional_assert_never_counts():
    t = (_FH + "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n"
         "    def test_b(self):\n        if m.f(0) == 0:\n            self.assertEqual(m.f(1), 2)\n")
    assert not _pc(t, _PURE)


# ---------------------------------------------------------------- round 7: fix ranking
def _hidden_ok(ws, lib_asserts):
    import subprocess, sys
    code = "from mylib import *\n" + "\n".join("assert " + a for a in lib_asserts)
    try:
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=ws, capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return p.returncode == 0


def _solve_case(tmp_path, buggy, vis_call, vis_exp):
    tests = (_FH + "class T(unittest.TestCase):\n    def test_x(self):\n"
             f"        self.assertEqual(m.{vis_call}, {vis_exp})\n")
    t = _custom(tmp_path, buggy, tests)
    return t, solve(t)


def test_rank_is_sorted_not_plus_zero(tmp_path):
    buggy = "def is_sorted(xs):\n    return all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1))\n"
    t, r = _solve_case(tmp_path, buggy, "is_sorted([1, 2, 3])", "True")
    assert _hidden_ok(t.workspace, ["is_sorted([1,1,2])==True", "is_sorted([3,1])==False"])


def test_rank_gcd_prefers_non_hanging_fix(tmp_path):
    buggy = "def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n"
    t, r = _solve_case(tmp_path, buggy, "gcd(12, 8)", "4")
    assert _hidden_ok(t.workspace, ["gcd(7,3)==1", "gcd(100,75)==25"])


def test_rank_even_sq_pow_mutation(tmp_path):
    buggy = "def even_sq(xs):\n    return [x * 2 for x in xs if x % 2 == 0]\n"
    t, r = _solve_case(tmp_path, buggy, "even_sq([1, 3, 4])", "[16]")
    assert _hidden_ok(t.workspace, ["even_sq([3,6])==[36]"])


def test_pair_only_fix_is_solved_not_tentative(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "0")   # the check would leave it tentative
    buggy = "def f(a, b):\n    return a - b * 3\n"
    t, r = _solve_case(tmp_path, buggy, "f(1, 2)", "5")
    assert r.status == "solved" and not r.tentative
    assert r.note == "repaired by mutation"


def test_pow_mutants():
    from harness.solvers.mutations import mutants
    assert "def f(x):\n    return x ** 2" in list(mutants("def f(x):\n    return x * 2"))
    assert "def f(x):\n    return x * 2" in list(mutants("def f(x):\n    return x ** 2"))


def test_degenerate_zero_offset_only_when_tautology():
    from harness.solvers.coding_solver import _degenerate
    assert _degenerate(ast.parse("xs[i] >= xs[i + 0]")) > 0
    for e in ("x + 0", "x - 0", "0 + x", "xs[:n - 0]", "x + 1"):
        assert _degenerate(ast.parse(e)) == 0, e


def test_pow_mutants_only_for_small_constant_multipliers():
    from harness.solvers.mutations import mutants
    assert not any("**" in m for m in mutants("def f(x, y):\n    return x * y"))
    assert not any("**" in m for m in mutants("def f(x):\n    return x * 4"))
    assert not any("**" in m for m in mutants("def f(x):\n    return x + 2"))
    assert any("x ** 3" in m for m in mutants("def f(x):\n    return x * 3"))


def test_rank_power_right_fix_is_offered(tmp_path):
    buggy = "def power(b, e):\n    r = 1\n    for _ in range(e):\n        r += b\n    return r\n"
    t, r = _solve_case(tmp_path, buggy, "power(2, 3)", "8")
    assert r.tentative and any("r *= b" in c for c in r.extra["ambiguous"][0]["candidates"])

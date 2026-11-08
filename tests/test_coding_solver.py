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

AMBIG_LIB = "def count_even(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"
AMBIG_TEST = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
              "    def test_count(self):\n        self.assertEqual(mylib.count_even([1, 2, 4, 6, 7]), 3)\n")


def test_ambiguous_fix_returns_clusters(tmp_path):
    t = _custom(tmp_path, AMBIG_LIB, AMBIG_TEST)
    res = solve(t)
    assert res.status == "solved" and res.answer == "DONE" and res.tentative
    amb = res.extra["ambiguous"]
    assert len(amb) == 1 and amb[0]["func"] == "count_even"
    cands = amb[0]["candidates"]
    assert len(cands) >= 2 and len(set(cands)) == len(cands)
    written = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    import ast
    assert ast.unparse(ast.parse(cands[amb[0]["chosen"]])) == ast.unparse(ast.parse(written))
    assert any("x % 2 == 0" in c for c in cands)
    assert any("count_even" in a for a in res.extra["visible_asserts"])
    assert run_tests(t.workspace).ok


def test_unambiguous_fix_has_no_extra(tmp_path):
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


def test_contradiction_ends_before_any_search(tmp_path, monkeypatch):
    from harness.solvers import coding_solver as cs

    def boom(*a, **k):
        raise AssertionError("repair search must not start")
    monkeypatch.setattr(cs, "_repair_round", boom)
    lib = "def f(x):\n    return x\n\ndef g(x):\n    return x + 1\n"
    tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
             "    def test_f(self):\n        self.assertEqual(m.f(7), 7)\n        self.assertEqual(m.f(7), 8)\n"
             "    def test_g(self):\n        self.assertEqual(m.g(1), 5)\n")
    r = solve(_custom(tmp_path, lib, tests))
    assert r.status == "infeasible" and r.answer is None


def test_stateful_callee_is_not_a_proof():
    from harness.solvers.coding_solver import provable_contradiction
    tests = ("import mylib as m\nclass T:\n    def test_f(self):\n"
             "        self.assertEqual(m.f(1), 1)\n        self.assertEqual(m.f(1), 2)\n")
    assert provable_contradiction(tests, "def f(x):\n    return x\n")
    assert not provable_contradiction(tests, "n = 0\ndef f(x):\n    global n\n    n += 1\n    return n\n")
    assert not provable_contradiction(tests, "c = []\ndef f(x):\n    c.append(x)\n    return len(c)\n")
    assert not provable_contradiction(tests, "import random\ndef f(x):\n    return random.random()\n")
    assert not provable_contradiction(tests, "class f:\n    def __init__(self, x): pass\n")
    assert not provable_contradiction(tests, "def f(x):\n    return g(x)\ndef g(x):\n    return id(x)\n")

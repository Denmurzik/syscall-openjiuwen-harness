"""Round 14, task 69 (expert 7, idea c): free proofs of infeasibility, and the doubts that keep them silent."""
import asyncio
import csv

from harness import llm_routes as L
from harness.solvers import coding_solver as cs
from harness.solvers import data_solver
from harness.tasks import Task

INS = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
       "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the "
       "word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
LIB = ("def is_palindrome(s):\n    return s == s[::-1]\n\n\n"
       "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n")
HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TAIL = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
BASE = ("    def test_is_palindrome(self):\n        self.assertTrue(m.is_palindrome('racecar'))\n"
        "        self.assertFalse(m.is_palindrome('abca'))\n\n"
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n")


def _file(extra: str, head: str = HEAD) -> str:
    return head + BASE + ("\n" + extra if extra else "") + TAIL


def _task(tmp_path, tests: str, lib: str = LIB, ins: str = INS) -> Task:
    ws = tmp_path / "t" / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("t", "coding", ins, 2.0, "unittest", tmp_path / "t", ws)


# ---- assertTrue and assertFalse of one call ----------------------------------------------------------------------

def test_true_and_false_of_one_call_is_a_contradiction():
    t = _file("    def test_spec(self):\n        self.assertTrue(m.is_palindrome('abca'))\n")
    assert cs.provable_contradiction(t, LIB, ["test_spec", "test_total"])


def test_true_and_false_of_different_calls_is_not():
    t = _file("    def test_spec(self):\n        self.assertTrue(m.is_palindrome('abcba'))\n")
    assert not cs.provable_contradiction(t, LIB, ["test_total"])


def test_true_and_false_of_different_functions_is_not():
    lib = LIB + "\n\ndef is_odd(s):\n    return len(s) % 2 == 1\n"
    t = _file("    def test_spec(self):\n        self.assertTrue(m.is_odd('abca'))\n")
    assert not cs.provable_contradiction(t, lib, ["test_spec", "test_total"])


def test_true_and_false_of_an_impure_call_is_not():
    lib = ("SEEN = []\n\n\ndef is_palindrome(s):\n    SEEN.append(s)\n    return len(SEEN) % 2 == 1\n\n\n"
           "def total(xs):\n    return -sum(xs)\n")
    t = _file("    def test_spec(self):\n        self.assertTrue(m.is_palindrome('abca'))\n")
    assert not cs.provable_contradiction(t, lib, ["test_spec", "test_total"])


def test_truthiness_pins_no_value():
    # assertTrue(c) and assertEqual(c, 'yes') agree (c = 'yes'); the assertFalse sits behind a return and never
    # runs. A rewrite of assertTrue into assertEqual(c, True) would have "proved" this feasible file infeasible
    t = (HEAD + "    def test_a(self):\n        self.assertTrue(m.is_palindrome('ab'))\n"
         "        self.assertEqual(m.is_palindrome('ab'), 'yes')\n\n"
         "    def test_b(self):\n        return\n        self.assertFalse(m.is_palindrome('ab'))\n" + TAIL)
    assert not cs.provable_contradiction(t, LIB, ["test_a"])


def test_true_and_false_with_a_conditional_side_is_not():
    t = _file("    def test_spec(self):\n        if m.total([1]) == 1:\n"
              "            self.assertTrue(m.is_palindrome('abca'))\n")
    assert not cs.provable_contradiction(t, LIB, ["test_spec", "test_total"])


def test_true_and_false_with_a_non_literal_argument_is_not():
    t = _file("    def test_spec(self):\n        self.assertTrue(m.is_palindrome(str(m.total([]))))\n"
              "        self.assertFalse(m.is_palindrome(str(m.total([]))))\n")
    assert not cs.provable_contradiction(t, LIB, ["test_spec", "test_total"])


# ---- self.fail ---------------------------------------------------------------------------------------------------

def test_self_fail_is_unpassable_only_when_that_test_is_red():
    t = _file("    def test_impossible(self):\n        self.fail('cannot pass')\n")
    assert cs.provable_contradiction(t, LIB, ["test_impossible", "test_total"])
    assert not cs.provable_contradiction(t, LIB, ["test_total"])   # it did not run red: no proof


def test_self_fail_behind_a_condition_is_not_a_proof():
    t = _file("    def test_maybe(self):\n        if m.total([1]) != 1:\n            self.fail('wrong')\n")
    assert not cs.provable_contradiction(t, LIB, ["test_maybe", "test_total"])


def test_self_fail_in_try_else_or_with_is_not_a_proof():
    for body in ("        try:\n            m.total(None)\n        except TypeError:\n            pass\n"
                 "        else:\n            self.fail('no error')\n",
                 "        try:\n            m.total(None)\n            self.fail('no error')\n"
                 "        except TypeError:\n            pass\n",
                 "        with self.assertRaises(TypeError):\n            m.total(None)\n            self.fail('x')\n"):
        t = _file("    def test_raises(self):\n" + body)
        assert not cs.provable_contradiction(t, LIB, ["test_raises", "test_total"]), body


def test_self_fail_after_a_return_is_not_a_proof():
    t = _file("    def test_maybe(self):\n        self.assertEqual(m.total([1]), 1)\n        return\n        self.fail('x')\n")
    assert not cs.provable_contradiction(t, LIB, ["test_maybe", "test_total"])


def test_expected_failure_or_skip_is_not_a_proof():
    for extra in ("    @unittest.expectedFailure\n    def test_impossible(self):\n        self.fail('cannot pass')\n",
                  "    @unittest.skip('later')\n    def test_impossible(self):\n        self.fail('cannot pass')\n",
                  "    def test_impossible(self):\n        if m.total([]) == 0:\n            self.skipTest('x')\n"
                  "        self.fail('cannot pass')\n"):
        assert not cs.provable_contradiction(_file(extra), LIB, ["test_impossible", "test_total"]), extra


def test_self_fail_in_a_helper_is_not_a_proof():
    t = _file("    def check(self):\n        self.fail('x')\n")
    assert not cs.provable_contradiction(t, LIB, ["test_total", "check"])


def test_self_fail_with_a_computed_message_is_not_a_proof():
    t = _file("    def test_impossible(self):\n        self.fail(m.total([1]))\n")
    assert not cs.provable_contradiction(t, LIB, ["test_impossible", "test_total"])


def test_phase1_gives_infeasible_for_self_fail(tmp_path):
    task = _task(tmp_path, _file("    def test_impossible(self):\n        self.fail('cannot pass')\n"))
    res = cs.solve(task)
    assert res.status == "infeasible" and "contradict" in res.note


def test_phase1_still_repairs_without_the_impossible_test(tmp_path):
    res = cs.solve(_task(tmp_path, _file("")))
    assert res.status == "solved"


# ---- a module the tests import before mylib ----------------------------------------------------------------------

def _head(module: str) -> str:
    return HEAD.replace("import mylib as m", f"import {module}\nimport mylib as m")


def test_missing_module_in_the_tests(tmp_path):
    res = cs.solve(_task(tmp_path, _file("", _head("quantum_flux_sdk"))))
    assert res.status == "infeasible" and "quantum_flux_sdk" in res.note


def test_module_that_stands_in_the_workspace_is_not_missing(tmp_path):
    task = _task(tmp_path, _file("", _head("helper")))
    (task.workspace / "helper.py").write_text("X = 1\n", encoding="utf-8")
    assert cs.solve(task).status == "solved"


def test_missing_import_needs_the_error_in_the_output(tmp_path):
    src = _file("", _head("json"))
    assert cs.missing_test_import(src, tmp_path, "FAIL: test_total", INS) is None
    assert cs.missing_test_import(src, tmp_path, "ModuleNotFoundError: No module named 'other'", INS) is None


def test_missing_import_after_mylib_is_not_a_proof(tmp_path):
    src = _file("", HEAD.replace("import mylib as m", "import mylib as m\nimport quantum_flux_sdk"))
    out = "ModuleNotFoundError: No module named 'quantum_flux_sdk'"
    assert cs.missing_test_import(src, tmp_path, out, INS) is None


def test_stdlib_and_popular_packages_are_never_missing(tmp_path):
    for name in ("tkinter", "numpy", "pytest", "hypothesis"):
        out = f"ModuleNotFoundError: No module named '{name}'"
        assert cs.missing_test_import(_file("", _head(name)), tmp_path, out, INS) is None, name


def test_a_module_the_task_names_is_not_missing(tmp_path):
    src, out = _file("", _head("helpers")), "ModuleNotFoundError: No module named 'helpers'"
    assert cs.missing_test_import(src, tmp_path, out, INS) == "helpers"
    assert cs.missing_test_import(src, tmp_path, out, INS + " The helpers module must be written too.") is None
    assert cs.missing_test_import(src, tmp_path, out, "Make the tests in test_mylib.py pass.") is None
    (tmp_path / "README.md").write_text("Create helpers.py with the shared constants.\n", encoding="utf-8")
    assert cs.missing_test_import(src, tmp_path, out, INS) is None


def test_a_statement_before_the_imports_keeps_it_silent(tmp_path):
    src = _file("", HEAD.replace("import mylib as m",
                                 "try:\n    import quantum_flux_sdk\nexcept ImportError:\n    pass\nimport mylib as m"))
    out = "ModuleNotFoundError: No module named 'quantum_flux_sdk'"
    assert cs.missing_test_import(src, tmp_path, out, INS) is None


def test_the_ladder_spends_nothing_on_a_missing_module(tmp_path):
    task = _task(tmp_path, _file("", _head("quantum_flux_sdk")))

    class NoCalls:
        async def chat(self, *a, **k):
            raise AssertionError("no model call expected")

    res = asyncio.run(L.fix_coding(task, NoCalls()))
    assert res.status == "infeasible" and "quantum_flux_sdk" in res.note


# ---- data: a quantity no column provides -------------------------------------------------------------------------

def _data_task(tmp_path, question: str, header=("order_id", "region", "product", "month", "units", "price"),
               rows=((1, "west", "widget", 5, 5, 33), (2, "north", "gadget", 6, 28, 28))) -> Task:
    ws = tmp_path / "d" / "workspace"
    ws.mkdir(parents=True)
    with open(ws / "sales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    ins = (f"The file sales.csv has columns {','.join(header)} ({len(rows)} rows). {question} "
           "Use bash with python3 to compute it. Work only inside the current directory. When you are done, write "
           "ONLY the final answer to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
    return Task("d", "data_analysis", ins, 2.0, "exact_int", tmp_path / "d", ws)


def test_absent_column_with_an_untaken_filter(tmp_path):
    for i, q in enumerate(("What is the total discount given on product 'gadget'?",
                           "What is the average discount of region 'west'?",
                           "What is the total discount given?",
                           "What was the maximum tax?")):
        res = data_solver.solve(_data_task(tmp_path / str(i), q))
        assert res.status == "infeasible", (q, res.note)


def test_a_real_column_is_not_absent(tmp_path):
    res = data_solver.solve(_data_task(tmp_path, "What is the average price of region 'west'?"))
    assert res.status != "infeasible"


def test_a_variant_of_the_column_is_not_absent(tmp_path):
    for i, header in enumerate((("order_id", "region", "product", "month", "units", "price", "discount_pct"),
                                ("order_id", "region", "product", "month", "units", "price", "Discounts"))):
        rows = ((1, "west", "widget", 5, 5, 33, 1), (2, "north", "gadget", 6, 28, 28, 2))
        res = data_solver.solve(_data_task(tmp_path / str(i), "What is the average discount of region 'west'?",
                                           header, rows))
        assert res.status != "infeasible", header


def test_a_derivable_quantity_is_not_absent(tmp_path):
    header = ("order_id", "region", "product", "month", "units", "price", "cost")
    rows = ((1, "west", "widget", 5, 5, 33, 20), (2, "north", "gadget", 6, 28, 28, 10))
    for i, q in enumerate(("What is the total profit?", "What is the average margin of region 'west'?")):
        res = data_solver.solve(_data_task(tmp_path / str(i), q, header, rows))
        assert res.status != "infeasible", q
    header = ("order_id", "region", "product", "month", "units", "price", "revenue")
    rows = ((1, "west", "widget", 5, 5, 33, 150), (2, "north", "gadget", 6, 28, 28, 700))
    res = data_solver.solve(_data_task(tmp_path / "r", "What is the total discount given?", header, rows))
    assert res.status != "infeasible"


def test_another_file_with_the_quantity_is_not_absent(tmp_path):
    task = _data_task(tmp_path, "What is the total discount given on product 'gadget'?")
    (task.workspace / "discounts.json").write_text('{"gadget": 5}', encoding="utf-8")
    assert data_solver.solve(task).status != "infeasible"
    (task.workspace / "discounts.json").unlink()
    (task.workspace / "notes.txt").write_text("Each order got a Discount of 2 dollars.", encoding="utf-8")
    assert data_solver.solve(task).status != "infeasible"

"""Round 16, task 75 (expert pass 8, patch a): coding tasks with plain test functions, another module name,
`import *`."""
import asyncio

from harness import llm_routes as L
from harness.llm import ChatResult
from harness.solvers import probe
from harness.solvers.testrun import run_tests
from harness.tasks import Task

INS = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
       "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the "
       "word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost=6)


def _ws(tmp_path, files, name="t"):
    ws = tmp_path / name / "workspace"
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return ws


# ------------------------------------------------------------------ `from mylib import *`
def test_visible_call_sees_a_star_import():
    tests = "import unittest\nfrom mylib import *\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(top(3, [1]), 1)\n"
    assert probe.visible_call(tests, "top") == ((3, [1]), "self.assertEqual(top(3, [1]), 1)")
    assert probe.visible_call(tests, "_private") is None
    named = tests.replace("import *", "import other")
    assert probe.visible_call(named, "top") is None


# ------------------------------------------------------------------ test files without a TestCase
PLAIN = ("from mylib import double\n\n\ndef test_two():\n    assert double(2) == 4\n\n\n"
         "def test_zero():\n    assert double(0) == 0\n")


def test_plain_test_functions_are_run(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": "def double(x):\n    return x + 2\n", "test_mylib.py": PLAIN})
    rep = run_tests(ws)
    assert not rep.ok and rep.failed == ["test_zero"]
    (ws / "mylib.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    assert run_tests(ws).ok


def test_a_function_that_takes_a_fixture_is_not_called_blindly(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": "def double(x):\n    return x * 2\n",
                        "test_mylib.py": "from mylib import double\n\n\ndef test_two(tmp_path):\n    assert double(2) == 4\n"})
    assert not run_tests(ws).ok   # no way to run it here: never called green


def test_a_script_of_bare_asserts_is_its_own_test(tmp_path):
    script = "from mylib import double\n\nif __name__ == '__main__':\n    assert double(2) == 4, 'double'\n    print('ok')\n"
    ws = _ws(tmp_path, {"mylib.py": "def double(x):\n    return x + 1\n", "test_mylib.py": script})
    rep = run_tests(ws)
    assert not rep.ok and rep.failed == ["<script>"] and "AssertionError" in rep.output
    (ws / "mylib.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    assert run_tests(ws).ok


def test_unittest_files_run_as_before(tmp_path):
    tests = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_a(self):\n        self.assertEqual(m.double(2), 4)\n")
    ws = _ws(tmp_path, {"mylib.py": "def double(x):\n    return x + 1\n", "test_mylib.py": tests})
    rep = run_tests(ws)
    assert not rep.ok and rep.failed == ["test_a"]


# ------------------------------------------------------------------ a module under another name
def test_target_file_is_mylib_whenever_it_exists(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": "x = 1\n", "inventory.py": "y = 2\n"})
    assert L._target_file(ws, "from inventory import y\n") == "mylib.py"


def test_target_file_follows_the_import_of_the_test_file(tmp_path):
    ws = _ws(tmp_path, {"inventory.py": "y = 2\n", "helpers.py": "z = 3\n", "test_inventory.py": ""})
    assert L._target_file(ws, "import unittest\nfrom inventory import y\n") == "inventory.py"
    assert L._target_file(ws, "import unittest\nimport helpers as h\n") == "helpers.py"
    assert L._target_file(ws, "import unittest\n") == "mylib.py"   # two candidates, none imported


def test_patch_prompt_with_mylib_is_the_old_text():
    m = L._patch_messages("LIB", "TESTS", "OUT")
    assert m[0]["content"] == L.PATCH_SYSTEM
    assert m[1]["content"].startswith("mylib.py:\nLIB\n\ntest_mylib.py:\nTESTS\n\nOutput of the failing tests:\nOUT")
    other = L._patch_messages("LIB", "TESTS", "OUT", "inventory.py", "test_inventory.py")
    assert "inventory.py" in other[0]["content"] and "mylib.py" not in other[0]["content"]
    assert other[1]["content"].startswith("inventory.py:\nLIB\n\ntest_inventory.py:\n")


def test_the_ladder_repairs_a_module_of_another_name(tmp_path):
    tests = ("import unittest\nfrom inventory import low\n\n\nclass T(unittest.TestCase):\n"
             "    def test_low(self):\n        self.assertEqual(low({'a': 5, 'b': 9}, 5), ['a'])\n")
    ws = _ws(tmp_path, {"inventory.py": "def low(stock, t):\n    return sorted(k for k, v in stock.items() if v < t)\n",
                        "test_inventory.py": tests})
    task = Task("t", "coding", INS.replace("mylib", "inventory"), 2.0, "unittest", tmp_path / "t", ws)
    client = Fake(["FUNC low\ndef low(stock, t):\n    return sorted(k for k, v in stock.items() if v <= t)\n"])
    res = asyncio.run(L.fix_coding(task, client))
    assert res.status == "solved" and "<= t" in (ws / "inventory.py").read_text(encoding="utf-8")
    assert client.calls[0][1]["content"].startswith("inventory.py:\n")


# ------------------------------------------------------------------ fix round 1: script mode fails closed
BROKEN = "def double(x):\n    return x + 2\n"
FIXED = "def double(x):\n    return x * 2\n"


def _red_then_green(tmp_path, tests, failed):
    ws = _ws(tmp_path, {"mylib.py": BROKEN, "test_mylib.py": tests})
    rep = run_tests(ws)
    assert not rep.ok and rep.failed == failed
    (ws / "mylib.py").write_text(FIXED, encoding="utf-8")
    assert run_tests(ws).ok
    return ws


def test_a_pytest_style_class_is_run(tmp_path):
    tests = ("from mylib import double\n\n\nclass TestDouble:\n    def test_two(self):\n        assert double(2) == 4\n\n"
             "    def test_zero(self):\n        assert double(0) == 0\n")
    ws = _red_then_green(tmp_path, tests, ["test_zero"])
    (ws / "mylib.py").write_text(BROKEN, encoding="utf-8")
    assert run_tests(ws, only=["test_zero"]).failed == ["test_zero"]   # by name: the whole file is run


def test_setup_method_of_a_pytest_style_class_is_called(tmp_path):
    tests = ("from mylib import double\n\n\nclass TestDouble:\n    def setup_method(self, method):\n        self.x = 0\n\n"
             "    def test_zero(self):\n        assert double(self.x) == 0\n")
    _red_then_green(tmp_path, tests, ["test_zero"])


def test_plain_functions_and_a_class_together(tmp_path):
    tests = ("from mylib import double\n\n\ndef test_two():\n    assert double(2) == 4\n\n\n"
             "class TestMore:\n    def test_zero(self):\n        assert double(0) == 0\n")
    _red_then_green(tmp_path, tests, ["test_zero"])


def test_a_class_that_needs_arguments_is_red(tmp_path):
    tests = ("from mylib import double\n\n\nclass TestDouble:\n    def __init__(self, x):\n        self.x = x\n\n"
             "    def test_zero(self):\n        assert double(0) == 0\n")
    ws = _ws(tmp_path, {"mylib.py": FIXED, "test_mylib.py": tests})
    assert not run_tests(ws).ok


def test_async_and_generator_tests_are_red(tmp_path):
    for i, tests in enumerate(["from mylib import double\n\n\nasync def test_zero():\n    assert double(0) == 0\n",
                               "from mylib import double\n\n\ndef test_zero():\n    yield\n    assert double(0) == 0\n",
                               "from mylib import double\n\n\nclass TestD:\n    async def test_zero(self):\n"
                               "        assert double(0) == 0\n"]):
        ws = _ws(tmp_path, {"mylib.py": FIXED, "test_mylib.py": tests}, f"t{i}")
        assert not run_tests(ws).ok, tests


def test_a_file_of_uncalled_checks_is_red(tmp_path):
    tests = '"""Checks."""\nfrom mylib import double\n\nN = 2\n\n\ndef check_zero():\n    assert double(0) == 0\n'
    ws = _ws(tmp_path, {"mylib.py": FIXED, "test_mylib.py": tests})
    assert not run_tests(ws).ok
    # the same checks called at the top level are a script: green with the right mylib
    # (a failing top-level assert already fails the import: red as "<unknown>", as before)
    _red_then_green(tmp_path / "b", tests + "\n\ncheck_zero()\n", ["<unknown>"])


# ------------------------------------------------------------------ fix round 2: a pytest class with a base
def test_a_pytest_class_with_an_object_base_runs_by_name(tmp_path):
    tests = ("from mylib import double\n\n\nclass TestDouble(object):\n    def test_zero(self):\n"
             "        assert double(0) == 0\n")
    ws = _ws(tmp_path, {"mylib.py": FIXED, "test_mylib.py": tests})
    assert run_tests(ws, only=["test_zero"]).ok          # the whole file: unittest cannot load it by id
    (ws / "mylib.py").write_text(BROKEN, encoding="utf-8")
    assert run_tests(ws, only=["test_zero"]).failed == ["test_zero"]


def test_phase1_repairs_a_pytest_class_with_an_object_base(tmp_path):
    from harness.solvers import coding_solver
    for i, base in enumerate(("", "(object)")):
        ws = _ws(tmp_path, {"mylib.py": "def top(xs, n):\n    return sorted(xs)[:n - 1]\n",
                            "test_mylib.py": f"from mylib import top\n\n\nclass TestTop{base}:\n    def test_top(self):\n"
                                             "        assert top([3, 1, 2], 2) == [1, 2]\n        assert top([5], 1) == [5]\n"},
                 f"c{i}")
        r = coding_solver.solve(Task(f"c{i}", "coding", INS, 2.0, "unittest", ws.parent, ws))
        assert "no repair found" not in r.note, (base, r.note)
        assert "repaired" in r.note and run_tests(ws).ok, (base, r.note)   # e.g. `n - 1` -> `n - 0`


def test_testcase_files_keep_their_per_test_ids(tmp_path):
    from harness.solvers.testrun import _test_ids
    p = tmp_path / "test_mylib.py"
    p.write_text("import unittest\nfrom unittest import TestCase\n\n\nclass A(unittest.TestCase):\n    def test_a(self):\n"
                 "        pass\n\n\nclass B(TestCase):\n    def test_b(self):\n        pass\n\n\n"
                 "class C(object):\n    def test_c(self):\n        pass\n", encoding="utf-8")
    assert _test_ids(p, ["test_a", "test_b"]) == ["test_mylib.A.test_a", "test_mylib.B.test_b"]
    assert _test_ids(p, ["test_a", "test_c"]) is None     # a pytest-style class: the whole file


# ------------------------------------------------------------------ fix round 3: a local TestCase base keeps its ids
LIB2 = "def top(xs, n):\n    return sorted(xs)[:n - 1]\n\n\ndef double(x):\n    return x + 2\n"
BODY2 = ("    def test_top(self):\n        self.assertEqual(top([3, 1, 2], 2), [1, 2])\n\n"
         "    def test_double(self):\n        self.assertEqual(double(3), 6)\n")


def test_phase1_repairs_two_bugs_through_a_local_testcase_base(tmp_path):
    """Reviewer's base3.py: `class Base(unittest.TestCase)` + `class TestM(Base)`, two bugs, repaired as on main."""
    from harness.solvers import coding_solver
    tests = ("import unittest\nfrom mylib import top, double\n\n\nclass Base(unittest.TestCase):\n    pass\n\n\n"
             "class TestM(Base):\n" + BODY2)
    ws = _ws(tmp_path, {"mylib.py": LIB2, "test_mylib.py": tests})
    r = coding_solver.solve(Task("c", "coding", INS, 2.0, "unittest", ws.parent, ws))
    assert r.status == "solved" and "repaired by mutation" in r.note, r.note
    assert run_tests(ws).ok


def test_which_classes_keep_per_test_ids(tmp_path):
    from harness.solvers.testrun import _test_ids
    p = tmp_path / "test_mylib.py"
    body = "    def test_a(self):\n        pass\n"
    src = {
        "local_testcase": "import unittest\nclass Base(unittest.TestCase):\n    pass\nclass Mid(Base):\n    pass\n"
                          "class T(Mid):\n" + body,
        "imported": "from helpers import BaseTest\nclass T(BaseTest):\n" + body,
        "mixin": "import unittest\nclass H:\n    pass\nclass T(H, unittest.TestCase):\n" + body,
        "cycle": "class A(T):\n    pass\nclass T(A):\n" + body,
        "local_pytest": "class H(object):\n    pass\nclass M(H):\n    pass\nclass T(M):\n" + body,
        "object": "class T(object):\n" + body,
        "no_base": "class T:\n" + body,
    }
    out = {}
    for k, v in src.items():
        p.write_text(v, encoding="utf-8")
        out[k] = _test_ids(p, ["test_a"])
    for k in ("local_testcase", "imported", "mixin", "cycle"):
        assert out[k] == ["test_mylib.T.test_a"], k
    for k in ("local_pytest", "object", "no_base"):
        assert out[k] is None, k

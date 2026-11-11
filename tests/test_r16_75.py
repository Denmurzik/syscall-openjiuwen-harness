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

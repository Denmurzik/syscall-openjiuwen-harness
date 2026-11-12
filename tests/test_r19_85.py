"""Test files below the workspace root and packages under test (round 19, task 85, HARNESS_DEEP_TESTS)."""
import asyncio

import pytest

from harness import llm_routes as L
from harness.llm import ChatResult
from harness.solvers import probe
from harness.solvers.testrun import run_in_dir
from harness.tasks import Task

NO_PYTEST = "raise ImportError('no pytest in this image')\n"
LIB_BAD = "def double(x):\n    return x + 2\n"
LIB_GOOD = "def double(x):\n    return x * 2\n"
UNIT = ("import unittest\n\nfrom mylib import double\n\n\nclass T(unittest.TestCase):\n"
        "    def test_double(self):\n        self.assertEqual(double(3), 6)\n")
PLAIN = "from mylib import double\n\n\ndef test_double():\n    assert double(3) == 6\n"
PARAM = ("import pytest\n\nfrom mylib import double\n\n\n@pytest.mark.parametrize('x, want', [(1, 2), (3, 6)])\n"
         "def test_double(x, want):\n    assert double(x) == want\n")


def _ws(tmp_path, files, name="t"):
    d = tmp_path / name
    for rel, text in files.items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return Task(name, "coding", "Some tests fail. Fix the bugs (do not edit the tests).", 2.0, "unittest", d,
                d / "workspace")


@pytest.fixture
def deep_off(monkeypatch):
    monkeypatch.setenv("HARNESS_DEEP_TESTS", "0")


# discovery

def test_deep_tests_only_when_the_root_has_none(tmp_path):
    t = _ws(tmp_path, {"mylib.py": LIB_BAD, "tests/test_b.py": UNIT, "tests/unit/test_a.py": UNIT,
                       "test/x_test.py": UNIT, ".venv/lib/test_v.py": UNIT, "tests/__pycache__/test_c.py": UNIT,
                       "node_modules/p/test_n.py": UNIT})
    ws = t.workspace
    assert L._deep_tests(ws) == ["test/x_test.py", "tests/test_b.py", "tests/unit/test_a.py"]
    assert L._test_file(ws) == "test/x_test.py"
    (ws / "test_root.py").write_text(UNIT, encoding="utf-8")
    assert L._deep_tests(ws) == [] and L._test_file(ws) == "test_root.py"


def test_deep_tests_off_is_round_18(tmp_path, deep_off):
    ws = _ws(tmp_path, {"mylib.py": LIB_BAD, "tests/test_b.py": UNIT}).workspace
    assert L._deep_tests(ws) == [] and L._test_file(ws) == "test_mylib.py"
    assert L._target_file(_ws(tmp_path, {"pkg/__init__.py": "def f():\n    pass\n", "tests/test_p.py":
                                         "from pkg import f\n"}, "p").workspace, "from pkg import f\n") == "mylib.py"


def test_read_tests_joins_several_files_under_their_paths(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": LIB_BAD, "tests/test_a.py": UNIT, "tests/test_b.py": PLAIN}).workspace
    name, text = L._read_tests(ws)
    assert name == "tests/test_a.py, tests/test_b.py"
    assert text.startswith("# tests/test_a.py\n") and "# tests/test_b.py\n" + PLAIN in text


# running a test file below the root

@pytest.mark.parametrize("tests", [UNIT, PLAIN])
def test_run_in_dir_runs_a_test_file_below_the_root(tmp_path, tests):
    ws = _ws(tmp_path, {"mylib.py": LIB_BAD, "tests/__init__.py": "", "tests/test_mylib.py": tests}).workspace
    assert run_in_dir(ws, "tests/test_mylib.py").ok is False
    (ws / "mylib.py").write_text(LIB_GOOD, encoding="utf-8")
    assert run_in_dir(ws, "tests/test_mylib.py").ok is True


def test_a_nested_test_dir_reads_its_data_from_the_root(tmp_path):
    tests = ("import unittest\nfrom pathlib import Path\nfrom mylib import double\n\n\nclass T(unittest.TestCase):\n"
             "    def test_d(self):\n        self.assertEqual(double(int(Path('n.txt').read_text())), 6)\n")
    ws = _ws(tmp_path, {"mylib.py": LIB_GOOD, "n.txt": "3", "tests/unit/test_d.py": tests}).workspace
    assert run_in_dir(ws, "tests/unit/test_d.py").ok is True


def test_the_pytest_stand_in_serves_a_file_below_the_root(tmp_path):
    ws = _ws(tmp_path, {"pytest.py": NO_PYTEST, "mylib.py": LIB_BAD, "tests/test_mylib.py": PARAM}).workspace
    rep = run_in_dir(ws, "tests/test_mylib.py")
    assert not rep.ok and rep.failed == ["test_double"]
    (ws / "mylib.py").write_text(LIB_GOOD, encoding="utf-8")
    assert run_in_dir(ws, "tests/test_mylib.py").ok is True
    # a conftest.py with hooks next to the test file is not followed by the stand-in: red, never green (expert 10)
    (ws / "tests" / "conftest.py").write_text("def pytest_configure(config):\n    pass\n", encoding="utf-8")
    assert run_in_dir(ws, "tests/test_mylib.py").ok is False


# the file under repair

A_TEST = "\n\ndef test_x():\n    pass\n"   # a test file defines a test (round 19, M2)


def test_package_targets(tmp_path):
    cases = [
        ({"geom/__init__.py": "from .shapes import area\n", "geom/shapes.py": "def area(w, h):\n    return w * h\n"},
         "from geom import area\n", "geom/shapes.py"),
        ({"inventory/__init__.py": "", "inventory/store.py": "def add(s, i, q):\n    pass\n",
          "inventory/report.py": "def total(s):\n    pass\n"},
         "import inventory.store as store\nfrom inventory.report import total\nstore.add({}, 'a', 1)\n",
         "inventory/store.py"),
        ({"src/wallet/__init__.py": "", "src/wallet/ledger.py": "def balance(e):\n    pass\n"},
         "from wallet.ledger import balance\n", "src/wallet/ledger.py"),
        ({"pkg/__init__.py": "", "pkg/core.py": "def f():\n    pass\n"}, "from pkg import core\ncore.f()\n",
         "pkg/core.py"),
        ({"pkg/__init__.py": "def f():\n    pass\n"}, "from pkg import f\n", "pkg/__init__.py"),
        ({"textkit/core.py": "def slug(t):\n    pass\n", "tests/helpers.py": "def slug(t):\n    pass\n"},
         "from tests.helpers import slug\nfrom textkit.core import slug as s2\n", "textkit/core.py"),
    ]
    for i, (files, tests, want) in enumerate(cases):
        ws = _ws(tmp_path, {**files, "tests/test_x.py": tests + A_TEST}, f"c{i}").workspace
        assert L._target_file(ws, tests) == want, (i, L._target_file(ws, tests))
    # a root module the tests import still comes first, and mylib.py whenever there is one
    ws = _ws(tmp_path, {"calc.py": "def f():\n    pass\n", "pkg/core.py": "def f():\n    pass\n",
                        "tests/test_x.py": "from calc import f\n" + A_TEST}, "r").workspace
    assert L._target_file(ws, "from calc import f\n") == "calc.py"


def test_package_targets_plain_import_and_star(tmp_path):
    # `import pkg` with pkg/__init__.py, and `from pkg import *` with an __init__ that re-exports `*` (round 19, M1)
    files = {"geom/__init__.py": "from .shapes import *\n", "geom/shapes.py": "def area(w, h):\n    return w + h\n"}
    for i, tests in enumerate(["import geom\n\n\ndef test_a():\n    assert geom.area(3, 4) == 12\n",
                               "from geom import *\n\n\ndef test_a():\n    assert area(3, 4) == 12\n"]):
        ws = _ws(tmp_path, {**files, "tests/test_g.py": tests}, f"m{i}").workspace
        assert L._target_file(ws, tests) == "geom/shapes.py", i


def test_deep_tests_skip_data_modules_and_samples(tmp_path):
    # a data module named like a test, samples and fixtures are not the task's tests (round 19, M2)
    ws = _ws(tmp_path, {"mylib.py": LIB_BAD, "tests/test_m.py": UNIT,
                        "pkgdata/test_input.py": "CASES = [(1, 2), (3, 6)]\n",
                        "tests/test_cases.py": "from mylib import double\n\nCASES = [(1, 2)]\n",
                        "tests/test_nothing.py": "def helper():\n    return 1\n",
                        "examples/test_demo.py": UNIT, "tests/fixtures/test_f.py": UNIT,
                        "lib/test_script.py": "from mylib import double\nassert double(3) == 6\n",
                        "other/test_free.py": "def test_a():\n    assert 1 + 1 == 2\n"}).workspace
    assert L._deep_tests(ws) == ["lib/test_script.py", "tests/test_m.py"]


# the ladder

class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append((model, messages))
        r = self.replies.pop(0)
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=10, cost=11)


PKG = {"geom/__init__.py": "from .shapes import area\n",
       "geom/shapes.py": "def area(w, h):\n    return w + h\n",
       "tests/test_geom.py": ("import unittest\nfrom geom import area\n\n\nclass T(unittest.TestCase):\n"
                              "    def test_area(self):\n        self.assertEqual(area(3, 4), 12)\n")}
FIX = "FUNC area\ndef area(w, h):\n    return w * h\n"


def test_ladder_repairs_a_package_module(tmp_path):
    t = _ws(tmp_path, PKG)
    c = Fake([FIX])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and len(c.calls) == 1
    assert "w * h" in (t.workspace / "geom" / "shapes.py").read_text(encoding="utf-8")
    user = c.calls[0][1][1]["content"]
    assert user.startswith("geom/shapes.py:\n") and "tests/test_geom.py:\n" in user


def test_ladder_package_off_is_round_18(tmp_path, deep_off):
    t = _ws(tmp_path, PKG)
    c = Fake([FIX])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "cannot read workspace" in r.note and c.calls == []


def test_ladder_runs_every_test_file_below_the_root(tmp_path):
    lib = "def double(x):\n    return x + 2\n\n\ndef neg(x):\n    return x\n"
    t2 = ("import unittest\nfrom mylib import neg\n\n\nclass N(unittest.TestCase):\n"
          "    def test_neg(self):\n        self.assertEqual(neg(3), -3)\n")
    t = _ws(tmp_path, {"mylib.py": lib, "tests/test_a.py": UNIT, "tests/test_b.py": t2})
    half = "FUNC double\ndef double(x):\n    return x * 2\n"
    both = half + "\nFUNC neg\ndef neg(x):\n    return -x\n"
    c = Fake([half, both])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and len(c.calls) == 2   # the half fix leaves tests/test_b.py red
    assert "return -x" in (t.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "# tests/test_b.py\n" in c.calls[0][1][1]["content"]


DISC = ("def bulk_price(unit, qty):\n    \"\"\"unit * qty, with 10 % off for 10 or more items.\"\"\"\n"
        "    total = unit * qty\n    if qty > 10:\n        total = total * 0.9\n    return round(total, 2)\n")
DISC_FIX = DISC.replace("qty > 10", "qty >= 10")
DISC_TESTS = ("import unittest\nfrom shop.pricing import bulk_price\n\n\nclass T(unittest.TestCase):\n"
              "    def test_small(self):\n        self.assertEqual(bulk_price(2, 3), 6)\n\n"
              "    def test_big(self):\n        self.assertEqual(bulk_price(1, 20), 18)\n")


def test_visible_call_sees_the_module_under_test_of_another_layout(tmp_path):
    assert probe.visible_call(DISC_TESTS, "bulk_price") == ((2, 3), "self.assertEqual(bulk_price(2, 3), 6)")
    aliased = "import inventory.store as store\n\n\ndef test_a():\n    assert store.add({}, 'a', 1) == {'a': 1}\n"
    assert probe.visible_call(aliased, "add")[0] == ({}, "a", 1)
    # a file that imports mylib reads only mylib's calls, as in round 18
    mixed = "from mylib import f\nfrom other import g\n\n\ndef test_a():\n    assert g(1) == f(2)\n"
    assert probe.visible_call(mixed, "g") is None and probe.visible_call(mixed, "f")[0] == (2,)
    # the standard library is never the module under test
    assert probe.visible_call("from math import gcd\n\n\ndef test_a():\n    assert gcd(4, 6) == 2\n", "gcd") is None


def test_visible_call_off_is_round_18(deep_off):
    assert probe.visible_call(DISC_TESTS, "bulk_price") is None

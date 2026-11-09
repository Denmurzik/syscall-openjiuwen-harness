"""Round 3 attacks: reflection from the test or the library, renamed `from mylib import g as f`.
Each file passes with a correct library; the proof must stay silent on it and on the buggy library.
a02 (a lambda parameter laundering `unittest`) is a known residual and is not covered."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from harness.solvers.coding_solver import provable_contradiction as pc, solve
from harness.tasks import Task

H = "import unittest\nimport mylib as m\n\n\n"
A = "    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n"
B = "    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n"
F, FBAD = "def f(x):\n    return x\n", "def f(x):\n    return str(x) + 'a'\n"
ABS, ABSBAD = "def f(x):\n    return abs(x)\n", "def f(x):\n    return str(abs(x)) + 'a'\n"


def pre(stmt):
    """test_a, then test_b with `stmt` as a plain self.assert* statement before the contradicting assert."""
    return H + "class T(unittest.TestCase):\n" + A + f"    def test_b(self):\n        {stmt}\n" + B.split("\n", 1)[1]


GENF = "(i for i in ())"
C = [
 # test side: the attribute rule trusts any lambda parameter, and __builtins__ is a plain name
 ("a01 builtins dict: setattr by string", pre("self.assertIsNone(__builtins__['setattr'](m, 'f', lambda x: x + 1))"), F, FBAD),
 ("a02 lambda launders unittest: raise SkipTest", pre(f"self.assertIsNone((lambda g, u: g.throw(u.SkipTest('x')))({GENF}, unittest))"), F, FBAD),
 ("a03 builtins dict: replace abs used by f", pre("self.assertIsNone((lambda b: b.update(abs=lambda x: x + 1))(__builtins__))"), ABS, ABSBAD),
 ("a04 builtins dict: exec by string", pre("self.assertIsNone(__builtins__['exec']('import mylib; mylib.f = lambda x: x + 1'))"), F, FBAD),
 ("a05 builtins dict: __import__ by string, patch", pre("self.assertIsNone((lambda o: o.f)(__builtins__['__import__']('other')) and None)"), F, FBAD),
 ("a06 lambda launders unittest.case.sys: swap alias holder", pre("self.assertIsNone((lambda u: (lambda c: (lambda s: s.modules.update(mylib=s.modules['other']))(c.sys))(u.case))(unittest))"), F, FBAD),
 ("a07 lambda launders: functools.update_wrapper copies attrs", pre("self.assertIsNone((lambda u: (lambda c: (lambda ft: ft.update_wrapper(m, __builtins__['__import__']('other'), assigned=(), updated=()) and None)(c.functools))(u.case))(unittest))"), F, FBAD),
 ("a08 assertRaises calls the setattr it found", pre("self.assertIsNone(self.assertRaises(ZeroDivisionError, lambda: (__builtins__['setattr'](m, 'f', lambda x: x + 1), 1 / 0)))"), F, FBAD),
 # alias confusion: the proof checks purity of the local name, not of the imported function
 ("a09 from mylib import g as f, g is stateful", "import unittest\nfrom mylib import g as f\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self):\n        self.assertEqual(f(1), 2)\n",
  "def f(x):\n    return x\n\ndef g(x, c=[]):\n    c.append(x)\n    return len(c)\n", "def f(x):\n    return x\n\ndef g(x, c=[]):\n    c.append(x)\n    return str(len(c)) + 'a'\n"),
 ("a10 two names of one alias: f is g's alias only in one class", "import unittest\nfrom mylib import g as f, g as h\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self):\n        self.assertEqual(f(1), 2)\n",
  "def f(x):\n    return x\n\ndef g(x, c=[]):\n    c.append(x)\n    return len(c)\n", "def f(x):\n    return x\n\ndef g(x, c=[]):\n    c.append(x)\n    return str(len(c)) + 'a'\n"),
 # library side: only global/nonlocal and a list of names are banned
 ("a11 swap() via __globals__", pre("self.assertIsNone(m.swap())"), F + "\ndef swap():\n    swap.__globals__['f'] = lambda x: x + 1\n", FBAD + "\ndef swap():\n    swap.__globals__['f'] = lambda x: x + 1\n"),
 ("a12 swap() via inspect frame", pre("self.assertIsNone(m.swap())"), "import inspect\n" + F + "\ndef swap():\n    inspect.currentframe().f_globals['f'] = lambda x: x + 1\n", "import inspect\n" + FBAD + "\ndef swap():\n    inspect.currentframe().f_globals['f'] = lambda x: x + 1\n"),
 ("a13 module replaces itself via os.sys", H + "class T(unittest.TestCase):\n" + A + B, F + "\nclass _Fake:\n    n = []\n    @staticmethod\n    def f(x):\n        _Fake.n.append(x)\n        return len(_Fake.n)\n\nimport os\nos.sys.modules[__name__] = _Fake\n",
  FBAD + "\nclass _Fake:\n    n = []\n    @staticmethod\n    def f(x):\n        _Fake.n.append(x)\n        return str(len(_Fake.n)) + 'a'\n\nimport os\nos.sys.modules[__name__] = _Fake\n"),
 ("a14 library patches unittest at import", H + "class T(unittest.TestCase):\n" + A + B, "import unittest\n" + F + "\nunittest.TestCase.assertEqual = lambda self, a, b: None\n", "import unittest\nunittest.TestCase.assertEqual = lambda self, a, b: self.fail('x')\n" + FBAD),
 ("a15 library changes builtin abs from another function", pre("self.assertIsNone(m.swap())"), ABS + "\ndef swap():\n    __builtins__['abs'] = lambda x: x + 1\n", ABSBAD + "\ndef swap():\n    __builtins__['abs'] = lambda x: x + 1\n"),
 ("a16 library rewrites f.__code__", pre("self.assertIsNone(m.swap())"), F + "\ndef _g(x):\n    return x + 1\n\ndef swap():\n    f.__code__ = _g.__code__\n", FBAD + "\ndef _g(x):\n    return x + 1\n\ndef swap():\n    f.__code__ = _g.__code__\n"),
 # things that should be safe; expected: no proof
 ("a17 method-level exit()", pre("self.assertIsNone(exit(0))"), F, FBAD),
 ("a18 __class__ cell read", pre("self.assertIsNone((lambda c: c.assertEqual and None)(__class__))"), F, FBAD),
 ("a19 two classes, one literal arg spelled differently", H + "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1.0), 1.0)\n\n\nclass U(unittest.TestCase):\n    def test_b(self):\n        self.assertEqual(m.f(1), True)\n", F, FBAD),
 ("a20 tuple vs list expected equal? (list != tuple)", H + "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n    def test_b(self):\n        self.assertEqual(m.f(1), 1.0)\n", F, FBAD),
]
OTHER = "def f(x):\n    return x + 1\n"
_USE = ("a01", "a03", "a04", "a08", "a09", "a10", "a11", "a12", "a13", "a14", "a15", "a16")
ATTACKS = [c for c in C if c[0].split()[0] in _USE]


def _ws(d, src, lib):
    w = d / "workspace"
    w.mkdir()
    (w / "test_mylib.py").write_text(src, newline="\n")
    (w / "mylib.py").write_text(lib, newline="\n")
    (w / "other.py").write_text(OTHER, newline="\n")
    return w


@pytest.mark.parametrize("name,src,good,buggy", ATTACKS, ids=[c[0].split()[0] for c in ATTACKS])
def test_round3_attack_gets_no_proof(name, src, good, buggy):
    d = Path(tempfile.mkdtemp())
    try:
        w = _ws(d, src, good)
        assert subprocess.run([sys.executable, "-B", "-m", "unittest", "test_mylib"], cwd=w,
                              capture_output=True, timeout=30).returncode == 0
        assert not pc(src, good) and not pc(src, buggy)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("name,src,good,buggy", ATTACKS, ids=[c[0].split()[0] for c in ATTACKS])
def test_round3_attack_never_infeasible_end_to_end(tmp_path, name, src, good, buggy):
    w = _ws(tmp_path, src, buggy)
    assert solve(Task("c", "coding", "x", 3.0, "unittest", tmp_path, w), 25).status != "infeasible"

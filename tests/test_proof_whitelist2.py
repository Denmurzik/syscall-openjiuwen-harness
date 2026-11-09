"""Round 2 attacks on the strict proof: bodies, helper modules and classes, library state.
Every file here passes with a correct library (checked for real); the proof must stay silent."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from harness.solvers.coding_solver import provable_contradiction as pc, solve
from harness.tasks import Task

LIB = "def f(x):\n    return x\n"
BUG = "def f(x):\n    return str(x) + 'a'\n"
H = "import unittest\nimport mylib as m\n\n"
A = "    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n"
B = "    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n"
TWO = "class T(unittest.TestCase):\n" + A + B
OTHER = ("import mylib\n"
         "def install():\n    mylib.f = lambda x: x + 1\n"
         "def load_tests(l, t, p):\n    import test_mylib\n    return l.loadTestsFromTestCase(test_mylib.T)\n")
SWAP = "import mylib\nn = [0]\ndef _f(x):\n    n[0] += 1\n    return n[0]\nmylib.f = _f\n"
C = {
 "n1 class body of a helper class deletes the test": H + TWO + "class Evil:\n    T.test_b = None\n",
 "n2 alias copy, then patch": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        alias = m\n        alias.f = lambda x: x + 1\n        self.assertEqual(m.f(1), 2)\n",
 "n3 helper module patches mylib when called": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        other.install()\n        self.assertEqual(m.f(1), 2)\n",
 "n4 helper module swaps in a stateful f on import": H + "import swap\n" + TWO,
 "n5 load_tests imported from a helper": H + "from other import load_tests\nclass T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n" + B,
 "n6 class named like the alias": "import unittest\nimport mylib as m\nclass m:\n    n = [0]\n    @staticmethod\n    def f(x):\n        m.n[0] += 1\n        return m.n[0]\n\n" + TWO,
 "n7 local def named like the imported function": "import unittest\nfrom mylib import f\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self):\n        def f(x):\n            return x + 1\n        self.assertEqual(f(1), 2)\n",
 "n8 match capture rebinding alias": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        match other:\n            case m:\n                pass\n        self.assertEqual(m.f(1), 2)\n",
 "n9 instance override of assertEqual": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertEqual = lambda a, b: None\n        self.assertEqual(m.f(1), 2)\n",
 "n10 yield makes the test a generator, body never runs": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n        yield\n",
 "n11 class patches TestCase.assertEqual": H + TWO + "class Evil:\n    unittest.TestCase.assertEqual = lambda self, a, b: None\n",
 "n12 setUpModule imported from a helper": H + "from helperfix import setUpModule\n" + TWO,
 "n6b class named like the alias, stateful lambda": "import unittest\nimport mylib as m\nclass m:\n    f = staticmethod(lambda x, s=[]: (s.append(x), len(s))[1])\n\n" + TWO,
 "n7b local class named like the imported function": "import unittest\nfrom mylib import f\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self):\n        class f:\n            __new__ = lambda c, x: x + 1\n        self.assertEqual(f(1), 2)\n",
 "n13 library function rebinds f, called from the test": ("LIB2", H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        m.swap()\n        self.assertEqual(m.f(1), 2)\n"),
 "n14 same, the call hidden inside an assert": ("LIB2", H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertIsNone(m.swap())\n        self.assertEqual(m.f(1), 2)\n"),
 "n15 m.__setattr__ inside an assert": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertIsNone(m.__setattr__('f', lambda x: x + 1))\n        self.assertEqual(m.f(1), 2)\n",
 "must prove: plain two classes": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n" + B + "\nif __name__ == '__main__':\n    unittest.main()\n",
 "must prove: same method": H + "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n        self.assertEqual(m.f(1), 2)\n",
}
LIB2 = LIB + "\ndef swap():\n    global f\n    f = lambda x: x + 1\n"
OTHER = "def f(x):\n    return x + 1\n" + OTHER
HELP = {"other.py": OTHER, "swap.py": SWAP, "helperfix.py": "import mylib\nn = [0]\ndef _f(x):\n    n[0] += 1\n    return n[0]\ndef setUpModule():\n    mylib.f = _f\n"}


def _case(name):
    src = C[name]
    lib = LIB
    if isinstance(src, tuple):
        lib, src = LIB2, src[1]
    return lib, src


def _workspace(d, lib, src):
    w = d / "workspace"
    w.mkdir()
    (w / "mylib.py").write_text(lib, newline="\n")
    (w / "test_mylib.py").write_text(src, newline="\n")
    for k, v in HELP.items():
        (w / k).write_text(v, newline="\n")
    return w


@pytest.mark.parametrize("name", list(C))
def test_strict_proof_silent_on_passable_files(name):
    lib, src = _case(name)
    d = Path(tempfile.mkdtemp())
    try:
        w = _workspace(d, lib, src)
        passes = subprocess.run([sys.executable, "-B", "-m", "unittest", "test_mylib"], cwd=w,
                                capture_output=True, timeout=30).returncode == 0
        res = pc(src, lib)
        if name.startswith("must prove"):
            assert res and not passes
        else:
            assert passes and not res
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("name", [n for n in C if not n.startswith("must prove")])
def test_unrepairable_bug_never_infeasible_for_round2_attacks(tmp_path, name):
    lib, src = _case(name)
    w = _workspace(tmp_path, lib, src)
    (w / "mylib.py").write_text(BUG + lib[len(LIB):], newline="\n")
    t = Task("c", "coding", "x", 3.0, "unittest", tmp_path, w)
    assert solve(t, 30).status != "infeasible"

"""File-wide contradiction proof: no false `infeasible` on test files a correct library passes
(each case is run for real against LIB and `other.py`), real contradictions are still proven."""
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from harness.solvers.coding_solver import provable_contradiction as pc, solve
from harness.tasks import Task

LIB = "def f(x):\n    return x\n\ndef g(x):\n    return x + 1\n"
H = "import unittest\nimport mylib as m\n\n"
A = "    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n"
B = "    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n"
TWO = "class T(unittest.TestCase):\n" + A + B
TWOC = "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n" + B
HELP = {"other.py": "def f(x):\n    return x + 1\n"}
C = {
 "del class at module level": H + TWOC + "del U\n",
 "del method at module level": H + TWO + "del T.test_b\n",
 "rebind class to None": H + TWOC + "U = None\n",
 "rebind method to None": H + TWO + "T.test_b = None\n",
 "load_tests picks one class": H + TWOC + "def load_tests(l, t, p):\n    return l.loadTestsFromTestCase(T)\n",
 "__unittest_skip__ class attr": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    __unittest_skip__ = True\n    __unittest_skip_why__ = 'x'\n" + B,
 "method-local import other as m": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        import other as m\n        self.assertEqual(m.f(1), 2)\n",
 "method-local from other import f": "import unittest\nfrom mylib import f\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self):\n        from other import f\n        self.assertEqual(f(1), 2)\n",
 "param default shadows imported name": "import unittest\nfrom mylib import f\nimport other\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n    def test_b(self, f=other.f):\n        self.assertEqual(f(1), 2)\n",
 "param default shadows alias": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self, m=other):\n        self.assertEqual(m.f(1), 2)\n",
 "for-loop rebinding alias": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        for m in [other]:\n            pass\n        self.assertEqual(m.f(1), 2)\n",
 "with-as rebinding alias": H + "import other, contextlib\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        with contextlib.nullcontext(other) as m:\n            pass\n        self.assertEqual(m.f(1), 2)\n",
 "walrus rebinding alias": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        (m := other)\n        self.assertEqual(m.f(1), 2)\n",
 "both methods import locally, different modules": "import unittest\nimport mylib as m\nclass T(unittest.TestCase):\n    def test_a(self):\n        import mylib as m\n        self.assertEqual(m.f(1), 1)\n    def test_b(self):\n        import other as m\n        self.assertEqual(m.f(1), 2)\n",
 "globals() rebinding": H + "import other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        globals()['m'] = other\n        self.assertEqual(m.f(1), 2)\n",
 "exec rebinding": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        exec('import other; m.f = other.f')\n        self.assertEqual(m.f(1), 2)\n",
 "sys.modules lookup": H + "import sys, other\nclass T(unittest.TestCase):\n" + A + "    def test_b(self):\n        m = sys.modules['other']\n        self.assertEqual(m.f(1), 2)\n",
 "overridden assertEqual in one class": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    def assertEqual(self, a, b):\n        pass\n" + B,
 "addTypeEqualityFunc in __init__": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    def __init__(self, *a):\n        super().__init__(*a)\n        self.addTypeEqualityFunc(int, lambda a, b, msg=None: None)\n" + B,
 "setUp via assignment": H + "def _s(self):\n    import other\n    global m\n    m = other\nclass T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    setUp = _s\n" + B,
 "asyncSetUp patches (IsolatedAsyncioTestCase)": H + "import other\nclass T(unittest.TestCase):\n" + A + "\nclass U(unittest.IsolatedAsyncioTestCase):\n    async def asyncSetUp(self):\n        global m\n        m = other\n" + B,
 "run() override": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    def run(self, result=None):\n        return result\n" + B,
 "user class named *TestCase, not a unittest class": H + "class MyTestCase:\n    pass\nclass T(unittest.TestCase):\n" + A + "\nclass U(MyTestCase):\n" + B,
 "mixin with setUp": H + "import other\nclass Mix:\n    def setUp(self):\n        global m\n        m = other\nclass T(unittest.TestCase):\n" + A + "\nclass U(Mix, unittest.TestCase):\n" + B,
 "base test overridden in subclass": H + "class Base(unittest.TestCase):\n" + B + "\nclass T(Base):\n    def test_b(self):\n        pass\n" + A,
 "assertNotEqual same call": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertNotEqual(m.f(1), 2)\n",
 "class attr & shared state": H + "class T(unittest.TestCase):\n    v = 1\n" + A + "    def test_b(self):\n        T.v = 2\n        self.assertEqual(m.f(self.v), 2)\n",
 "parametrised helper": H + "class T(unittest.TestCase):\n    def _c(self, x, y):\n        self.assertEqual(m.f(x), y)\n    def test_a(self):\n        self._c(1, 1)\n    def test_b(self):\n        self._c(2, 2)\n",
 "arg object mutated": H + "class T(unittest.TestCase):\n    xs = [1]\n    def test_a(self):\n        self.assertEqual(m.f(self.xs), [1])\n    def test_b(self):\n        self.xs.append(2)\n        self.assertEqual(m.f(self.xs), [1, 2])\n",
 "unittest.main(defaultTest=T) guard": H + TWOC + "if __name__ == '__main__':\n    unittest.main(defaultTest='T')\n",
 "module-level if deletes class": H + TWOC + "if True:\n    del U\n",
 "class in nested function never run": H + "class T(unittest.TestCase):\n" + A + "\ndef _unused():\n    class U(unittest.TestCase):\n" + textwrap.indent(B, "    "),
 "__test__ False": H + "class T(unittest.TestCase):\n" + A + "\nclass U(unittest.TestCase):\n    __test__ = False\n" + B,
 "metaclass strips test_b": H + "class Meta(type):\n    def __new__(c, n, b, d):\n        d.pop('test_b', None)\n        return super().__new__(c, n, b, d)\nclass T(unittest.TestCase, metaclass=Meta):\n" + A + B,
 "__init_subclass__ strips": H + "class Base(unittest.TestCase):\n    def __init_subclass__(cls):\n        cls.test_b = None\nclass T(Base):\n" + A + B,
 "__init_subclass__ on TestCase-ending base": H + "class BaseTestCase(unittest.TestCase):\n    def __init_subclass__(cls):\n        cls.test_b = None\nclass T(BaseTestCase, unittest.TestCase):\n" + A + B,
 "two names: from mylib f and g as f (module order)": "import unittest\nfrom mylib import f\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 2)\nfrom mylib import g as f\nclass U(unittest.TestCase):\n    def test_b(self):\n        self.assertEqual(f(1), 2)\n",
 "other module imported over alias": "import unittest\nimport mylib as m\nimport other as m\nclass T(unittest.TestCase):\n    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n",
 "other module over alias, two methods same value": "import unittest\nimport mylib as m\nimport other as m\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 2)\n    def test_b(self):\n        self.assertEqual(m.f(1), 2)\n",
 "two alias names": "import unittest\nimport mylib as m\nimport other as n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(m.f(1), 1)\n    def test_b(self):\n        self.assertEqual(n.f(1), 2)\n",
 "skip via self.fail no; addCleanup harmless": H + "class T(unittest.TestCase):\n" + A + "    def test_b(self):\n        self.assertEqual(m.f(2), 2)\n",
 "contradiction in a real pair (must prove)": H + TWO,
 "contradiction in two classes (must prove)": H + TWOC,
 "with unittest.main() guard (must prove)": H + TWOC + "if __name__ == '__main__':\n    unittest.main()\n",
}
MUST_PROVE = {n for n in C if "(must prove)" in n}


@pytest.mark.parametrize("name", list(C))
def test_proof_never_infeasible_on_a_passable_file(name):
    src = C[name]
    d = Path(tempfile.mkdtemp())
    try:
        (d / "mylib.py").write_text(LIB)
        (d / "test_mylib.py").write_text(src)
        for k, v in HELP.items():
            (d / k).write_text(v)
        p = subprocess.run([sys.executable, "-B", "-m", "unittest", "test_mylib"], cwd=d,
                           capture_output=True, text=True, timeout=30)
        res = pc(src, LIB)
        if name in MUST_PROVE:
            assert res and p.returncode != 0
        else:
            assert not (res and p.returncode == 0), "false infeasible"
            if name in NEVER_PROVE:
                assert not res
    finally:
        shutil.rmtree(d, ignore_errors=True)


NEVER_PROVE = set(C) - MUST_PROVE


_BUG = "def f(x):\n    return str(x) + 'a'\n"   # no mutation repairs this; `return x` does
_GOOD = "def f(x):\n    return x\n"
_E2E = {n: C[n] for n in ("del class at module level", "overridden assertEqual in one class",
                          "method-local import other as m", "load_tests picks one class")}


@pytest.mark.parametrize("name", list(_E2E))
def test_unrepairable_bug_on_passable_file_is_not_infeasible(tmp_path, name):
    w = tmp_path / "workspace"
    w.mkdir()
    (w / "mylib.py").write_text(_BUG, newline="\n")
    (w / "test_mylib.py").write_text(_E2E[name],
                                     newline="\n")
    (w / "other.py").write_text(HELP["other.py"], newline="\n")
    t = Task("c", "coding", "x", 3.0, "unittest", tmp_path, w)
    assert solve(t, 40).status != "infeasible"

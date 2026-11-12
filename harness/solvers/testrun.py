"""Run a task's unittest file in a throw-away copy of the workspace."""
import ast
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from harness.proc import run_python

_MARK = "@@HARNESS_RESULT@@"

# Runs the given unittest targets and prints one compact JSON line with the failed method
# names, so the verdict never depends on (8 KB-truncated) traceback output.
_RUNNER = r'''
import json, sys, unittest
try:
    import pytest  # the real one, when the image has it
except Exception:
    # The image has no pytest: a test file that says `import pytest` (for pytest.raises, approx, parametrize, a
    # fixture) could not even load, so no repair of the task could ever turn green. A stand-in with the parts that
    # plain tests use; `_fixtures` lets the loader below fill fixture parameters.
    import re as _re, types as _types
    _pt = _types.ModuleType("pytest")
    _pt._fixtures = {}

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match, self.value, self.type = exc, match, None, None

        def __enter__(self):
            return self

        def __exit__(self, tp, val, tb):
            if tp is None:
                raise AssertionError("DID NOT RAISE %%r" %% (self.exc,))
            if not issubclass(tp, self.exc):
                return False
            if self.match is not None and not _re.search(self.match, str(val)):
                raise AssertionError("the exception message does not match %%r" %% (self.match,))
            self.value, self.type = val, tp
            return True

    def _raises(exc, *args, match=None, **kw):
        if args:
            with _Raises(exc):
                args[0](*args[1:], **kw)
            return None
        return _Raises(exc, match)

    class _Approx:
        def __init__(self, expected, rel=None, abs=None, nan_ok=False):
            self.expected, self.rel, self.abs = expected, rel, abs

        def _close(self, a, e):
            if isinstance(e, (list, tuple)):
                return isinstance(a, (list, tuple)) and len(a) == len(e) and all(self._close(x, y) for x, y in zip(a, e))
            if isinstance(e, dict):
                return isinstance(a, dict) and a.keys() == e.keys() and all(self._close(a[k], e[k]) for k in e)
            try:
                tol = max((1e-6 if self.rel is None else self.rel) * abs(e), 1e-12 if self.abs is None else self.abs)
                return abs(a - e) <= tol
            except TypeError:
                return a == e

        def __eq__(self, other):
            return self._close(other, self.expected)

        def __ne__(self, other):
            return not self == other

        def __repr__(self):
            return "approx(%%r)" %% (self.expected,)

    class _Param:
        def __init__(self, *values, **kw):
            self.values = values

    def _parametrize(names, values, **_kw):
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)

        def deco(f):
            def run(*a, **k):
                for v in values:
                    case = v.values if isinstance(v, _Param) else (v if len(names) > 1 else (v,))
                    f(*a, **dict(zip(names, case)), **k)
            run.__name__, run.__module__, run.__qualname__ = f.__name__, f.__module__, f.__qualname__
            run._params = set(names) | getattr(f, "_params", set())
            return run
        return deco

    def _skipping(*a, **k):
        def skipped(*_a, **_k):
            raise unittest.SkipTest("skipped")

        def deco(f):
            skipped.__name__, skipped.__module__ = f.__name__, f.__module__
            return skipped
        return deco(a[0]) if len(a) == 1 and callable(a[0]) and not k else deco

    def _skipif(cond=False, *a, **k):
        return _skipping if cond else (lambda f: f)

    def _plain(*a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f)

    class _Mark:
        parametrize = staticmethod(_parametrize)
        skip = staticmethod(_skipping)
        xfail = staticmethod(_skipping)
        skipif = staticmethod(_skipif)

        def __getattr__(self, name):
            return _plain

    def _fixture(*a, **k):
        def deco(f):
            _pt._fixtures[k.get("name") or f.__name__] = f
            return f
        return deco(a[0]) if len(a) == 1 and callable(a[0]) and not k else deco

    def _fail(reason=""):
        raise AssertionError(reason)

    def _skip(reason=""):
        raise unittest.SkipTest(reason)

    _pt.raises, _pt.approx, _pt.param, _pt.mark, _pt.fixture = _raises, _Approx, _Param, _Mark(), _fixture
    _pt.fail, _pt.skip = _fail, _skip
    sys.modules["pytest"] = _pt
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
script_ok = None
if suite.countTestCases() == 0 and len(sys.argv) == 2 and "." not in sys.argv[1]:
    # no TestCase in the file: plain test functions (pytest style), or a script of bare asserts
    import ast, importlib, inspect
    mod = importlib.import_module(sys.argv[1])

    def _free(f, skip=0):
        return not any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                       for p in list(inspect.signature(f).parameters.values())[skip:])

    def _filled(f, skip=0):
        # the test with its fixture parameters filled from the stand-in's registry; None when it cannot be called
        if _free(f, skip):
            return f
        reg = getattr(sys.modules.get("pytest"), "_fixtures", None)
        need = [p.name for p in list(inspect.signature(f).parameters.values())[skip:]
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not reg or any(n not in reg for n in need):
            return None

        def value(name, depth=0):
            g = _filled(reg[name]) if depth < 5 else None
            if g is None:
                raise RuntimeError("fixture %%r cannot be built" %% (name,))
            v = g()
            return next(v) if inspect.isgenerator(v) else v

        def run(*a):
            return f(*a, **{n: value(n) for n in need})
        run.__name__, run.__module__ = f.__name__, f.__module__
        return run

    def _sync(f):
        # an async or generator test returns without running its body when called: it can never pass here
        return not (inspect.iscoroutinefunction(f) or inspect.isgeneratorfunction(f) or inspect.isasyncgenfunction(f))

    def _method_case(c, m):
        def run():
            o = c()   # a class whose constructor needs arguments errors here: red
            hook = getattr(o, "setup_method", None)
            if callable(hook):
                hook(getattr(o, m))
            try:
                _filled(getattr(c, m), 1)(o)
            finally:
                hook = getattr(o, "teardown_method", None)
                if callable(hook):
                    hook(getattr(o, m))
        run.__name__ = m
        return unittest.FunctionTestCase(run)

    def _acts(path):
        # a statement that tests something when the file runs: anything but imports, defs, classes, assignments
        # and docstrings (a file of uncalled check_* functions tests nothing)
        try:
            with open(path, encoding="utf-8") as fh:
                body = ast.parse(fh.read()).body
        except Exception:
            return True
        quiet = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                 ast.AnnAssign, ast.AugAssign, ast.Pass)
        return any(not isinstance(n, quiet) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
                   for n in body)
    fns = [f for n, f in vars(mod).items() if n.startswith("test") and inspect.isfunction(f)
           and f.__module__ == mod.__name__]
    # pytest-style classes: Test* without a TestCase base, their test* methods
    classes = [c for n, c in vars(mod).items() if n.startswith("Test") and inspect.isclass(c)
               and c.__module__ == mod.__name__ and not issubclass(c, unittest.TestCase)
               and getattr(c, "__test__", True) is not False]
    methods = [(c, m) for c in classes for m in dir(c) if m.startswith("test") and inspect.isfunction(getattr(c, m))]
    if (fns or methods) and all(_filled(f) is not None and _sync(f) for f in fns) and all(
            _filled(getattr(c, m), 1) is not None and _sync(getattr(c, m)) for c, m in methods):
        suite = unittest.TestSuite([unittest.FunctionTestCase(_filled(f)) for f in fns]
                                   + [_method_case(c, m) for c, m in methods])
    elif not fns and not methods and not _acts(mod.__file__):
        pass   # nothing in the file tests anything: no test ran, red
    elif not fns and not methods:
        # `python3 test_x.py` is the whole test: it passes when the script ends without an error
        import runpy
        try:
            runpy.run_path(mod.__file__, run_name="__main__")
            script_ok = True
        except SystemExit as e:
            script_ok = e.code in (None, 0)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            script_ok = False
        sys.stdout.flush()
        print("%s" + json.dumps({"ok": script_ok, "failed": [] if script_ok else ["<script>"]}), flush=True)
        sys.exit(0)
res = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)
names = []
for t, _ in list(res.failures) + list(res.errors):
    t = getattr(t, "test_case", t)
    n = (getattr(getattr(t, "_testFunc", None), "__name__", None) or getattr(t, "_testMethodName", None)
         or t.id().rsplit(".", 1)[-1])
    if n not in names:
        names.append(n)
sys.stdout.flush()
print("%s" + json.dumps({"ok": res.wasSuccessful() and res.testsRun > 0, "failed": names}), flush=True)
''' % (_MARK, _MARK)


@dataclass(frozen=True)
class TestReport:
    __test__ = False  # not a pytest class

    ok: bool
    failed: list[str] = field(default_factory=list)
    output: str = ""


def _pytest_style(cls: ast.ClassDef, local: dict, seen: frozenset = frozenset()) -> bool:
    """A class unittest cannot load by id: no base, or only `object` and local classes that are pytest-style
    themselves (followed through the file, a cycle counts as not pytest-style). Any other base (`unittest.TestCase`,
    an imported or unknown name, a local class that reaches TestCase) keeps the per-test ids."""
    if cls.name in seen:
        return False
    for b in cls.bases:
        if isinstance(b, ast.Name) and b.id == "object":
            continue
        if isinstance(b, ast.Name) and b.id in local and _pytest_style(local[b.id], local, seen | {cls.name}):
            continue
        return False
    return True


def _test_ids(test_path: Path, names: list[str]) -> list[str] | None:
    """Resolve method names to unittest ids; None when any name is unknown or sits in a pytest-style class
    (`class TestX:`, `class TestX(object):`), which unittest cannot load by id: the whole file is run instead."""
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    local = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}
    ids = []
    for name in names:
        hits = [c for c in tree.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == name]
        if not hits or any(_pytest_style(c, local) for c in hits):
            return None
        found = [f"{test_path.stem}.{c.name}.{name}" for c in hits]
        ids += found
    return ids


def _parse(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_MARK):
            try:
                return json.loads(line[len(_MARK):])
            except ValueError:
                return None
    return None


def run_in_dir(directory: Path, test_file: str = "test_mylib.py", timeout: float = 20,
               only: list[str] | None = None) -> TestReport:
    """Run the tests in `directory` itself (callers pass a disposable directory)."""
    module = Path(test_file).stem
    targets = [module]
    if only:
        targets = _test_ids(directory / test_file, only) or [module]
    res = run_python(["-c", _RUNNER, *targets], cwd=directory, timeout=timeout)
    output = res.stderr + res.stdout
    if res.timed_out:
        return TestReport(False, ["<timeout>"], output)
    verdict = _parse(res.stdout)
    if verdict is None or res.returncode != 0:
        return TestReport(False, ["<unknown>"], output)
    if verdict["ok"]:
        return TestReport(True, [], output)
    return TestReport(False, list(verdict["failed"]) or ["<unknown>"], output)


def run_tests(workspace: Path, test_file: str = "test_mylib.py", timeout: float = 20,
              only: list[str] | None = None) -> TestReport:
    tmp = Path(tempfile.mkdtemp(prefix="coding_run_"))
    try:
        shutil.copytree(workspace, tmp / "ws", ignore=shutil.ignore_patterns("__pycache__"))
        return run_in_dir(tmp / "ws", test_file, timeout, only)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

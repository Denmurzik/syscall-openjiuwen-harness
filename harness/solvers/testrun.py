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
_STUB = None


def _wants_stub():
    # the stand-in serves only a test file that imports pytest outside a try block: a file without pytest, or one
    # that guards the import (`except ImportError: pytest = None`), runs exactly as before the stand-in existed
    import ast
    try:
        with open(sys.argv[1].split(".")[0] + ".py", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except Exception:
        return False
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in node.body:
                guarded.update(id(n) for n in ast.walk(sub))
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "pytest" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "pytest":
            return True
    return False


_need_stub = False
if _wants_stub():
    try:
        import pytest  # the real one, when the image has it
    except Exception:
        _need_stub = True
if _need_stub:
    # The image has no pytest: a test file that says `import pytest` (for pytest.raises, approx, parametrize, a
    # fixture) could not even load, so no repair of the task could ever turn green. A stand-in with the parts that
    # plain tests use. Marks only record themselves; `_collect` below reads them. Whatever the stand-in cannot do as
    # pytest does (a conftest.py, fixture params, indirect parametrize, an unknown fixture, xunit module hooks, marks
    # on a TestCase) makes the file red, never green.
    import inspect as _inspect, os as _os, platform as _platform, re as _re, types as _types
    import warnings as _warnings
    _pt = _types.ModuleType("pytest")
    _pt._fixtures = {}    # name -> (function, scope, autouse)
    _pt._bad = []         # reasons the file cannot run here as under pytest
    _pt._wide = ({}, [])  # module-scoped fixture values and their open generators
    if _os.path.exists("conftest.py"):
        _pt._bad.append("conftest.py")

    class _Failed(BaseException):
        # as pytest's: `except Exception` in a test does not swallow pytest.fail or DID NOT RAISE
        pass

    class _Unsupported(Exception):
        pass

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match, self.value, self.type = exc, match, None, None

        def __enter__(self):
            return self

        def __exit__(self, tp, val, tb):
            if tp is None:
                raise _Failed("DID NOT RAISE " + repr(self.exc))
            if not issubclass(tp, self.exc):
                return False
            if self.match is not None and not _re.search(self.match, str(val)):
                raise _Failed("the exception message does not match " + repr(self.match))
            self.value, self.type = val, tp
            return True

    def _raises(exc, *args, match=None, **kw):
        if args:
            if match is not None:
                kw["match"] = match   # pytest passes it on to the callable
            with _Raises(exc) as r:
                args[0](*args[1:], **kw)
            return r
        return _Raises(exc, match)

    class _Warns:
        def __init__(self, cls=Warning, match=None):
            self.cls, self.match, self.list = cls, match, []

        def __enter__(self):
            self._cm = _warnings.catch_warnings(record=True)
            self.list = self._cm.__enter__()
            _warnings.simplefilter("always")
            return self

        def __exit__(self, tp, val, tb):
            self._cm.__exit__(tp, val, tb)
            if tp is not None:
                return False
            if not any(issubclass(w.category, self.cls) and (self.match is None or _re.search(self.match, str(w.message)))
                       for w in self.list):
                raise _Failed("DID NOT WARN " + repr(self.cls))
            return False

    def _warns(cls=Warning, *args, match=None, **kw):
        if args:
            with _Warns(cls, match):
                return args[0](*args[1:], **kw)
        return _Warns(cls, match)

    class _Approx:
        def __init__(self, expected, rel=None, abs=None, nan_ok=False):
            self.expected, self.rel, self.abs = expected, rel, abs

        def _close(self, a, e):
            if isinstance(e, (list, tuple)):
                return isinstance(a, (list, tuple)) and len(a) == len(e) and all(self._close(x, y) for x, y in zip(a, e))
            if isinstance(e, dict):
                return isinstance(a, dict) and a.keys() == e.keys() and all(self._close(a[k], e[k]) for k in e)
            try:
                if a == e:
                    return True
                # as pytest: abs alone is the whole tolerance; else the larger of rel * |expected| and abs
                if self.rel is None and self.abs is not None:
                    tol = self.abs
                else:
                    tol = max((1e-6 if self.rel is None else self.rel) * abs(e), 1e-12 if self.abs is None else self.abs)
                return bool(abs(a - e) <= tol)
            except (TypeError, ValueError):
                return False

        def __eq__(self, other):
            return self._close(other, self.expected)

        def __ne__(self, other):
            return not self == other

        def __repr__(self):
            return "approx(" + repr(self.expected) + ")"

    class _Param:
        def __init__(self, *values, marks=(), id=None):
            self.values = values
            self.marks = list(marks) if isinstance(marks, (list, tuple)) else [marks]

    class _MarkDeco:
        # pytest.mark.<name>(...): applied to a function or a class it records itself there (`_harness_marks`)
        def __init__(self, name, args=(), kwargs=None):
            self.name, self.args, self.kwargs = name, tuple(args), dict(kwargs or {})

        def __call__(self, *a, **k):
            if len(a) == 1 and not k and (_inspect.isclass(a[0]) or isinstance(a[0], (staticmethod, classmethod))
                                          or (callable(a[0]) and getattr(a[0], "__name__", "") != "<lambda>")):
                target = a[0]
                obj = target.__func__ if isinstance(target, (staticmethod, classmethod)) else target
                if "_harness_marks" not in getattr(obj, "__dict__", {}):
                    setattr(obj, "_harness_marks", [])
                obj._harness_marks.append(self)
                return target
            return _MarkDeco(self.name, self.args + a, {**self.kwargs, **k})

    class _Mark:
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return _MarkDeco(name)

    def _fixture(*a, **k):
        def deco(f):
            scope = k.get("scope", "function")
            if k.get("params") is not None or not isinstance(scope, str) or _inspect.iscoroutinefunction(f) \
                    or _inspect.isasyncgenfunction(f):
                _pt._bad.append("fixture " + f.__name__)
            _pt._fixtures[k.get("name") or f.__name__] = (f, scope, bool(k.get("autouse")))
            return f
        return deco(a[0]) if len(a) == 1 and callable(a[0]) and not k else deco

    def _fail(reason="", pytrace=True):
        raise _Failed(reason)

    def _skip(reason="", allow_module_level=False):
        raise unittest.SkipTest(reason)

    def _xfail(reason=""):
        raise unittest.SkipTest("xfail: " + str(reason))

    def _importorskip(name, minversion=None, reason=None):
        import importlib
        try:
            return importlib.import_module(name)
        except ImportError:
            raise unittest.SkipTest(reason or ("no module " + name))

    def _ini_strict():
        # the xfail_strict option of a pytest config file next to the tests
        for fn in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
            try:
                with open(fn, encoding="utf-8") as fh:
                    m = _re.search(r"(?m)^\s*xfail_strict\s*=\s*(\S+)", fh.read())
            except Exception:
                continue
            if m:
                return m.group(1).strip("\"'").lower() in ("true", "1", "yes", "on")
        return False

    def _need(f, skip=0):
        # the parameters pytest fills: without a default, positional or keyword-only, after `skip` (self, cls)
        ps = list(_inspect.signature(f).parameters.values())[skip:]
        return [p.name for p in ps if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                                                                            p.KEYWORD_ONLY)]

    _WIDE = ("module", "session", "package")
    _KINDS = ("parametrize", "skip", "skipif", "xfail", "usefixtures")

    def _check_fixture(name, chain=()):
        # every fixture is known, without a cycle, a wide one needs only wide ones: else the file is red
        if name not in _pt._fixtures or name in chain:
            raise _Unsupported("fixture " + name)
        f, scope, _ = _pt._fixtures[name]
        for n in _need(f):
            _check_fixture(n, chain + (name,))
            if scope in _WIDE and _pt._fixtures[n][1] not in _WIDE:
                raise _Unsupported("scope of " + n)

    def _get(name, cache, gens):
        f, scope, _ = _pt._fixtures[name]
        store, opened = _pt._wide if scope in _WIDE else (cache, gens)   # a class scope is one test's here
        if name in store:
            return store[name]
        v = f(**{n: _get(n, cache, gens) for n in _need(f)})
        if _inspect.isgenerator(v):
            opened.append(v)
            try:
                v = next(v)
            except StopIteration:
                raise _Failed("fixture " + name + " did not yield")
        store[name] = v
        return v

    def _close(gens):
        # teardown of yield fixtures, last opened first; the first error is the test's
        err = None
        while gens:
            g = gens.pop()
            try:
                next(g)
            except StopIteration:
                continue
            except BaseException as e:
                err = err or e
                continue
            err = err or _Failed("fixture yielded twice")
        return err

    def _finish():
        _pt._wide[0].clear()
        return _close(_pt._wide[1]) is None

    def _marks(obj):
        return list(getattr(obj, "__dict__", {}).get("_harness_marks", []))

    def _attr_marks(holder):
        # `pytestmark = ...` of a module or a class
        pm = getattr(holder, "__dict__", {}).get("pytestmark", [])
        pm = list(pm) if isinstance(pm, (list, tuple)) else [pm]
        if not all(isinstance(m, _MarkDeco) for m in pm):
            raise _Unsupported("pytestmark")
        return pm

    def _class_marks(c):
        out = []
        for k in c.__mro__:
            out += _marks(k) + _attr_marks(k)
        return out

    def _cond(c, glb):
        if isinstance(c, str):
            try:
                return bool(eval(c, {**glb, "os": _os, "sys": sys, "platform": _platform}))
            except Exception:
                raise _Unsupported("condition " + c)
        return bool(c)

    def _conds(mk):
        return mk.args if mk.args else ((mk.kwargs["condition"],) if "condition" in mk.kwargs else ())

    def _verdict(marks, glb):
        # None, "skip" or the xfail mark that applies (pytest: skip marks first, then the first xfail that holds)
        for mk in marks:
            if mk.name == "skip":
                return "skip"
            if mk.name == "skipif":
                cs = _conds(mk)
                if not cs or any(_cond(c, glb) for c in cs):
                    return "skip"
        for mk in marks:
            if mk.name == "xfail":
                cs = _conds(mk)
                if not cs or any(_cond(c, glb) for c in cs):
                    return mk
        return None

    def _cases(marks):
        # [(arguments, marks of the case)]; [] for an empty parameter set
        cases = [({}, [])]
        for mk in marks:
            if mk.name != "parametrize":
                continue
            if mk.kwargs.get("indirect"):
                raise _Unsupported("indirect")
            a = list(mk.args) + [None, None]
            names = mk.kwargs.get("argnames", a[0])
            values = mk.kwargs.get("argvalues", a[1])
            if values is None:
                raise _Unsupported("parametrize")
            names = [n.strip() for n in names.split(",") if n.strip()] if isinstance(names, str) else list(names)
            if "_harness_values" not in mk.__dict__:
                mk._harness_values = list(values)
            new = []
            for v in mk._harness_values:
                vals, vm = (v.values, v.marks) if isinstance(v, _Param) else (
                    (tuple(v) if len(names) > 1 else (v,)), [])
                if len(vals) != len(names):
                    raise _Unsupported("parametrize arity")
                new.append((dict(zip(names, vals)), vm))
            if any(set(names) & set(c) for c, _ in cases):
                raise _Unsupported("parametrize twice")
            cases = [({**c, **d}, cm + dm) for c, cm in cases for d, dm in new]
        return cases

    def _case(name, call, need, args, marks, glb, strict):
        verdict = _verdict(marks, glb)
        use = [n for mk in marks if mk.name == "usefixtures" for n in mk.args]
        auto = [n for n, (f, _, a) in _pt._fixtures.items() if a]
        fill = [n for n in need if n not in args]
        for n in auto + use + fill:
            _check_fixture(n)

        def body():
            cache, gens, exc = {}, [], None
            try:
                for n in auto + use:
                    _get(n, cache, gens)
                call({**args, **{n: _get(n, cache, gens) for n in fill}})
            except BaseException as e:
                exc = e
            err = _close(gens)
            if exc is not None:
                raise exc
            if err is not None:
                raise err

        def run():
            if verdict == "skip":
                raise unittest.SkipTest("skip mark")
            if verdict is None:
                return body()
            if not verdict.kwargs.get("run", True):
                raise unittest.SkipTest("xfail, not run")
            try:
                body()
            except unittest.SkipTest:
                raise
            except BaseException as e:
                raises = verdict.kwargs.get("raises")
                if raises is not None and not isinstance(e, raises):
                    raise
                raise unittest.SkipTest("xfail")
            if verdict.kwargs.get("strict", strict):
                raise _Failed("XPASS(strict)")
        run.__name__ = name
        return unittest.FunctionTestCase(run)

    def _flat(suite):
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                yield from _flat(t)
            else:
                yield t

    def _collect(suite):
        # (suite, no_tests): the tests of the file as pytest would run them; an empty suite (red) where the
        # stand-in cannot follow pytest
        mod = sys.modules.get(sys.argv[1].split(".")[0])
        red = (unittest.TestSuite(), False)
        if mod is None:
            return suite, False
        try:
            mod_marks = _attr_marks(mod)
            fixture_fns = {id(v[0]) for v in _pt._fixtures.values()}
            fns = [f for n, f in vars(mod).items() if n.startswith("test") and _inspect.isfunction(f)
                   and f.__module__ == mod.__name__ and id(f) not in fixture_fns]
            classes = [c for n, c in vars(mod).items() if n.startswith("Test") and _inspect.isclass(c)
                       and c.__module__ == mod.__name__ and not issubclass(c, unittest.TestCase)
                       and getattr(c, "__test__", True) is not False]
            if suite.countTestCases() > 0:
                # unittest classes: unittest runs them; pytest marks, autouse fixtures or pytest-style tests beside
                # them are not followed here
                if _pt._bad or fns or classes or any(a for _, _, a in _pt._fixtures.values()) or any(
                        mk.name in _KINDS for mk in mod_marks):
                    return red
                for t in _flat(suite):
                    meth = getattr(type(t), getattr(t, "_testMethodName", ""), None)
                    if any(mk.name in _KINDS for mk in _class_marks(type(t)) + _marks(meth)):
                        return red
                return suite, False
            if len(sys.argv) != 2 or "." in sys.argv[1]:
                return suite, False
            items = []
            for f in fns:
                items.append((f.__name__, f, None, 0))
            for c in classes:
                for m in dir(c):
                    if not m.startswith("test"):
                        continue
                    raw = _inspect.getattr_static(c, m)
                    if isinstance(raw, staticmethod):
                        items.append((m, raw.__func__, c, "static"))
                    elif isinstance(raw, classmethod):
                        items.append((m, raw.__func__, c, "class"))
                    elif _inspect.isfunction(raw):
                        items.append((m, raw, c, "self"))
            if not items:
                return unittest.TestSuite(), True
            if _pt._bad or any(hasattr(mod, h) for h in ("setup_module", "teardown_module", "setup_function",
                                                         "teardown_function", "setup", "teardown")):
                return red
            strict, glb, tests = _ini_strict(), vars(mod), []
            for name, f, c, kind in items:
                if _inspect.iscoroutinefunction(f) or _inspect.isgeneratorfunction(f) or _inspect.isasyncgenfunction(f):
                    return red
                if c is not None and any(hasattr(c, h) for h in ("setup_class", "teardown_class", "setup",
                                                                 "teardown")):
                    return red
                marks = _marks(f) + (_class_marks(c) if c is not None else []) + mod_marks
                need = _need(f, 1 if kind in ("self", "class") else 0)
                cases = _cases(marks)
                if not cases:   # an empty parameter set: pytest skips the test
                    tests.append(_case(name, lambda kw: None, [], {}, [_MarkDeco("skip")], glb, strict))
                    continue
                for args, cmarks in cases:
                    tests.append(_case(name, _caller(f, c, kind, name), need, args, marks + cmarks, glb, strict))
            return unittest.TestSuite(tests), False
        except _Unsupported:
            return red

    def _caller(f, c, kind, m):
        if c is None or kind == "static":
            return lambda kw: f(**kw)
        if kind == "class":
            return lambda kw: f(c, **kw)

        def call(kw):
            o = c()   # a class whose constructor needs arguments errors here: red
            hook = getattr(o, "setup_method", None)
            if callable(hook):
                hook(getattr(o, m))
            try:
                f(o, **kw)
            finally:
                hook = getattr(o, "teardown_method", None)
                if callable(hook):
                    hook(getattr(o, m))
        return call

    _pt.raises, _pt.warns, _pt.approx, _pt.param, _pt.mark, _pt.fixture = _raises, _warns, _Approx, _Param, _Mark(), _fixture
    _pt.fail, _pt.skip, _pt.xfail, _pt.importorskip = _fail, _skip, _xfail, _importorskip
    _pt.fail.Exception, _pt.skip.Exception = _Failed, unittest.SkipTest
    _pt._collect, _pt._finish = _collect, _finish
    sys.modules["pytest"] = _pt
    _STUB = _pt
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
script_ok = None
_no_tests = False
if _STUB is not None:
    suite, _no_tests = _STUB._collect(suite)   # no test in the file: the script rules below decide
if _no_tests or (_STUB is None and suite.countTestCases() == 0 and len(sys.argv) == 2 and "." not in sys.argv[1]):
    # no TestCase in the file: plain test functions (pytest style), or a script of bare asserts
    import ast, importlib, inspect
    mod = importlib.import_module(sys.argv[1])

    def _free(f, skip=0):
        return not any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                       for p in list(inspect.signature(f).parameters.values())[skip:])

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
                getattr(o, m)()
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
    if (fns or methods) and all(_free(f) and _sync(f) for f in fns) and all(
            _free(getattr(c, m), 1) and _sync(getattr(c, m)) for c, m in methods):
        suite = unittest.TestSuite([unittest.FunctionTestCase(f) for f in fns]
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
ok = res.wasSuccessful() and res.testsRun > 0
if _STUB is not None and not _STUB._finish():   # module-scoped yield fixtures close after the last test
    ok = False
    names.append("<teardown>")
print("%s" + json.dumps({"ok": ok, "failed": names}), flush=True)
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

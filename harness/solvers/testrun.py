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
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
script_ok = None
if suite.countTestCases() == 0 and len(sys.argv) == 2 and "." not in sys.argv[1]:
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
print("%s" + json.dumps({"ok": res.wasSuccessful() and res.testsRun > 0, "failed": names}), flush=True)
''' % (_MARK, _MARK)


@dataclass(frozen=True)
class TestReport:
    __test__ = False  # not a pytest class

    ok: bool
    failed: list[str] = field(default_factory=list)
    output: str = ""


def _is_testcase_base(base: ast.expr) -> bool:
    """`TestCase`, `unittest.TestCase`, `MyTestCase`: a base whose name ends in TestCase."""
    name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
    return name.endswith("TestCase")


def _test_ids(test_path: Path, names: list[str]) -> list[str] | None:
    """Resolve method names to unittest ids; None when any name is unknown or sits in a class with no `...TestCase`
    base (a pytest-style class, `class TestX:` or `class TestX(object):`, which unittest cannot load by id: the
    whole file is run instead)."""
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    ids = []
    for name in names:
        hits = [c for c in tree.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == name]
        if not hits or not all(any(_is_testcase_base(b) for b in c.bases) for c in hits):
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

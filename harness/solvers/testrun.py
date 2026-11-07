"""Run a task's unittest file in a throw-away copy of the workspace."""
import ast
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from harness.proc import run_python

_FAIL_LINE = re.compile(r"^(?:FAIL|ERROR): (\w+) \(", re.MULTILINE)


@dataclass(frozen=True)
class TestReport:
    __test__ = False  # not a pytest class

    ok: bool
    failed: list[str] = field(default_factory=list)
    output: str = ""


def _test_ids(test_path: Path, names: list[str]) -> list[str] | None:
    """Resolve method names to unittest ids; None when any name is unknown."""
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    ids = []
    for name in names:
        found = [f"{test_path.stem}.{c.name}.{f.name}"
                 for c in tree.body if isinstance(c, ast.ClassDef)
                 for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and f.name == name]
        if not found:
            return None
        ids += found
    return ids


def run_in_dir(directory: Path, test_file: str = "test_mylib.py", timeout: float = 20,
               only: list[str] | None = None) -> TestReport:
    """Run the tests in `directory` itself (callers pass a disposable directory)."""
    module = Path(test_file).stem
    targets = [module]
    if only:
        targets = _test_ids(directory / test_file, only) or [module]
    res = run_python(["-m", "unittest", *targets], cwd=directory, timeout=timeout)
    output = (res.stderr + res.stdout)
    if res.timed_out:
        return TestReport(False, ["<timeout>"], output)
    if res.returncode == 0:
        return TestReport(True, [], output)
    failed = list(dict.fromkeys(_FAIL_LINE.findall(output)))
    return TestReport(False, failed or ["<unknown>"], output)


def run_tests(workspace: Path, test_file: str = "test_mylib.py", timeout: float = 20,
              only: list[str] | None = None) -> TestReport:
    tmp = Path(tempfile.mkdtemp(prefix="coding_run_"))
    try:
        shutil.copytree(workspace, tmp / "ws", ignore=shutil.ignore_patterns("__pycache__"))
        return run_in_dir(tmp / "ws", test_file, timeout, only)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

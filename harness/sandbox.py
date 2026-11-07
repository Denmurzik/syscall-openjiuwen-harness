"""Run model-written programs against a throw-away copy of a task workspace."""
import shutil
import tempfile
from pathlib import Path

from harness.proc import ProcResult, run_python


def run_program(code: str, workspace: Path, timeout: float = 20) -> ProcResult:
    """Run `code` with cwd = a temporary copy of `workspace`; the real one is never touched.

    The program file lives outside the copy, so it cannot be mistaken for task data.
    """
    tmp = Path(tempfile.mkdtemp(prefix="prog_run_"))
    try:
        ws = tmp / "ws"
        shutil.copytree(workspace, ws, ignore=shutil.ignore_patterns("__pycache__"))
        prog = tmp / "prog.py"
        prog.write_text(code, encoding="utf-8")
        return run_python([str(prog)], cwd=ws, timeout=timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

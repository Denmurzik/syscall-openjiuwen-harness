"""Run model-written programs against a throw-away copy of a task workspace.

Not a security boundary: isolation comes from the container. This only protects the real workspace from accidents."""
import os
import shutil
import tempfile
from pathlib import Path

from harness import config
from harness.proc import ProcResult, run_python


# default of HARNESS_BIG_WS. Expert 11, patch e: every run of a model's program copies the whole workspace into the
# temporary directory, which under the jury's flags is a tmpfs of 2 GB counted against the container's memory; three
# programs at once over a workspace of several hundred megabytes fill it, and the copy fails. On: a workspace above
# BIG_WS_BYTES (or a copy that fails) is not copied: the run's directory holds the same tree of directories with a
# symbolic link per file. Off: f296ee6
BIG_WS = True
BIG_WS_BYTES = 100 * 1024 * 1024


def _big(workspace: Path) -> bool:
    if not config._env_bool("HARNESS_BIG_WS", BIG_WS):
        return False
    total = 0
    try:
        for root, _dirs, files in os.walk(workspace):
            for f in files:
                total += os.lstat(os.path.join(root, f)).st_size
                if total > BIG_WS_BYTES:
                    return True
    except OSError:
        return False
    return False


def _link_tree(workspace: Path, ws: Path) -> bool:
    """The directories of `workspace` under `ws`, a symbolic link for every file. False (and nothing left) when a
    link cannot be made."""
    try:
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            dst = ws / os.path.relpath(root, workspace)
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                os.symlink(os.path.abspath(os.path.join(root, f)), dst / f)
        return True
    except OSError:
        shutil.rmtree(ws, ignore_errors=True)
        return False


def run_program(code: str, workspace: Path, timeout: float = 20) -> ProcResult:
    """Run `code` with cwd = a temporary copy of `workspace`; the real one is never touched.

    The program file lives outside the copy, so it cannot be mistaken for task data.
    """
    tmp = Path(tempfile.mkdtemp(prefix="prog_run_"))
    try:
        ws = tmp / "ws"
        if Path(workspace).is_dir():
            if not _big(Path(workspace)) or not _link_tree(Path(workspace), ws):
                try:
                    shutil.copytree(workspace, ws, ignore=shutil.ignore_patterns("__pycache__"))
                except OSError:   # the temporary directory is full (a tmpfs of 2 GB under the jury's flags)
                    shutil.rmtree(ws, ignore_errors=True)
                    if not (config._env_bool("HARNESS_BIG_WS", BIG_WS) and _link_tree(Path(workspace), ws)):
                        raise
        else:  # tasks without files (e.g. math) have no workspace directory
            ws.mkdir()
        prog = tmp / "prog.py"
        prog.write_text(code, encoding="utf-8")
        return run_python([str(prog)], cwd=ws, timeout=timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

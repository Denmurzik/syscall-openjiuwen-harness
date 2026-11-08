import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_OUT = 8 * 1024
_RLIMIT_AS = 2 * 1024 ** 3


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _cut(b: bytes | str | None) -> str:
    if b is None:
        return ""
    if isinstance(b, bytes):
        b = b.decode("utf-8", errors="replace")
    if len(b) <= MAX_OUT:
        return b
    mark = "\n...[truncated]...\n"
    return b[:2048] + mark + b[-(MAX_OUT - 2048 - len(mark)):]


def _posix_descendants(pid: int) -> list[int]:
    """Pids of all descendants of `pid` (via /proc); best effort, [] when /proc is unavailable."""
    children: dict[int, list[int]] = {}
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    ppid = int(f.read().rsplit(b")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(name))
    except OSError:
        return []
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def kill_descendants(pid: int) -> None:
    """Kill everything `pid` started (not `pid` itself). A solver subprocess runs in its own session
    (posix) or is merely a child (Windows), so killing only the phase-1 child would orphan it, and an
    orphaned endless loop burns CPU for the rest of the run."""
    try:
        if os.name == "posix":
            import signal
            for c in _posix_descendants(pid):
                try:
                    os.kill(c, signal.SIGKILL)
                except OSError:
                    pass
        else:
            # taskkill /T walks the parent links of live processes, so run it before the parent dies
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, check=False, timeout=15)
    except Exception:
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, check=False, timeout=15)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def make_limits(timeout: float):
    """preexec_fn that sets RLIMIT_AS / RLIMIT_CPU; never raises (a failing preexec breaks every Popen)."""
    import resource

    def limits():
        for res, wanted in ((resource.RLIMIT_AS, _RLIMIT_AS),
                            (resource.RLIMIT_CPU, int(timeout) * 4 + 5)):
            try:
                _, hard = resource.getrlimit(res)
                if hard != resource.RLIM_INFINITY:
                    wanted = min(wanted, hard)
                resource.setrlimit(res, (wanted, wanted))
            except Exception:
                pass
    return limits


def run_python(args: list[str], cwd: Path, timeout: float, stdin: str | None = None,
               env: dict | None = None) -> ProcResult:
    kwargs: dict = {}
    if env is not None:
        kwargs["env"] = {**os.environ, **env}
    if os.name == "posix":
        import resource

        limits = make_limits(timeout)

        kwargs.update(start_new_session=True, preexec_fn=limits)
    proc = subprocess.Popen(
        [sys.executable, "-B", *args], cwd=cwd,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    try:
        out, err = proc.communicate(stdin.encode("utf-8") if stdin is not None else None, timeout=timeout)
        return ProcResult(proc.returncode, _cut(out), _cut(err), False)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as e:
            out, err = e.stdout, e.stderr
            for pipe in (proc.stdout, proc.stderr, proc.stdin):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
        rc = proc.returncode if proc.returncode not in (0, None) else -9
        return ProcResult(rc, _cut(out), _cut(err), True)

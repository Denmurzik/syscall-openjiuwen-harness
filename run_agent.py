"""Entry point: python run_agent.py [--tasks-dir DIR | DIR] [--out FILE] (always exits 0)."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_BUDGET = 100000.0


def _budget() -> float:
    try:
        v = float(os.environ.get("BUDGET", ""))
        if 0 < v < float("inf"):
            return v
    except ValueError:
        pass
    return DEFAULT_BUDGET


def _probe_tasks_dir(tasks_dir: Path) -> None:
    """Report (one stderr line) if answers cannot be created in the tasks dir; never raises."""
    try:
        fd, name = tempfile.mkstemp(prefix=".probe_", dir=tasks_dir)
        os.close(fd)
        os.unlink(name)
    except Exception:
        try:
            st = os.stat(tasks_dir)
            uid = os.getuid() if hasattr(os, "getuid") else "?"
            print(f"TASKS DIR NOT WRITABLE: uid={uid} owner={st.st_uid} mode={oct(st.st_mode & 0o777)}",
                  file=sys.stderr, flush=True)
        except Exception:
            print("TASKS DIR NOT WRITABLE: uid=? owner=? mode=?", file=sys.stderr, flush=True)


def _resolve_paths(argv=None, env=None) -> tuple[Path, str, str | None]:
    """(tasks_dir, source, out): --tasks-dir > positional > env TASKS > /data/tasks."""
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path)
    ap.add_argument("--out", "--output", dest="out")
    ap.add_argument("tasks", nargs="?", type=Path)
    args, _ = ap.parse_known_args(argv)
    if args.tasks_dir:
        tasks, src = args.tasks_dir, "--tasks-dir"
    elif args.tasks:
        tasks, src = args.tasks, "positional"
    elif env.get("TASKS"):
        tasks, src = Path(env["TASKS"]), "env TASKS"
    else:
        tasks, src = Path("/data/tasks"), "default /data/tasks"
    return tasks, src, args.out or env.get("OUT") or None


def main(argv=None) -> None:
    summary: dict = {}
    try:
        tasks_dir, src, out = _resolve_paths(argv)
        tasks_dir = tasks_dir.resolve()  # before any chdir
        print(f"tasks dir: {tasks_dir} (from {src}); answers file: {out or 'none'}",
              file=sys.stderr, flush=True)
        if out:
            try:
                from harness.tasks import set_answer_sink
                set_answer_sink(Path(out) if out.endswith(("/", "\\")) else Path(out).resolve())
            except Exception as e:
                print(f"answers file unavailable: {e!r}", file=sys.stderr)
        log_dir = Path(os.environ.get("HARNESS_LOG_DIR") or Path(tempfile.gettempdir()) / "harness")
        os.environ["HARNESS_LOG_DIR"] = str(log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        from harness import config, scheduler
        _probe_tasks_dir(tasks_dir)
        budget = _budget()
        base, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
        client = None
        if base and key:
            try:
                os.chdir(log_dir)  # before openjiuwen is imported
            except OSError as e:
                print(f"cannot chdir to {log_dir}: {e!r}", file=sys.stderr)
            try:
                from harness.budget import TokenLedger
                from harness.llm import LLMClient
                ledger = TokenLedger(budget, budget * config.RESERVE_FRACTION)
                client = LLMClient(base, key, ledger)
            except Exception as e:
                print(f"LLM client unavailable: {e!r}", file=sys.stderr)
        summary = scheduler.run(tasks_dir, budget, client)
    except BaseException as e:  # never crash
        print(f"run failed: {e!r}", file=sys.stderr)
    try:
        print(f"solved={summary.get('solved', 0)} infeasible={summary.get('infeasible', 0)} "
              f"unparsed={summary.get('unparsed', 0)} spent={summary.get('spent', 0):.1f}", flush=True)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()

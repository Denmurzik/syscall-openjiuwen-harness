"""Entry point: python run_agent.py --tasks-dir DIR (always exits 0)."""
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


def main(argv=None) -> None:
    summary: dict = {}
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--tasks-dir", required=True, type=Path)
        args, _ = ap.parse_known_args(argv)
        tasks_dir = args.tasks_dir.resolve()  # before any chdir
        log_dir = Path(os.environ.get("HARNESS_LOG_DIR") or Path(tempfile.gettempdir()) / "harness")
        os.environ["HARNESS_LOG_DIR"] = str(log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        from harness import config, scheduler
        budget = _budget()
        base, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
        client = None
        if base and key:
            try:
                os.chdir(log_dir)  # before openjiuwen is imported
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

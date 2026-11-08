"""Phase-1 bench: generate a task set, run ONLY the deterministic solvers (no LLM, no Docker),
score it and write bench.json + bench.md (accuracy per type/difficulty, timing, failures by pattern).

    python -m tools.bench --gen-n 2000 --seed 1 --difficulty mixed --out runs/bench1 [--workers 8]
"""
import argparse
import faulthandler
import json
import os
import threading
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from tools import evaluate, gen_tasks

TYPES = gen_tasks.TYPES


def _stall_watchdog(progress: list, stop: threading.Event, stall_s: float, dump_path: Path) -> None:
    """Diagnostics: when no task finished for `stall_s` seconds, dump parent thread tracebacks
    and the process list to `dump_path`; repeats at most every `stall_s`."""
    import subprocess
    NL = chr(10)
    last_dump = 0.0
    while not stop.wait(5.0):
        now = time.monotonic()
        if now - progress[0] < stall_s or now - last_dump < stall_s:
            continue
        last_dump = now
        try:
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(f"=== no progress for {now - progress[0]:.0f}s, {progress[1]} done ===" + NL)
                try:
                    cmd = (["tasklist", "/FI", "IMAGENAME eq python.exe"] if os.name == "nt"
                           else ["ps", "-eo", "pid,ppid,stat,etime,time,cmd"])
                    f.write(subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout)
                except Exception as e:
                    f.write(f"process list failed: {e!r}" + NL)
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
        except Exception:
            pass


def run_phase1(tasks_dir: Path, workers: int, deadline_s: float = 24 * 3600) -> dict:
    """Run the harness phase-1 entry on `tasks_dir`; write answers like the scheduler does.
    Returns {task_id: (SolveResult, elapsed_s)}."""
    from harness import scheduler
    from harness.result import SolveResult
    from harness.tasks import load_tasks, write_answer

    tasks = load_tasks(tasks_dir)
    results: dict = {}
    progress = [time.monotonic(), 0]
    t_start = time.monotonic()
    stop = threading.Event()
    stall_s = float(os.environ.get("BENCH_STALL_S", "120"))
    threading.Thread(target=_stall_watchdog, daemon=True,
                     args=(progress, stop, stall_s, Path(os.environ.get("BENCH_STALL_DUMP", "bench_stall.txt")))).start()

    def on_result(task, res, elapsed):
        progress[0] = time.monotonic()
        progress[1] += 1
        if progress[1] % 250 == 0:
            print(f"phase 1: {progress[1]}/{len(tasks)} tasks done, {time.monotonic() - t_start:.0f}s", file=sys.stderr, flush=True)
        if res.status == "solved" and res.answer is None:
            res = SolveResult("unparsed", note="solved without answer")
        elif res.status == "solved":
            try:
                write_answer(task, res.answer)
            except Exception as e:
                res = SolveResult("unparsed", note=f"write failed: {e!r}")
        results[task.task_id] = (res, elapsed)

    try:
        scheduler._phase1(tasks, workers, time.monotonic(), deadline_s, on_result)
    finally:
        stop.set()
    return results


def _pattern(note: str) -> str:
    note = re.sub(r"'[^']*'|\"[^\"]*\"", "'_'", note or "")
    note = re.sub(r"\d+", "N", note)
    return note[:90]


def _pct(vals, q):
    if not vals:
        return 0.0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def score(set_dir: Path, results: dict) -> dict:
    """Classify every task: correct / wrong / unparsed / (false or right) infeasible."""
    set_dir = Path(set_dir)
    diffs = {}
    dj = set_dir / "difficulty.json"
    if dj.is_file():
        diffs = json.loads(dj.read_text(encoding="utf-8"))
    rows = []
    for tdir in sorted((set_dir / "tasks").iterdir()):
        meta = json.loads((tdir / "task.json").read_text(encoding="utf-8"))
        tid = meta["task_id"]
        res, elapsed = results.get(tid, (None, 0.0))
        status = res.status if res else "unparsed"
        note = res.note if res else "no result"
        feasible = meta["expected"] is not None
        answer = (tdir / "answer.txt").read_text(encoding="utf-8", errors="replace").strip() \
            if (tdir / "answer.txt").is_file() else None
        if status == "unparsed":
            outcome = "unparsed"
        elif status == "infeasible":
            outcome = "infeasible_ok" if not feasible else "wrong"
        elif not feasible:  # an answer on an infeasible task: unscored, neither right nor wrong
            outcome = "infeasible_answered"
        else:  # solved
            outcome = "correct" if evaluate.check(tdir, set_dir / "hidden" / tid) else "wrong"
        rows.append({"task_id": tid, "type": meta["task_type"], "difficulty": diffs.get(tid, "easy"),
                     "feasible": feasible, "reward": meta["reward"], "status": status, "outcome": outcome,
                     "expected": meta["expected"], "got": answer, "note": note, "elapsed": round(elapsed, 3),
                     "instruction": meta["instruction"]})
    return {"rows": rows}


def summarize(rows: list) -> dict:
    def acc(sub):
        feas = [r for r in sub if r["feasible"]]
        inf = [r for r in sub if not r["feasible"]]
        c = Counter(r["outcome"] for r in sub)
        return {
            "total": len(sub), "feasible": len(feas), "infeasible": len(inf),
            "correct": c["correct"], "infeasible_ok": c["infeasible_ok"],
            "infeasible_answered": c["infeasible_answered"], "wrong": c["wrong"],
            "unparsed": c["unparsed"],
            "accuracy_feasible": round(sum(r["outcome"] == "correct" for r in feas) / len(feas), 4) if feas else None,
            "accuracy_infeasible": round(sum(r["outcome"] == "infeasible_ok" for r in inf) / len(inf), 4) if inf else None,
            "accuracy_all": round(sum(r["outcome"] in ("correct", "infeasible_ok") for r in sub) / len(sub), 4) if sub else None,
        }

    def timing(sub):
        t = [r["elapsed"] for r in sub]
        return {"p50": round(_pct(t, 0.5), 3), "p95": round(_pct(t, 0.95), 3), "max": round(max(t, default=0), 3),
                "mean": round(statistics.fmean(t), 3) if t else 0.0}

    out = {"overall": acc(rows), "by_type": {}, "by_difficulty": {}, "by_type_difficulty": {}, "timing": {}}
    for typ in sorted({r["type"] for r in rows}):
        out["by_type"][typ] = acc([r for r in rows if r["type"] == typ])
        out["timing"][typ] = timing([r for r in rows if r["type"] == typ])
    for d in sorted({r["difficulty"] for r in rows}):
        out["by_difficulty"][d] = acc([r for r in rows if r["difficulty"] == d])
    for typ in sorted({r["type"] for r in rows}):
        for d in sorted({r["difficulty"] for r in rows}):
            sub = [r for r in rows if r["type"] == typ and r["difficulty"] == d]
            if sub:
                out["by_type_difficulty"][f"{typ}/{d}"] = acc(sub)
    out["timing"]["all"] = timing(rows)
    return out


def failure_groups(rows: list) -> list:
    groups = defaultdict(list)
    for r in rows:
        if r["outcome"] in ("wrong", "unparsed"):
            kind = r["outcome"] + ("" if r["feasible"] else "/infeasible-task")
            groups[(r["type"], kind, _pattern(r["note"]))].append(r)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def _trunc(s, n):
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def write_reports(out: Path, meta: dict, total_s: float, summary: dict, groups: list, per_group: int = 5) -> None:
    out = Path(out)
    failing = [{k: v for k, v in r.items()} for _, rs in groups for r in rs]
    (out / "bench.json").write_text(json.dumps({
        "meta": meta, "total_seconds": round(total_s, 1), "summary": summary,
        "failure_groups": [{"type": k[0], "kind": k[1], "pattern": k[2], "count": len(rs),
                            "task_ids": [r["task_id"] for r in rs]} for k, rs in groups],
        "failures": failing}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def table(d, label):
        lines = [f"| {label} | total | feasible | correct | wrong | unparsed | infeasible ok | infeasible answered | acc feasible | acc infeasible | acc all |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, a in d.items():
            lines.append(f"| {k} | {a['total']} | {a['feasible']} | {a['correct']} | {a['wrong']} | {a['unparsed']} | "
                         f"{a['infeasible_ok']}/{a['infeasible']} | {a['infeasible_answered']} | {pct(a['accuracy_feasible'])} | "
                         f"{pct(a['accuracy_infeasible'])} | {pct(a['accuracy_all'])} |")
        return lines

    o = summary["overall"]
    md = [f"# Phase-1 bench", "",
          f"n={meta['n']} seed={meta['seed']} difficulty={meta['difficulty']} workers={meta['workers']}; "
          f"total {total_s:.1f}s", "",
          f"Overall: feasible accuracy {pct(o['accuracy_feasible'])} ({o['correct']}/{o['feasible']}), "
          f"infeasible recognised {o['infeasible_ok']}/{o['infeasible']}, wrong {o['wrong']}, infeasible-but-answered {o['infeasible_answered']} (unscored), unparsed {o['unparsed']}.", "",
          "## By type", *table(summary["by_type"], "type"), "",
          "## By difficulty", *table(summary["by_difficulty"], "difficulty"), "",
          "## By type and difficulty", *table(summary["by_type_difficulty"], "type/difficulty"), "",
          "## Timing per task (seconds, solver wall time incl. process start)",
          "| scope | p50 | p95 | max | mean |", "|---|---|---|---|---|"]
    for k, t in summary["timing"].items():
        md.append(f"| {k} | {t['p50']} | {t['p95']} | {t['max']} | {t['mean']} |")
    md += ["", f"## Failures ({sum(len(rs) for _, rs in groups)} tasks in {len(groups)} patterns)", ""]
    for (typ, kind, pat), rs in groups:
        md.append(f"### {typ} / {kind}: {pat or '(no note)'}  (x{len(rs)})")
        for r in rs[:per_group]:
            md.append(f"- `{r['task_id']}` [{r['difficulty']}] expected={_trunc(str(r['expected']), 40)!r} "
                      f"got={_trunc(str(r['got']), 40)!r} status={r['status']} note={_trunc(r['note'], 120)!r}")
            md.append(f"  - {_trunc(r['instruction'], 260)}")
        if len(rs) > per_group:
            md.append(f"- ... {len(rs) - per_group} more (ids in bench.json)")
        md.append("")
    (out / "bench.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def bench(out: Path, n: int, seed: int, difficulty: str, workers: int = 8,
          infeasible: float = 0.2, novel: float = 0.2) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    set_dir = out / "set"
    t0 = time.monotonic()
    gen_tasks.generate(set_dir, n, seed, infeasible, novel, difficulty)
    t_gen = time.monotonic() - t0
    t1 = time.monotonic()
    results = run_phase1(set_dir / "tasks", workers)
    t_run = time.monotonic() - t1
    rows = score(set_dir, results)["rows"]
    summary = summarize(rows)
    groups = failure_groups(rows)
    meta = {"n": n, "seed": seed, "difficulty": difficulty, "workers": workers, "gen_seconds": round(t_gen, 1),
            "phase1_seconds": round(t_run, 1), "infeasible": infeasible, "novel": novel}
    write_reports(out, meta, time.monotonic() - t0, summary, groups)
    return {"summary": summary, "groups": groups, "meta": meta}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--infeasible", type=float, default=0.2)
    ap.add_argument("--novel", type=float, default=0.2)
    a = ap.parse_args(argv)
    r = bench(a.out, a.gen_n, a.seed, a.difficulty, a.workers, a.infeasible, a.novel)
    o = r["summary"]["overall"]
    print(json.dumps({"overall": o, "by_type": {k: v["accuracy_feasible"] for k, v in r["summary"]["by_type"].items()}},
                     indent=1))
    print(f"report: {a.out / 'bench.md'}")


if __name__ == "__main__":
    main()

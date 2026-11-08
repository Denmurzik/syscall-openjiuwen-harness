"""Fuzz the phase-1 parsers (math / data_analysis / search): generate tasks over many seeds, difficulties
and novelty levels, run ONLY the deterministic solvers, and tally correct / wrong / unparsed /
infeasible-answered per type x difficulty x template family. A WRONG phase-1 answer is final and costs
points; `unparsed` only costs tokens.

    python -m tools.fuzz --n 6000 --chunk 400 --seed 1000 --out runs/fuzz [--workers 6]

Each chunk is generated into a temporary directory, solved, scored and deleted, so RAM and disk stay bounded.
Writes <out>/<stamp>/fuzz.json and fuzz.md (every WRONG with its instruction).
"""
import argparse
import json
import random
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from tools import bench, gen_tasks

FUZZ_TYPES = ["math", "data_analysis", "search"]
DIFFS = ("easy", "medium", "hard")
NOVELTY = (0.0, 0.2, 0.5, 1.0)
INFEASIBLE = (0.1, 0.2, 0.3)


def build_chunk(set_dir: Path, n: int, seed: int, difficulty: str, novel: float, infeasible: float) -> None:
    """Like gen_tasks.generate, but only the non-coding types and a fixed difficulty."""
    for sub in ("tasks", "hidden"):
        shutil.rmtree(set_dir / sub, ignore_errors=True)
        (set_dir / sub).mkdir(parents=True)
    rng = random.Random(seed)
    inf_set = set(rng.sample(range(n), round(n * infeasible)))
    nov_set = set(rng.sample(range(n), round(n * novel)))
    diffs = {}
    for i in range(n):
        typ = FUZZ_TYPES[i % len(FUZZ_TYPES)]
        tid = f"{typ}_{i:04d}"
        tdir = set_dir / "tasks" / tid
        ws = tdir / "workspace"
        ws.mkdir(parents=True)
        diffs[tid] = difficulty
        if difficulty == "easy":
            spec = gen_tasks._gen_one(typ, "easy", rng, i in nov_set, i in inf_set, ws)
            reward = rng.randint(1, 5)
        else:
            trng = random.Random(f"{seed}/{i}")
            spec = gen_tasks._gen_one(typ, difficulty, trng, i in nov_set, i in inf_set, ws)
            reward = trng.randint(1, 5)
        spec.pop("_hidden", None)
        task = {"task_id": tid, "task_type": typ, "instruction": spec["instruction"], "reward": reward,
                "expected": spec["expected"], "check": spec["check"]}
        (tdir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")
    (set_dir / "difficulty.json").write_text(json.dumps(diffs) + "\n", encoding="utf-8")


def family(instruction: str) -> str:
    from harness.templates import templatize
    try:
        return templatize(instruction)[0][:140]
    except Exception:
        return "(untemplatizable)"


def run_chunk(base: Path, n: int, seed: int, difficulty: str, novel: float, infeasible: float,
              workers: int) -> list[dict]:
    set_dir = Path(tempfile.mkdtemp(prefix="fuzz_", dir=base))
    try:
        build_chunk(set_dir, n, seed, difficulty, novel, infeasible)
        results = bench.run_phase1(set_dir / "tasks", workers)
        rows = bench.score(set_dir, results)["rows"]
        for r in rows:
            r["family"] = family(r["instruction"])
            r["seed"], r["novel"] = seed, novel
            r["task_id"] = f"{seed}:{r['task_id']}"
        return rows
    finally:
        shutil.rmtree(set_dir, ignore_errors=True)


def tally(rows: list[dict]) -> dict:
    def count(sub):
        c = Counter(r["outcome"] for r in sub)
        feas = sum(r["feasible"] for r in sub)
        return {"total": len(sub), "feasible": feas, "correct": c["correct"], "wrong": c["wrong"],
                "unparsed": c["unparsed"], "infeasible_ok": c["infeasible_ok"],
                "infeasible_answered": c["infeasible_answered"]}

    out = {"overall": count(rows), "by_type": {}, "by_type_difficulty": {}, "by_family": {}}
    for t in FUZZ_TYPES:
        sub = [r for r in rows if r["type"] == t]
        out["by_type"][t] = count(sub)
        for d in DIFFS:
            s = [r for r in sub if r["difficulty"] == d]
            if s:
                out["by_type_difficulty"][f"{t}/{d}"] = count(s)
    fam = defaultdict(list)
    for r in rows:
        fam[(r["type"], r["family"])].append(r)
    out["by_family"] = {f"{t} | {f}": count(s) for (t, f), s in
                        sorted(fam.items(), key=lambda kv: -len(kv[1]))}
    return out


def write_reports(out: Path, meta: dict, rows: list[dict]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = tally(rows)
    bad = [r for r in rows if r["outcome"] in ("wrong", "infeasible_answered")]
    (out / "fuzz.json").write_text(json.dumps({
        "meta": meta, "summary": summary,
        "wrong": [r for r in rows if r["outcome"] == "wrong"],
        "infeasible_answered": [r for r in rows if r["outcome"] == "infeasible_answered"]},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = ["# Phase-1 fuzz", "", json.dumps(meta), ""]

    def table(d, label):
        lines = [f"| {label} | total | correct | WRONG | unparsed | infeasible ok | infeasible answered |",
                 "|---|---|---|---|---|---|---|"]
        for k, a in d.items():
            lines.append(f"| {k} | {a['total']} | {a['correct']} | {a['wrong']} | {a['unparsed']} | "
                         f"{a['infeasible_ok']} | {a['infeasible_answered']} |")
        return lines

    md += ["## Overall", *table({"all": summary["overall"]}, "scope"), "", "## By type",
           *table(summary["by_type"], "type"), "", "## By type / difficulty",
           *table(summary["by_type_difficulty"], "type/difficulty"), "",
           "## By template family (top 40 by size, plus every family with a WRONG)"]
    fams = summary["by_family"]
    keep = {k: v for i, (k, v) in enumerate(fams.items()) if i < 40 or v["wrong"] or v["infeasible_answered"]}
    md += table({k[:150].replace("|", "/"): v for k, v in keep.items()}, "family")
    md += ["", f"## WRONG / infeasible-but-answered ({len(bad)})", ""]
    for r in bad:
        md.append(f"- `{r['task_id']}` [{r['type']}/{r['difficulty']}] {r['outcome']} expected={r['expected']!r} "
                  f"got={r['got']!r} note={r['note']!r}")
        md.append(f"  - {r['instruction'][:600]}")
    (out / "fuzz.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000, help="total tasks (math/data/search only)")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000, help="first chunk seed")
    ap.add_argument("--out", type=Path, default=Path("runs/fuzz"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out / stamp
    base = Path(tempfile.mkdtemp(prefix="fuzzbase_"))
    rows: list[dict] = []
    t0 = time.monotonic()
    done = k = 0
    try:
        while done < a.n:
            n = min(a.chunk, a.n - done)
            seed = a.seed + k
            rng = random.Random(seed)
            diff = DIFFS[k % len(DIFFS)]
            novel, inf = rng.choice(NOVELTY), rng.choice(INFEASIBLE)
            rows += run_chunk(base, n, seed, diff, novel, inf, a.workers)
            done += n
            k += 1
            c = Counter(r["outcome"] for r in rows)
            print(f"[{done}/{a.n}] seed={seed} {diff} novel={novel} inf={inf} wrong={c['wrong']} "
                  f"unparsed={c['unparsed']} correct={c['correct']} {time.monotonic() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    meta = {"n": len(rows), "chunk": a.chunk, "first_seed": a.seed, "workers": a.workers,
            "seconds": round(time.monotonic() - t0, 1)}
    summary = write_reports(out, meta, rows)
    print(json.dumps(summary["by_type"], indent=1))
    print(f"report: {out / 'fuzz.md'}")


if __name__ == "__main__":
    main()

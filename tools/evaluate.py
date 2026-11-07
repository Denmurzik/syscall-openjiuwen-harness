"""Scorer for a finished run over a generated (or example) task set."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_TIMEOUT = 60


def _read_answer(task_dir: Path) -> str | None:
    p = Path(task_dir) / "answer.txt"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _run_unittest(workspace: Path, test_file: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(workspace / "mylib.py", tmp / "mylib.py")
        shutil.copy(test_file, tmp / test_file.name)
        try:
            r = subprocess.run([sys.executable, "-B", test_file.name], cwd=tmp, capture_output=True,
                               timeout=TEST_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return r.returncode == 0


def check(task_dir: Path, hidden_dir: Path) -> bool:
    task_dir, hidden_dir = Path(task_dir), Path(hidden_dir)
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    expected = meta.get("expected")
    answer = _read_answer(task_dir)
    if expected is None:  # infeasible: best outcome is no answer; evaluate() leaves it unscored either way
        return answer is None
    if answer is None:
        return False
    kind = meta.get("check")
    try:
        if kind == "exact_int":
            return int(answer.strip()) == int(str(expected).strip())
        if kind == "exact_str":
            return answer.strip() == str(expected).strip()
        if kind == "unittest":
            if answer.strip() != "DONE":
                return False
            workspace = task_dir / "workspace"
            hidden = hidden_dir / "test_hidden.py"
            test = hidden if hidden.is_file() else workspace / "test_mylib.py"
            return (workspace / "mylib.py").is_file() and test.is_file() and _run_unittest(workspace, test)
    except (ValueError, OSError):
        return False
    return False


def _tokens(log: Path) -> float:
    total = 0.0
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    total += float(rec.get("cost", 0) or 0)
                except (ValueError, TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return total


def evaluate(run: Path, log: Path) -> dict:
    run = Path(run)
    tasks_dir = run / "tasks" if (run / "tasks").is_dir() else run
    score = max_score = 0.0
    by_type: dict[str, dict] = {}
    infeasible = {"total": 0, "skipped": 0}  # unscored: 0 points with or without answer.txt
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        try:
            meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
            reward = float(meta["reward"])
            typ = str(meta["task_type"])
        except Exception:
            continue
        if meta.get("expected") is None:
            infeasible["total"] += 1
            infeasible["skipped"] += not (d / "answer.txt").is_file()
            continue
        ok = check(d, run / "hidden" / d.name)
        s = by_type.setdefault(typ, {"solved": 0, "correct": 0, "total": 0})
        s["total"] += 1
        s["solved"] += (d / "answer.txt").is_file()
        s["correct"] += ok
        max_score += reward
        score += reward if ok else 0
    return {"score": score, "max_score": max_score, "by_type": by_type, "infeasible": infeasible, "tokens": _tokens(log)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args(argv)
    log = a.log or Path(os.environ.get("HARNESS_LOG_DIR", "/tmp/harness")) / "trajectory.jsonl"
    print(json.dumps(evaluate(a.run, log), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from harness.tasks import load_tasks
from tools import evaluate, gen_tasks

FIX = Path(__file__).parent / "fixtures" / "examples"
ROOT = Path(__file__).resolve().parent.parent


def _read(task):
    return json.loads((task.dir / "task.json").read_text(encoding="utf-8"))


def test_generated_tasks_load_and_examples_solvable(tmp_path):
    gen_tasks.main(["--out", str(tmp_path), "--n", "40", "--seed", "1"])
    tasks = load_tasks(tmp_path / "tasks")
    assert len(tasks) == 40
    types = {}
    infeasible = 0
    for t in tasks:
        data = _read(t)
        types[t.task_type] = types.get(t.task_type, 0) + 1
        assert t.workspace.is_dir()
        assert 1 <= data["reward"] <= 5
        if data["expected"] is None:
            infeasible += 1
        else:
            assert isinstance(data["expected"], str) and data["expected"]
    assert types == {"math": 10, "search": 10, "data_analysis": 10, "coding": 10}
    assert infeasible == 8
    for t in tasks:
        if t.task_type == "coding" and _read(t)["expected"] is not None:
            assert (tmp_path / "hidden" / t.task_id / "test_hidden.py").is_file()


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        gen_tasks.main(["--out", str(d), "--n", "12", "--seed", "7"])
    for f in sorted((a / "tasks").rglob("*")):
        if f.is_file():
            assert f.read_bytes() == (b / "tasks" / f.relative_to(a / "tasks")).read_bytes()


def test_cli_module_entry(tmp_path):
    r = subprocess.run([sys.executable, "-B", "-m", "tools.gen_tasks", "--out", str(tmp_path / "o"),
                        "--n", "8", "--seed", "3", "--infeasible", "0", "--novel", "0"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    tasks = load_tasks(tmp_path / "o" / "tasks")
    assert len(tasks) == 8
    assert all(_read(t)["expected"] is not None for t in tasks)


def test_evaluate_counts_examples(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIX, run / "tasks")
    r = evaluate.evaluate(run, tmp_path / "none.jsonl")
    assert r["score"] == 0 and r["max_score"] == 5 + 2 + 3 + 3 + 1 + 2 + 2 + 2
    assert r["by_type"]["math"] == {"solved": 0, "correct": 0, "total": 2}
    (run / "tasks" / "math_0000" / "answer.txt").write_text("389\n")
    (run / "tasks" / "math_0004" / "answer.txt").write_text("1.0\n")
    r = evaluate.evaluate(run, tmp_path / "none.jsonl")
    assert r["score"] == 5
    assert r["by_type"]["math"] == {"solved": 2, "correct": 1, "total": 2}


def test_evaluate_cli_and_tokens(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIX, run / "tasks")
    log = tmp_path / "t.jsonl"
    log.write_text('{"cost": 10.5}\n{"cost": 4}\nnot json\n{"x": 1}\n')
    r = subprocess.run([sys.executable, "-B", "-m", "tools.evaluate", "--run", str(run), "--log", str(log)],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["tokens"] == 14.5 and out["score"] == 0


def test_generated_answers_roundtrip_and_infeasible(tmp_path):
    gen_tasks.main(["--out", str(tmp_path), "--n", "60", "--seed", "2", "--infeasible", "0.3"])
    tasks = load_tasks(tmp_path / "tasks")
    for t in tasks:
        exp = _read(t)["expected"]
        hidden = tmp_path / "hidden" / t.task_id
        if t.task_type == "coding" and exp is not None:
            assert not evaluate.check(t.dir, hidden)
            (t.dir / "answer.txt").write_text("DONE\n")
            assert not evaluate.check(t.dir, hidden)  # mutated library still fails hidden tests
        elif exp is None:
            assert evaluate.check(t.dir, hidden)  # no answer.txt is correct
            (t.dir / "answer.txt").write_text("0\n")
            assert not evaluate.check(t.dir, hidden)
        else:
            (t.dir / "answer.txt").write_text(exp + "\n")
            assert evaluate.check(t.dir, hidden)
            (t.dir / "answer.txt").write_text("xx\n")
            assert not evaluate.check(t.dir, hidden)


def test_coding_fixed_library_passes_hidden(tmp_path):
    gen_tasks.main(["--out", str(tmp_path), "--n", "40", "--seed", "5", "--infeasible", "0"])
    n = 0
    for t in load_tasks(tmp_path / "tasks"):
        if t.task_type != "coding":
            continue
        src = (t.workspace / "mylib.py").read_text()
        names = re.findall(r"^def (\w+)\(", src, re.M)
        assert 3 <= len(names) <= 5
        visible = (t.workspace / "test_mylib.py").read_text()
        assert visible.count("assertEqual") == len(names)
        hidden = (tmp_path / "hidden" / t.task_id / "test_hidden.py").read_text()
        assert hidden.count("assertEqual") == 3 * len(names)
        (t.workspace / "mylib.py").write_text(gen_tasks.correct_library(names))
        (t.dir / "answer.txt").write_text("DONE\n")
        assert evaluate.check(t.dir, tmp_path / "hidden" / t.task_id)
        (t.dir / "answer.txt").write_text("done please\n")
        assert not evaluate.check(t.dir, tmp_path / "hidden" / t.task_id)
        n += 1
    assert n == 10


def test_library_has_enough_functions():
    assert len(gen_tasks.LIB) >= 12


def test_coding_visible_tests_fail_before_fix(tmp_path):
    gen_tasks.main(["--out", str(tmp_path), "--n", "120", "--seed", "9", "--infeasible", "0.25"])
    seen = 0
    for t in load_tasks(tmp_path / "tasks"):
        if t.task_type != "coding":
            continue
        r = subprocess.run([sys.executable, "-B", "test_mylib.py"], cwd=t.workspace,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode != 0, t.task_id  # buggy library or contradictory tests: never green at start
        seen += 1
    assert seen == 30

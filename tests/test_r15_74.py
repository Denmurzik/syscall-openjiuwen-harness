"""Round 15, task 74: a model-independent infeasible verdict in phase 2 no longer withdraws the stored guess. The
guess and its answer.txt stay (an answer to an infeasible task is not counted; if the verdict is wrong, the guess may
score); the task is still counted infeasible and gets no further model calls."""
from pathlib import Path

import pytest

from harness import scheduler
from harness.result import SolveResult
from harness.tasks import Task
from tests.test_scheduler import _make, guess_target


def _run(tmp_path, monkeypatch, note, n=1):
    tasks = tmp_path / "tasks"
    dirs = [_make(tasks, f"s{i}", "search", f"q {i}?") for i in range(n)]
    seen = []

    async def grp(ttype, template, members, client):
        seen.extend(("grp", t.task_id) for t, _ in members)
        return {t.task_id: SolveResult("unparsed") for t, _ in members}

    async def prog(task, client):
        seen.append(("prog", task.task_id))
        return SolveResult("infeasible", note=note)

    monkeypatch.setattr(scheduler, "_phase1_target", guess_target)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    counts = scheduler.run(tasks, 100000, object())
    return counts, dirs, seen


@pytest.mark.parametrize("note", ["start entity not found: X", "division by zero in step 2"])
def test_model_independent_verdict_keeps_the_guess_on_disk(tmp_path, monkeypatch, note):
    counts, (d,), seen = _run(tmp_path, monkeypatch, note)
    assert (d / "answer.txt").read_text(encoding="utf-8").strip() == "42"
    assert counts["infeasible"] == 1 and counts["solved"] == 0


def test_model_independent_verdict_ends_the_model_calls_on_the_task(tmp_path, monkeypatch):
    counts, dirs, seen = _run(tmp_path, monkeypatch, "start entity not found: X", n=2)
    for d in dirs:
        tid = d.name
        assert seen.count(("prog", tid)) == 1  # the verdict is final: no later stage asks again
        last = max(i for i, s in enumerate(seen) if s[1] == tid)
        assert seen[last] == ("prog", tid)
        assert (d / "answer.txt").read_text(encoding="utf-8").strip() == "42"
    assert counts["infeasible"] == 2


def test_model_verdict_unchanged_guess_counts_solved(tmp_path, monkeypatch):
    counts, (d,), _ = _run(tmp_path, monkeypatch, "program printed IMPOSSIBLE")
    assert (d / "answer.txt").read_text(encoding="utf-8").strip() == "42" and counts["solved"] == 1


def _task(root: Path, tid: str) -> Task:
    d = _make(root, tid, "search", "q?")
    return Task(task_id=tid, task_type="search", instruction="q?", reward=1.0, check="exact_str", dir=d,
                workspace=d / "workspace")


def _guess(v="7"):
    return SolveResult("unparsed", extra={"best_guess": v})


def test_write_guesses_keeps_a_proven_task_infeasible(tmp_path):
    a, b, c = (_task(tmp_path, t) for t in ("a", "b", "c"))
    (a.dir / "answer.txt").write_text("7\n", encoding="utf-8")  # a: our guess already on disk
    by_id = {t.task_id: t for t in (a, b, c)}
    status = {"a": "infeasible", "b": "infeasible", "c": "unparsed"}
    scheduler._write_guesses(by_id, {"a": _guess(), "b": _guess("8"), "c": _guess("9")}, {}, status,
                             written={"a"}, proven={"a", "b"})
    assert status == {"a": "infeasible", "b": "infeasible", "c": "solved"}
    assert (a.dir / "answer.txt").read_text(encoding="utf-8").strip() == "7"
    assert (b.dir / "answer.txt").read_text(encoding="utf-8").strip() == "8"  # a guess not yet on disk is written
    assert (c.dir / "answer.txt").read_text(encoding="utf-8").strip() == "9"


def test_write_guesses_without_proven_is_unchanged(tmp_path):
    a = _task(tmp_path, "a")
    status = {"a": "infeasible"}  # a model's IMPOSSIBLE: the guess counts as an answer, as before
    scheduler._write_guesses({"a": a}, {"a": _guess()}, {}, status)
    assert status == {"a": "solved"} and (a.dir / "answer.txt").exists()

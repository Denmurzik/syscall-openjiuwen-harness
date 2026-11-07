import os
from pathlib import Path

from harness.tasks import copy_task, load_tasks, write_answer

FIX = Path(__file__).parent / "fixtures" / "examples"


def test_load_tasks_reads_all_examples():
    tasks = load_tasks(FIX)
    assert sorted(t.task_id for t in tasks) == ["coding_0003", "coding_0007", "data_analysis_0002",
        "data_analysis_0006", "math_0000", "math_0004", "search_0001", "search_0005"]
    m = next(t for t in tasks if t.task_id == "math_0000")
    assert m.task_type == "math" and m.reward == 5 and m.check == "exact_int"
    assert m.workspace == m.dir / "workspace"


def test_load_tasks_skips_broken_dirs(tmp_path):
    (tmp_path / "no_json").mkdir()
    (tmp_path / "bad_json").mkdir(); (tmp_path / "bad_json" / "task.json").write_text("{oops")
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "task.json").write_text('{"task_id":"ok","task_type":"math","instruction":"x","reward":1}')
    assert [t.task_id for t in load_tasks(tmp_path)] == ["ok"]   # workspace may be missing; dir still loads


def test_write_answer_next_to_task_json(tmp_path):
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    write_answer(t, " 389 ")
    assert (t.dir / "answer.txt").read_text() == "389\n"
    assert not (t.workspace / "answer.txt").exists()


def test_write_answer_falls_back_when_replace_denied(tmp_path, monkeypatch):
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError()))
    write_answer(t, "7")
    assert (t.dir / "answer.txt").read_text() == "7\n"


def test_load_tasks_nested_layout(tmp_path):
    for sub in ("__MACOSX", ".hidden"):
        (tmp_path / "outer" / sub / "x").mkdir(parents=True)
        (tmp_path / "outer" / sub / "x" / "task.json").write_text("{}")
    for tid in ("a", "b"):
        d = tmp_path / "outer" / tid
        d.mkdir(parents=True)
        (d / "task.json").write_text(
            f'{{"task_id":"{tid}","task_type":"math","instruction":"x","reward":1}}')
    (tmp_path / "__MACOSX").mkdir()
    assert [t.task_id for t in load_tasks(tmp_path)] == ["a", "b"]


def test_load_tasks_ignores_macosx_and_dot_dirs_at_top(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__MACOSX" / "task.json").write_text("{}")
    d = tmp_path / "ok"
    d.mkdir()
    (d / "task.json").write_text('{"task_id":"ok","task_type":"math","instruction":"x","reward":1}')
    assert [t.task_id for t in load_tasks(tmp_path)] == ["ok"]

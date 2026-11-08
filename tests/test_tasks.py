import json
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


def test_answer_sink_file_and_dir(tmp_path):
    from harness.tasks import set_answer_sink
    ts = load_tasks(copy_task(load_tasks(FIX)[0], tmp_path).dir.parent)
    try:
        f = tmp_path / "o" / "a.json"
        set_answer_sink(f)
        write_answer(ts[0], " 1 ")
        write_answer(ts[0], "2")
        assert json.loads(f.read_text()) == {ts[0].task_id: "2"}
        d = tmp_path / "dd"; d.mkdir()
        set_answer_sink(d)
        write_answer(ts[0], "z")
        assert json.loads((d / "answers.json").read_text()) == {ts[0].task_id: "z"}
    finally:
        set_answer_sink(None)


def test_answer_sink_error_keeps_answer_txt(tmp_path):
    from harness.tasks import set_answer_sink
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    blocker = tmp_path / "f"; blocker.write_text("x")
    try:
        set_answer_sink(blocker / "x.json")
        write_answer(t, "7")
    finally:
        set_answer_sink(None)
    assert (t.dir / "answer.txt").read_text() == "7\n"


def _mk(tmp_path, raw: bytes, name="t"):
    (tmp_path / name).mkdir()
    (tmp_path / name / "task.json").write_bytes(raw)


def test_load_tasks_bom_and_reward_fallback(tmp_path):
    base = '"task_id":"%s","task_type":"math","instruction":"x"'
    _mk(tmp_path, b"\xef\xbb\xbf{" + (base % "bom").encode() + b',"reward":2}', "a")
    _mk(tmp_path, ("{" + base % "nokey" + "}").encode(), "b")
    _mk(tmp_path, ("{" + base % "junk" + ',"reward":"high"}').encode(), "c")
    _mk(tmp_path, ("{" + base % "null" + ',"reward":null}').encode(), "d")
    by = {t.task_id: t.reward for t in load_tasks(tmp_path)}
    assert by == {"bom": 2.0, "nokey": 1.0, "junk": 1.0, "null": 1.0}


def test_write_answer_replaces_read_only_file(tmp_path):
    import stat
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    a = t.dir / "answer.txt"
    a.write_text("stale\n")
    os.chmod(a, stat.S_IREAD)
    write_answer(t, "9")
    assert a.read_text() == "9\n"


def test_write_answer_read_only_when_replace_denied(tmp_path, monkeypatch):
    import stat
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    a = t.dir / "answer.txt"
    a.write_text("stale\n")
    os.chmod(a, stat.S_IREAD)
    monkeypatch.setattr(os, "replace", lambda *a_: (_ for _ in ()).throw(PermissionError()))
    write_answer(t, "9")
    assert a.read_text() == "9\n"


def test_reward_overflow_falls_back(tmp_path):
    base = '"task_id":"%s","task_type":"math","instruction":"x"'
    _mk(tmp_path, ("{" + base % "big" + ',"reward":' + "9" * 400 + "}").encode(), "a")
    _mk(tmp_path, ("{" + base % "inf" + ',"reward":1e999}').encode(), "b")
    by = {t.task_id: t.reward for t in load_tasks(tmp_path)}
    assert by == {"big": 1.0, "inf": 1.0}


def test_write_answer_chmod_keeps_other_mode_bits(tmp_path, monkeypatch):
    import stat
    t = copy_task(load_tasks(FIX)[0], tmp_path)
    a = t.dir / "answer.txt"
    a.write_text("stale" + chr(10))
    real_replace, real_chmod, calls = os.replace, os.chmod, []
    state = {"denied": False}

    def replace(src, dst):
        if not state["denied"]:
            state["denied"] = True
            raise PermissionError("read-only")
        return real_replace(src, dst)

    def chmod(path, mode, *x, **k):
        calls.append(mode)
        return real_chmod(path, mode, *x, **k)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "chmod", chmod)
    real_stat = os.stat

    def fake_stat(path, *x, **k):
        r = real_stat(path, *x, **k)
        if str(path) == str(a):
            return os.stat_result((stat.S_IFREG | 0o454,) + tuple(r)[1:])
        return r
    monkeypatch.setattr(os, "stat", fake_stat)
    write_answer(t, "9")
    assert calls and calls[0] & 0o777 == (0o454 | stat.S_IWRITE) & 0o777 and calls[0] & stat.S_IREAD
    assert a.read_text() == "9" + chr(10)

import json
import shutil
from pathlib import Path

import pytest

import run_agent
from harness import tasks as tasks_mod

FIX = Path(__file__).parent / "fixtures" / "examples"


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "TASKS", "OUT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    yield
    tasks_mod.set_answer_sink(None)


def _copy(tmp_path):
    d = tmp_path / "tasks"
    shutil.copytree(FIX, d)
    return d


def _run(argv):
    with pytest.raises(SystemExit) as e:
        run_agent.main(argv)
    assert e.value.code == 0


def test_resolve_chain(tmp_path):
    dirs = {}
    for n in "apt":
        d = tmp_path / n
        shutil.copytree(FIX, d)
        dirs[n] = d
    env = {"TASKS": str(dirs["t"])}
    assert run_agent._resolve_paths([], env)[0] == dirs["t"]
    assert run_agent._resolve_paths([str(dirs["p"])], env)[:2] == (dirs["p"], "positional")
    assert run_agent._resolve_paths(["--tasks-dir", str(dirs["a"]), str(dirs["p"])], env)[0] == dirs["a"]
    assert run_agent._resolve_paths(["--output", "o.json"], {"OUT": "e"})[2] == "o.json"
    assert run_agent._resolve_paths([], {"OUT": "e"})[2] == "e"


def test_bad_positional_falls_through(tmp_path):
    d = _copy(tmp_path)
    env = {"TASKS": str(d)}
    assert run_agent._resolve_paths(["--foo", "bar"], env)[:2] == (d, "env TASKS")
    assert run_agent._resolve_paths(["extra"], env)[0] == d
    empty = tmp_path / "empty"; empty.mkdir()
    assert run_agent._resolve_paths(["--tasks-dir", str(empty)], env)[0] == d
    assert run_agent._resolve_paths([str(d)], {})[1] == "positional"


def test_parse_errors_do_not_abort(tmp_path):
    d = _copy(tmp_path)
    env = {"TASKS": str(d)}
    for argv in (["--tasks-dir"], ["-h"], ["--out"], ["--o", "x"]):
        assert run_agent._resolve_paths(argv, env)[0] == d


def test_env_tasks_without_flag(monkeypatch, tmp_path):
    d = _copy(tmp_path)
    monkeypatch.setenv("TASKS", str(d))
    _run([])
    assert any(d.glob("*/answer.txt"))


def test_out_as_file(monkeypatch, tmp_path):
    d = _copy(tmp_path)
    out = tmp_path / "res" / "sol.json"
    _run(["--tasks-dir", str(d), "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data and all(isinstance(v, str) for v in data.values())
    for t in d.iterdir():
        if (t / "answer.txt").exists():
            assert data[t.name] == (t / "answer.txt").read_text().strip()


def test_out_as_directory(monkeypatch, tmp_path):
    d = _copy(tmp_path)
    out = tmp_path / "outdir"
    out.mkdir()
    monkeypatch.setenv("OUT", str(out))
    _run(["--tasks-dir", str(d)])
    assert json.loads((out / "answers.json").read_text(encoding="utf-8"))


def test_out_failure_does_not_break_answer(monkeypatch, tmp_path):
    d = _copy(tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    _run(["--tasks-dir", str(d), "--out", str(blocker / "sub" / "a.json")])
    assert any(d.glob("*/answer.txt"))


def test_trailing_slash_out_is_a_resolved_directory(tmp_path, monkeypatch):
    d = _copy(tmp_path)
    seen = []
    monkeypatch.setattr(tasks_mod, "set_answer_sink", lambda p: seen.append(p))
    monkeypatch.chdir(tmp_path)
    _run(["--tasks-dir", str(d), "--out", "outdir/"])
    p = str(seen[0]).replace("\\", "/")
    assert p.endswith("/outdir/") and Path(p).is_absolute()

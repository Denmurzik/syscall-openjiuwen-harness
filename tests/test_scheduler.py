import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness import config, scheduler
from harness.estimator import estimate_json_tokens, estimate_tokens
from harness.llm import BudgetExhausted, ChatResult
from harness.result import SolveResult
from harness.tasks import Task
from harness.trajectory import Trajectory

FIX = Path(__file__).parent / "fixtures" / "examples"
ROOT = Path(__file__).parent.parent


def _fixtures(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for d in sorted(FIX.iterdir()):
        shutil.copytree(d, dest / d.name)
    return dest


def _make(dest: Path, tid: str, ttype: str, instruction: str, reward=1.0, check="exact_str"):
    d = dest / tid
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({
        "task_id": tid, "task_type": ttype, "instruction": instruction,
        "reward": reward, "check": check}), encoding="utf-8")
    return d


# ---- top-level phase-1 stand-ins (spawn needs importable targets)

def sleepy_target(queue, task, solver_name=None):
    time.sleep(10)


def unparsed_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


@pytest.fixture(autouse=True)
def _log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))


def test_phase1_solves_all_examples_without_llm(tmp_path):
    tasks = _fixtures(tmp_path / "tasks")
    summary = scheduler.run(tasks, 1000, None)
    assert summary["solved"] == 8 and summary["spent"] == 0
    for d in tasks.iterdir():
        meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
        ans = (d / "answer.txt").read_text(encoding="utf-8").strip()
        if meta["task_type"] == "coding":
            assert ans == "DONE"
            r = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "workspace"],
                               cwd=d, capture_output=True)
            assert r.returncode == 0, r.stderr
        else:
            assert ans == str(meta["expected"])


class SlowFake:
    """2nd finishing call raises BudgetExhausted; counts concurrency."""
    def __init__(self):
        self.inflight = 0
        self.max_inflight = 0
        self.started = 0
        self.finished = 0
        self.started_after_exhaust = 0
        self.exhausted = False

    async def chat(self, model, messages, max_tokens, **extra):
        if self.exhausted:  # like the real client: a flagged ledger refuses new calls
            self.started_after_exhaust += 1
            raise BudgetExhausted("ledger exhausted")
        self.started += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.2)
        self.inflight -= 1
        self.finished += 1
        if self.finished == 2:
            self.exhausted = True
            raise BudgetExhausted("out")
        return ChatResult(text="not json", prompt_tokens=10, completion_tokens=5, cost=6)


def test_budget_exhaustion_stops_cleanly(tmp_path, monkeypatch):
    from harness import json_route
    units = {"inflight": 0, "max": 0, "started": 0, "after_exhaust": 0}

    async def counting_group(ttype, template, members, client):
        if client.exhausted:
            units["after_exhaust"] += 1
        units["started"] += 1
        units["inflight"] += 1
        units["max"] = max(units["max"], units["inflight"])
        try:
            return await json_route.solve_group(ttype, template, members, client)
        finally:
            units["inflight"] -= 1

    monkeypatch.setattr(scheduler, "solve_group", counting_group)
    tasks = _fixtures(tmp_path / "tasks")
    for f in list(tasks.iterdir()):
        if f.name != "math_0000":
            shutil.rmtree(f)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    for w in words:
        _make(tasks, f"search_{w}", "search", f"Frobnicate the {w} quietly?")
    fake = SlowFake()
    summary = scheduler.run(tasks, 100000, fake)
    assert fake.max_inflight <= 5
    assert units["max"] <= 5 and units["after_exhaust"] == 0 and units["started"] < 8
    assert fake.started <= 10  # 5 first calls + retries; the other 3 groups never start
    assert (tasks / "math_0000" / "answer.txt").read_text(encoding="utf-8").strip() == "389"
    assert summary["solved"] == 1


def test_hung_phase1_task_is_killed(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _make(tasks, "math_x", "math", "whatever")
    monkeypatch.setattr(scheduler, "_phase1_target", sleepy_target)
    monkeypatch.setitem(config.PHASE1_TIMEOUT, "default", 1)
    t0 = time.monotonic()
    summary = scheduler.run(tasks, 1000, None)
    assert time.monotonic() - t0 < 5
    assert summary["unparsed"] == 1 and summary["solved"] == 0


def test_run_agent_exits_zero_on_missing_dir(tmp_path):
    r = subprocess.run([sys.executable, "-B", str(ROOT / "run_agent.py"), "--tasks-dir",
                        str(tmp_path / "nope")], capture_output=True, cwd=tmp_path,
                       env={**__import__("os").environ, "HARNESS_LOG_DIR": str(tmp_path / "logs")})
    assert r.returncode == 0


def test_roi_order(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for tid, reward in (("coding_a", 1.0), ("coding_b", 5.0)):
        d = _make(tasks, tid, "coding", "fix it", reward, "unittest")
        (d / "workspace" / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (d / "workspace" / "test_mylib.py").write_text("import mylib\n", encoding="utf-8")
    order = []

    async def fake_fix(task, client):
        order.append(task.task_id)
        await asyncio.sleep(0.05)
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    monkeypatch.setattr(scheduler, "fix_coding", fake_fix)
    scheduler.run(tasks, 100000, object())
    assert order == ["coding_b", "coding_a"]


class SpecFake:
    def __init__(self):
        self.calls = 0

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls += 1
        spec = json.dumps({"start": "{N1}", "ops": [["-", "{N2}"]]})
        return ChatResult(text=spec, prompt_tokens=10, completion_tokens=5, cost=6)


def test_json_group_before_program(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for i, (a, b) in enumerate([(10, 3), (20, 5), (30, 7)]):
        _make(tasks, f"math_{i}", "math",
              f"A warehouse starts with {a} boxes. It loses {b} boxes to theft. "
              "How many boxes does it have at the end?", check="exact_int")

    async def no_program(task, client):
        raise AssertionError("program route must not run")

    monkeypatch.setattr(scheduler, "solve_with_program", no_program)
    fake = SpecFake()
    summary = scheduler.run(tasks, 100000, fake)
    assert fake.calls == 1 and summary["solved"] == 3
    got = [(tasks / f"math_{i}" / "answer.txt").read_text(encoding="utf-8").strip() for i in range(3)]
    assert got == ["7", "15", "23"]


def test_trajectory_writes_jsonl(tmp_path):
    t = Trajectory(tmp_path / "sub")
    t.record({"phase": 1, "task_id": "a", "cost": 0})
    t.record({"phase": 2, "task_id": "b", "cost": 3.5})
    lines = (tmp_path / "sub" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["task_id"] for x in lines] == ["a", "b"]


def test_estimator_constants(tmp_path):
    assert estimate_json_tokens("x" * 300) == pytest.approx(300 / 3 / 10 + 80)
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text("a" * 300, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text("b" * 300, encoding="utf-8")
    t = Task("c", "coding", "fix", 1.0, "unittest", d, d / "workspace")
    assert estimate_tokens(t) == pytest.approx((600 + 2000) / 3 / 10 + 400)


def test_single_event_loop_across_2a_and_2b(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search", "Frobnicate the alpha quietly?")
    loops = []

    async def grp(ttype, template, members, client):
        loops.append(asyncio.get_running_loop())
        return {t.task_id: SolveResult("unparsed") for t, _ in members}

    async def prog(task, client):
        loops.append(asyncio.get_running_loop())
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    scheduler.run(tasks, 100000, object())
    assert len(loops) == 2 and loops[0] is loops[1]


def test_phase1_does_not_import_openjiuwen(tmp_path):
    tasks = _fixtures(tmp_path / "tasks")
    code = "; ".join(["import sys", "from pathlib import Path", "from harness import scheduler",
                      f"scheduler.run(Path({str(tasks)!r}), 1000, None)",
                      "sys.exit(1 if 'openjiuwen' in sys.modules else 0)"])
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                       env={**__import__("os").environ, "HARNESS_LOG_DIR": str(tmp_path / "logs")})
    assert r.returncode == 0, r.stderr


def _spec_infeasible(note):
    async def grp(ttype, template, members, client):
        return {t.task_id: SolveResult("infeasible", note=note) for t, _ in members}
    return grp


def test_2a_model_infeasible_falls_through_to_2b(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search", "Frobnicate the alpha quietly?")
    seen = []

    async def prog(task, client):
        seen.append(task.task_id)
        return SolveResult("solved", answer="42")

    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", _spec_infeasible("'X' has no 'Boss'"))
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    summary = scheduler.run(tasks, 100000, object())
    assert seen == ["search_a"] and summary["solved"] == 1
    assert (tasks / "search_a" / "answer.txt").read_text().strip() == "42"


@pytest.mark.parametrize("note", ["division by zero", "start entity not found: 'Bob'"])
def test_2a_model_independent_infeasible_is_final(tmp_path, monkeypatch, note):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search", "Frobnicate the alpha quietly?")

    async def prog(task, client):
        raise AssertionError("2b must not run")

    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", _spec_infeasible(note))
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    summary = scheduler.run(tasks, 100000, object())
    assert summary["infeasible"] == 1 and not (tasks / "search_a" / "answer.txt").exists()


def test_tasks_dir_probe_reports_unwritable(tmp_path, monkeypatch, capsys):
    import run_agent

    def boom(*a, **k):
        raise PermissionError("nope")

    monkeypatch.setattr(run_agent.tempfile, "mkstemp", boom)
    run_agent._probe_tasks_dir(tmp_path)
    err = capsys.readouterr().err
    assert err.startswith("TASKS DIR NOT WRITABLE: uid=") and "owner=" in err and "mode=" in err
    assert err.count("\n") == 1


def test_tasks_dir_probe_silent_when_writable(tmp_path, capsys):
    import run_agent
    run_agent._probe_tasks_dir(tmp_path)
    assert capsys.readouterr().err == "" and list(tmp_path.iterdir()) == []


def test_chdir_failure_does_not_disable_llm(tmp_path, monkeypatch, capsys):
    import run_agent
    from harness import llm
    got = {}

    class StubClient:
        def __init__(self, base, key, ledger):
            got["client"] = self

    def fake_run(tasks_dir, budget, client):
        got["run_client"] = client
        return {}

    monkeypatch.setattr(llm, "LLMClient", StubClient)
    monkeypatch.setattr(scheduler, "run", fake_run)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(run_agent.os, "chdir", lambda p: (_ for _ in ()).throw(OSError("denied")))
    with pytest.raises(SystemExit) as e:
        run_agent.main(["--tasks-dir", str(tmp_path)])
    assert e.value.code == 0 and got["run_client"] is got["client"]
    assert "cannot chdir" in capsys.readouterr().err

"""Expert 12, patch a (HARNESS_PRIO_CAL): the 2b order uses the estimate corrected by the measured factor of the
task type; the estimate that gates a unit against the ledger is not changed."""
import asyncio
import json

import pytest

from harness import scheduler
from harness.estimator import estimate_tokens
from harness.result import SolveResult
from harness.tasks import load_tasks


def unparsed_target(queue, task, solver_name=None, known_texts=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


@pytest.fixture(autouse=True)
def _log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))


def _tasks(root):
    d = root / "coding_a"
    (d / "workspace").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": "coding_a", "task_type": "coding", "instruction": "fix it",
                                             "reward": 5.0, "check": "unittest"}), encoding="utf-8")
    (d / "workspace" / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text("import mylib\n", encoding="utf-8")
    d = root / "math_a"
    (d / "workspace").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": "math_a", "task_type": "math", "reward": 2.0,
                                             "instruction": "What is the sum of the first 12 odd numbers?",
                                             "check": "exact_int"}), encoding="utf-8")
    return root


def _run(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path / "tasks")
    order = []

    async def fake_fix(task, client, *a, **kw):
        order.append(task.task_id)
        await asyncio.sleep(0.05)
        return SolveResult("unparsed")

    async def fake_program(task, client, *a, **kw):
        order.append(task.task_id)
        await asyncio.sleep(0.05)
        return SolveResult("unparsed")

    async def fake_group(ttype, template, members, client):
        return {}

    monkeypatch.setenv("HARNESS_REASON_MATH", "0")
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")
    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    monkeypatch.setattr(scheduler, "fix_coding", fake_fix)
    monkeypatch.setattr(scheduler, "solve_with_program", fake_program)
    monkeypatch.setattr(scheduler, "solve_group", fake_group)
    scheduler.run(tasks, 100000, object())
    return order


def test_calibrated_order_puts_a_well_paid_coding_task_before_cheap_math(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_PRIO_CAL", raising=False)
    assert _run(tmp_path, monkeypatch) == ["coding_a", "math_a"]


def test_switch_off_restores_the_round_22_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_PRIO_CAL", "0")
    assert _run(tmp_path, monkeypatch) == ["math_a", "coding_a"]


def test_prio_is_reward_per_corrected_estimate(tmp_path, monkeypatch):
    by_id = {t.task_id: t for t in load_tasks(_tasks(tmp_path / "tasks"))}
    monkeypatch.delenv("HARNESS_PRIO_CAL", raising=False)
    for tid, t in by_id.items():
        est = estimate_tokens(t)
        assert scheduler._prio_2b(t, est) == pytest.approx(t.reward / (est * scheduler.PRIO_COST[t.task_type]))
    monkeypatch.setenv("HARNESS_PRIO_CAL", "0")
    for t in by_id.values():
        assert scheduler._prio_2b(t, 200.0) == t.reward / 200.0


def test_unknown_type_keeps_its_estimate(monkeypatch):
    class T:
        task_type, reward = "other", 3.0
    monkeypatch.delenv("HARNESS_PRIO_CAL", raising=False)
    assert scheduler._prio_2b(T(), 150.0) == 3.0 / 150.0

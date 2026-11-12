"""A second opinion on FAST's first IMPOSSIBLE in 2b (round 19, task 86, HARNESS_SECOND_IMPOSSIBLE)."""
import asyncio
import json

import pytest

from harness import llm_routes as L
from harness import scheduler
from harness.config import MODEL_ALT
from harness.llm import ChatResult, LLMUnavailable
from harness.result import SolveResult
from harness.tasks import Task

IMP = SolveResult("infeasible", note="program printed IMPOSSIBLE")
KB_MISSING = SolveResult("unparsed", note="kb page or field not found")


def _task(tmp_path, ttype="search", check="exact_str", instruction="Who is older, Ann Lee or Bo Chen?", files=None):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    for n, c in (files or {}).items():
        p = d / "workspace" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c, encoding="utf-8")
    return Task("t", ttype, instruction, 1.0, check, d, d / "workspace")


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append((model, messages))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=10, cost=11)


def test_first_impossible():
    assert L.first_impossible(IMP) and L.first_impossible(KB_MISSING)
    assert not L.first_impossible(SolveResult("infeasible", note="division by zero"))
    assert not L.first_impossible(SolveResult("unparsed", note="program timed out"))
    assert not L.first_impossible(SolveResult("solved", answer="1", note="model program"))
    assert not L.first_impossible(SolveResult("unparsed", note="kb page or field not found",
                                              extra={"best_guess": "x"}))


def test_second_opinion_takes_a_well_formed_answer_as_a_guess(tmp_path):
    t = _task(tmp_path)
    c = Fake(["```python\nborn = {'Ann Lee': 1970, 'Bo Chen': 1980}\nprint(min(born, key=born.get))\n```"])
    r = asyncio.run(L.second_opinion(t, c, IMP))
    assert r.status == "unparsed" and r.extra == {"best_guess": "Ann Lee", "fill_only": True} and MODEL_ALT in r.note
    assert len(c.calls) == 1 and c.calls[0][0] == MODEL_ALT
    assert c.calls[0][1] == L._program_messages(t)   # the prompt FAST had


@pytest.mark.parametrize("reply", ["print('IMPOSSIBLE')", "print('unknown')", "print(None)", "raise SystemExit(3)",
                                   "print({'a': 1})", "print('')",
                                   # a hard-coded literal: FAST's route never takes one as it is either
                                   "print('Ann Lee')", "print(1970 - 1)",
                                   # a refusal in other words
                                   "x = 'Cannot be determined'\nprint(x)", "x = 'IMPOSSIBLE: no data'\nprint(x)",
                                   "x = 'Not enough information'\nprint(x)"])
def test_second_opinion_keeps_the_verdict_on_anything_else(tmp_path, reply):
    t = _task(tmp_path)
    r = asyncio.run(L.second_opinion(t, Fake([f"```python\n{reply}\n```"]), IMP))
    assert r is IMP


def test_second_opinion_keeps_the_verdict_on_a_wrong_shape(tmp_path):
    t = _task(tmp_path, ttype="data_analysis", check="exact_int", instruction="How many rows are there?")
    assert asyncio.run(L.second_opinion(t, Fake(["n = ['many']\nprint(n[0])"]), IMP)) is IMP
    r = asyncio.run(L.second_opinion(t, Fake(["rows = [1] * 7\nprint(len(rows))"]), IMP))
    assert r.extra == {"best_guess": "7", "fill_only": True}


def test_second_opinion_model_unavailable(tmp_path):
    t = _task(tmp_path)
    assert asyncio.run(L.second_opinion(t, Fake([LLMUnavailable("down")]), KB_MISSING)) is KB_MISSING


def test_second_opinion_runs_the_kb_preamble(tmp_path):
    kb = {"kb/ann_lee.md": "# Ann Lee\nBorn: 1970\n", "kb/bo_chen.md": "# Bo Chen\nBorn: 1980\n"}
    t = _task(tmp_path, files=kb)
    code = ("a, b = kb['Ann Lee'], kb['Bo Chen']\n"
            "print('Ann Lee' if int(a['Born']) < int(b['Born']) else 'Bo Chen')")
    r = asyncio.run(L.second_opinion(t, Fake([code]), KB_MISSING))
    assert r.extra == {"best_guess": "Ann Lee", "fill_only": True}


# the scheduler

def _make(tasks, tid, ttype, check="exact_str", reward=1.0, instruction="Who is older, Ann Lee or Bo Chen?"):
    d = tasks / tid
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"task_id": tid, "task_type": ttype, "instruction": instruction,
                                             "reward": reward, "check": check}), encoding="utf-8")


def unparsed_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    seen = {"second": [], "program": []}

    async def grp(ttype, template, members, client):
        return {t.task_id: SolveResult("unparsed") for t, _ in members}

    async def prog(task, client):
        seen["program"].append(task.task_id)
        return IMP

    async def second(task, client, first):
        seen["second"].append(task.task_id)
        assert first is IMP
        return SolveResult("unparsed", note="alt", extra={"best_guess": "Ann Lee"})

    async def no_reason(task, client):
        return SolveResult("unparsed", note="no reason here")

    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "solve_with_reason", no_reason)
    monkeypatch.setattr(scheduler, "second_opinion", second)
    monkeypatch.setattr(scheduler, "is_prose_task", lambda t: False)
    return seen


def test_scheduler_asks_once_and_stores_the_guess(tmp_path, wired):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search")
    _make(tasks, "data_analysis_b", "data_analysis")
    scheduler.run(tasks, 100000, object())
    assert sorted(wired["second"]) == ["data_analysis_b", "search_a"]
    assert (tasks / "search_a" / "answer.txt").read_text(encoding="utf-8").strip() == "Ann Lee"


def test_scheduler_switch_off(tmp_path, wired, monkeypatch):
    monkeypatch.setenv("HARNESS_SECOND_IMPOSSIBLE", "0")
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search")
    scheduler.run(tasks, 100000, object())
    assert wired["second"] == [] and wired["program"] == ["search_a"]
    assert not (tasks / "search_a" / "answer.txt").exists()


def test_scheduler_no_second_opinion_on_math_or_prose(tmp_path, wired, monkeypatch):
    monkeypatch.setattr(scheduler, "_reason_2b", lambda *a: False)   # math goes to the FAST program here
    tasks = tmp_path / "tasks"
    _make(tasks, "math_a", "math", check="exact_int", instruction="How much is the total?")
    scheduler.run(tasks, 100000, object())
    assert wired["program"] == ["math_a"] and wired["second"] == []
    monkeypatch.setattr(scheduler, "is_prose_task", lambda t: True)
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "0")   # a prose task on the plain program route
    tasks = tmp_path / "tasks2"
    _make(tasks, "search_a", "search")
    scheduler.run(tasks, 100000, object())
    assert wired["second"] == []


def test_scheduler_no_second_opinion_past_half_the_budget(tmp_path, wired, monkeypatch):
    monkeypatch.setattr(scheduler, "_prose_more", lambda *a: False)
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search")
    scheduler.run(tasks, 100000, object())
    assert wired["second"] == [] and wired["program"] == ["search_a"]


def test_scheduler_a_failing_second_opinion_keeps_the_verdict(tmp_path, wired, monkeypatch):
    async def broken(task, client, first):
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler, "second_opinion", broken)
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search")
    scheduler.run(tasks, 100000, object())
    assert wired["program"] == ["search_a"] and not (tasks / "search_a" / "answer.txt").exists()
    rows = [json.loads(ln) for ln in (tmp_path / "logs" / "trajectory.jsonl").read_text().splitlines()]
    assert [r["note"] for r in rows if r.get("task_id") == "search_a" and r["phase"] == 2][0] == IMP.note


def test_scheduler_second_opinion_never_replaces_a_stored_guess(tmp_path, wired, monkeypatch):
    async def grp(ttype, template, members, client):   # 2a leaves a guess of its own
        return {t.task_id: SolveResult("unparsed", note="2a guess", extra={"best_guess": "Bo Chen"})
                for t, _ in members}

    async def second(task, client, first):
        wired["second"].append(task.task_id)
        return SolveResult("unparsed", note="alt", extra={"best_guess": "Ann Lee", "fill_only": True})

    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "second_opinion", second)
    tasks = tmp_path / "tasks"
    _make(tasks, "search_a", "search")
    scheduler.run(tasks, 100000, object())
    assert wired["second"] == ["search_a"]
    assert (tasks / "search_a" / "answer.txt").read_text(encoding="utf-8").strip() == "Bo Chen"

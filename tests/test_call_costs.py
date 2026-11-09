"""Per-call costs in the trajectory: each unit carries its own model calls, not a ledger delta."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from harness import config, scheduler
from harness.budget import TokenLedger
from harness.config import TIMEOUT_CHARGE
from harness.llm import LLMClient, LLMUnavailable
from harness.result import SolveResult
from harness.trajectory import CALLS
from tests.test_scheduler import _make, unparsed_target
from tools.evaluate import _tokens


class ScriptedModel:
    """Replies by template: 'flood' gets bad JSON fast and then a slow good spec; others one medium reply."""

    async def invoke(self, messages, **kw):
        user = messages[1]["content"]
        if "flood" in user:
            if len(messages) == 2:
                await asyncio.sleep(0.02)
                return _reply("not json", 100, 10)
            await asyncio.sleep(0.4)
            return _reply(json.dumps({"start": "{N1}", "steps": ["x-{N2}"]}), 100, 10)
        if "harbour" in user:
            return _reply('{"impossible": true}', 300, 0)
        await asyncio.sleep(0.15)
        return _reply(json.dumps({"start": "{N1}", "steps": ["x+{N2}"]}), 200, 30)


def _reply(text, p, c):
    return SimpleNamespace(content=text, usage_metadata=SimpleNamespace(input_tokens=p, output_tokens=c))


def _client(model=None, timeout=180):
    c = LLMClient("http://x/v1", "k", TokenLedger(100000, 1000), timeout=timeout)
    m = model or ScriptedModel()
    c._model = lambda name: m
    return c


def _traj(tmp_path):
    return [json.loads(x) for x in (tmp_path / "logs" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]


def _math(tasks, tid, text):
    _make(tasks, tid, "math", text + " How many boxes does it have at the end?", check="exact_int")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)


def test_parallel_units_carry_their_own_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")  # per-group units are what this test is about
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 10 boxes. It loses 3 boxes to a flood.")
    _math(tasks, "math_b", "A warehouse starts with 20 boxes. It loses 5 boxes to a flood.")
    _math(tasks, "math_c", "A warehouse starts with 30 boxes. It receives 4 boxes from a supplier.")
    client = _client()
    summary = scheduler.run(tasks, 100000, client)
    assert summary["solved"] == 3
    recs = {r["task_id"]: r for r in _traj(tmp_path) if r["phase"] == 2}
    flood = [recs["math_a"], recs["math_b"]]
    owner = next(r for r in flood if r["calls"])
    other = next(r for r in flood if r is not owner)
    # the flood unit made two calls (bad JSON, then a spec): 20 + 20; the other unit one call of 50
    assert [c["cost"] for c in owner["calls"]] == [20, 20] and owner["cost"] == 40
    assert other["calls"] == [] and other["cost"] == 0 and other["unit"] == owner["unit"]
    assert recs["math_c"]["cost"] == 50 and len(recs["math_c"]["calls"]) == 1
    for r in (owner, recs["math_c"]):
        assert r["model"] == config.MODEL_FAST
        assert all(set(c) >= {"model", "prompt_tokens", "completion_tokens", "cost"} for c in r["calls"])
    assert recs["math_c"]["calls"][0]["prompt_tokens"] == 200
    assert recs["math_c"]["calls"][0]["completion_tokens"] == 30
    assert _tokens(tmp_path / "logs" / "trajectory.jsonl") == pytest.approx(client.ledger.spent) == 90


def test_unit_without_settled_task_records_its_calls(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 10 boxes. It loses 3 boxes in the harbour.")

    async def prog(task, client):
        return SolveResult("unparsed", note="no program")

    monkeypatch.setattr(scheduler, "solve_with_program", prog)

    async def no_reason(task, client):   # the final REASON pass (task 36) is not what this test is about
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "solve_with_reason", no_reason, raising=False)
    client = _client()
    scheduler.run(tasks, 100000, client)
    recs = _traj(tmp_path)
    unit = [r for r in recs if r.get("status") == "unit"]
    assert len(unit) == 1 and unit[0]["unit"].startswith("group:math")
    assert unit[0]["cost"] == 30 and len(unit[0]["calls"]) == 1
    assert _tokens(tmp_path / "logs" / "trajectory.jsonl") == pytest.approx(client.ledger.spent) == 30


def test_failed_unit_record_carries_its_calls(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 30 boxes. It receives 4 boxes from a supplier.")

    async def grp(ttype, template, members, client):
        await client.chat(config.MODEL_ALT, [{"role": "system", "content": "s"},
                                             {"role": "user", "content": template}], 10)
        raise RuntimeError("boom")

    async def prog(task, client):
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    client = _client()
    scheduler.run(tasks, 100000, client)
    err = [r for r in _traj(tmp_path) if r.get("status") == "error"]
    assert len(err) == 1 and err[0]["cost"] == 50 and err[0]["model"] == config.MODEL_ALT
    assert _tokens(tmp_path / "logs" / "trajectory.jsonl") == pytest.approx(client.ledger.spent)


def test_charge_outside_calls_is_reported_unattributed(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 30 boxes. It receives 4 boxes from a supplier.")

    async def grp(ttype, template, members, client):
        client.ledger.charge(0, 7)
        return {t.task_id: SolveResult("unparsed") for t, _ in members}

    async def prog(task, client):
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)

    async def no_reason(task, client):   # the final REASON pass (task 36) is not what this test is about
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "solve_with_reason", no_reason, raising=False)
    client = _client()
    scheduler.run(tasks, 100000, client)
    extra = [r for r in _traj(tmp_path) if r.get("status") == "unattributed"]
    assert len(extra) == 1 and extra[0]["cost"] == pytest.approx(7)
    assert _tokens(tmp_path / "logs" / "trajectory.jsonl") == pytest.approx(client.ledger.spent) == 7


def test_phase1_records_have_zero_cost(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 30 boxes. It receives 4 boxes from a supplier.")
    scheduler.run(tasks, 100000, None)
    recs = _traj(tmp_path)
    assert [r["cost"] for r in recs if r["phase"] == 1] == [0]


# ------------------------------------------------------------------ client

def _call(c):
    return asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16))


def test_client_records_each_call_into_the_context_list():
    class One:
        async def invoke(self, messages, **kw):
            return _reply("hi", 500, 40)

    c = _client(One())

    async def go():
        calls = []
        tok = CALLS.set(calls)
        try:
            r = await c.chat("m", [{"role": "user", "content": "hi"}], 16)
        finally:
            CALLS.reset(tok)
        return r, calls

    r, calls = asyncio.run(go())
    assert calls == [{"model": "m", "prompt_tokens": 500, "completion_tokens": 40, "cost": 90}]
    assert r.cost == 90
    _call(c)  # no list set: nothing to record, nothing breaks


def test_client_records_timeout_charge():
    class Hang:
        async def invoke(self, messages, **kw):
            await asyncio.sleep(5)

    c = _client(Hang(), timeout=0.1)

    async def go():
        calls = []
        tok = CALLS.set(calls)
        try:
            with pytest.raises(LLMUnavailable):
                await c.chat("m", [{"role": "user", "content": "hi"}], 16)
        finally:
            CALLS.reset(tok)
        return calls

    calls = asyncio.run(go())
    assert len(calls) == 1 and 0 < calls[0]["cost"] <= TIMEOUT_CHARGE and calls[0]["cost"] == c.ledger.spent
    assert calls[0]["model"] == "m" and calls[0]["completion_tokens"] == calls[0]["cost"] and calls[0]["error"]

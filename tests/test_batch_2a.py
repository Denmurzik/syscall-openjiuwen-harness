import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import config, scheduler
from harness.json_route import BATCH_MAX, batchable, solve_batch
from harness.llm import BudgetExhausted, ChatResult
from harness.result import SolveResult
from harness.tasks import Task
from harness.templates import templatize
from tests.test_call_costs import _client, _math, _traj, _env  # noqa: F401
from tests.test_scheduler import unparsed_target  # noqa: F401


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=5, cost=6)


def _task(tid, start, loss, word="flood"):
    text = (f"A warehouse starts with {start} boxes. It loses {loss} boxes to a {word}. "
            "How many boxes does it have at the end?")
    return Task(tid, "math", text, 1.0, "exact_int", Path("."), Path("."))


def _item(tid, start, loss, word="flood"):
    t = _task(tid, start, loss, word)
    template, slots = templatize(t.instruction, "math")
    return template, [(t, slots)]


SPEC = {"start": "{N1}", "steps": ["x-{N2}"]}


def batch_reply(*specs, idx=None):
    return json.dumps({"specs": [{"i": (idx[k] if idx else k + 1), "spec": sp} for k, sp in enumerate(specs)]})


def test_batch_one_call_for_several_groups():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire"), _item("c", 9, 2, "storm")]
    client = FakeClient(batch_reply(SPEC, SPEC, SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert [out[k].answer for k in "abc"] == ["93", "45", "7"]
    assert len(client.calls) == 1 and client.calls[0]["max_tokens"] == 750
    assert "1. " in client.calls[0]["messages"][1]["content"] and "specs" in client.calls[0]["messages"][0]["content"]


def test_broken_element_falls_back_to_its_own_call():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    client = FakeClient(batch_reply(SPEC, "oops"), json.dumps(SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert out["a"].answer == "93" and out["b"].answer == "45" and len(client.calls) == 2


def test_unreadable_reply_or_wrong_length_falls_back_for_all():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    for first in ("not json", batch_reply(SPEC)):
        client = FakeClient(first, json.dumps(SPEC), json.dumps(SPEC))
        out = asyncio.run(solve_batch("math", items, client))
        assert out["a"].answer == "93" and out["b"].answer == "45" and len(client.calls) == 3


def test_batched_element_gets_the_step_count_check():
    from tests.test_json_route import MATH_0136, STEPS8, _spec
    t = Task("m", "math", MATH_0136, 1.0, "exact_int", Path("."), Path("."))
    template, slots = templatize(t.instruction, "math")
    other = _item("b", 50, 5, "fire")
    short = json.loads(_spec(STEPS8[:2] + STEPS8[3:]))
    client = FakeClient(batch_reply(short, SPEC), _spec(STEPS8))
    out = asyncio.run(solve_batch("math", [(template, [(t, slots)]), other], client))
    assert out["m"].answer == "9856" and out["b"].answer == "45"
    assert len(client.calls) == 2 and client.calls[1]["model"] == config.MODEL_ALT


def test_impossible_element_is_unparsed_without_extra_call():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    client = FakeClient(batch_reply({"impossible": True}, SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert out["a"].status == "unparsed" and out["b"].answer == "45" and len(client.calls) == 1


def test_budget_exhausted_on_the_batch_call_propagates():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    with pytest.raises(BudgetExhausted):
        asyncio.run(solve_batch("math", items, FakeClient(BudgetExhausted("x"))))


def test_batchable_rules():
    t, members = _item("a", 1, 2)
    assert batchable("math", t, members)
    assert not batchable("math", t, members * 3)
    assert not batchable("coding", t, members)
    s_t, _ = templatize("Whom does Ann Lee mentor?", "search")
    assert not batchable("search", s_t, members)  # reverse relation: no call at all


def test_knob_default_and_override(monkeypatch):
    monkeypatch.delenv("HARNESS_BATCH_2A", raising=False)
    assert config.batch_2a() is config.BATCH_2A
    monkeypatch.setenv("HARNESS_BATCH_2A", "1")
    assert config.batch_2a() is True
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")
    assert config.batch_2a() is False
    monkeypatch.setenv("HARNESS_BATCH_2A", "maybe")
    assert config.batch_2a() is config.BATCH_2A


# ------------------------------------------------------------------ scheduler
def _run_sched(tmp_path, monkeypatch, knob):
    tasks = tmp_path / "tasks"
    for i, word in enumerate(("flood", "fire", "storm", "riot", "quake")):
        _math(tasks, f"math_{i}", f"A warehouse starts with 30 boxes. It loses 4 boxes to a {word}.")
    seen = {"batch": [], "group": []}

    async def batch(ttype, chunk, client, on_group=None):
        seen["batch"].append(len(chunk))
        out = {t.task_id: SolveResult("solved", "26") for _, ms in chunk for t, _ in ms}
        for _, ms in chunk:
            on_group(ms, {t.task_id: out[t.task_id] for t, _ in ms})
        return out

    async def grp(ttype, template, members, client, spec=None):
        seen["group"].append(len(members))
        return {t.task_id: SolveResult("solved", "26") for t, _ in members}

    async def prog(task, client):
        return SolveResult("unparsed")

    monkeypatch.setattr(scheduler, "solve_batch", batch)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setenv("HARNESS_BATCH_2A", knob)
    counts = scheduler.run(tasks, 100000, _client())
    return seen, counts


def test_scheduler_batches_when_on(tmp_path, monkeypatch):
    seen, counts = _run_sched(tmp_path, monkeypatch, "1")
    assert seen["batch"] == [BATCH_MAX]
    assert seen["group"] == [1] and counts["solved"] == 5  # 5 groups: one batch of 4 + one single


def test_scheduler_does_not_batch_when_off(tmp_path, monkeypatch):
    seen, counts = _run_sched(tmp_path, monkeypatch, "0")
    assert seen["batch"] == [] and len(seen["group"]) == 5 and counts["solved"] == 5


def test_batched_unit_cost_goes_to_the_first_settled_member(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_BATCH_2A", "1")
    tasks = tmp_path / "tasks"
    _math(tasks, "math_a", "A warehouse starts with 10 boxes. It loses 3 boxes to a flood.")
    _math(tasks, "math_b", "A warehouse starts with 20 boxes. It loses 5 boxes to a fire.")

    class OneCall:
        async def invoke(self, messages, **kw):
            assert "specs" in messages[0]["content"]
            return SimpleNamespace(content=batch_reply(SPEC, SPEC),
                                   usage_metadata=SimpleNamespace(input_tokens=200, output_tokens=30))

    client = _client(OneCall())
    summary = scheduler.run(tasks, 100000, client)
    assert summary["solved"] == 2
    recs = {r["task_id"]: r for r in _traj(tmp_path) if r["phase"] == 2}
    owner = [r for r in recs.values() if r["calls"]]
    assert len(owner) == 1 and owner[0]["cost"] == 50 and len(owner[0]["calls"]) == 1
    assert sum(r["cost"] for r in recs.values()) == pytest.approx(client.ledger.spent) == 50


# ------------------------------------------------------------------ fix round 1: alignment and cancellation
def test_swapped_specs_with_wrong_index_fall_back():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    receive = {"start": "{N1}", "steps": ["x+{N2}"]}
    client = FakeClient(batch_reply(receive, SPEC, idx=[2, 1]), json.dumps(SPEC), json.dumps(receive))
    out = asyncio.run(solve_batch("math", items, client))
    assert len(client.calls) == 3 and out["a"].answer == "93" and out["b"].answer == "55"


def test_duplicate_index_makes_both_elements_fall_back():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    client = FakeClient(batch_reply(SPEC, SPEC, idx=[1, 1]), json.dumps(SPEC), json.dumps(SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert len(client.calls) == 3 and out["a"].answer == "93" and out["b"].answer == "45"


def test_missing_index_and_foreign_placeholder_fall_back():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    foreign = {"start": "{N1}", "steps": ["x-{N2}", "x-{N9}"]}
    reply = json.dumps({"specs": [{"spec": SPEC}, {"i": 2, "spec": foreign}]})
    client = FakeClient(reply, json.dumps(SPEC), json.dumps(SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert len(client.calls) == 3 and out["a"].answer == "93" and out["b"].answer == "45"


def test_mixed_good_and_bad_elements_only_bad_refetch():
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire"), _item("c", 9, 2, "storm")]
    client = FakeClient(batch_reply(SPEC, SPEC, SPEC, idx=[1, 3, 3]), json.dumps(SPEC), json.dumps(SPEC))
    out = asyncio.run(solve_batch("math", items, client))
    assert len(client.calls) == 3 and [out[k].answer for k in "abc"] == ["93", "45", "7"]


def test_finished_groups_are_delivered_before_a_cancellation():
    class Slow(FakeClient):
        async def chat(self, model, messages, max_tokens, **extra):
            if len(self.calls) >= 1:  # the batch call is answered; the fallback of the second group hangs
                self.calls.append({})
                await asyncio.sleep(30)
            return await super().chat(model, messages, max_tokens, **extra)

    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    got = {}
    client = Slow(batch_reply(SPEC, "oops"))

    async def go():
        await asyncio.wait_for(solve_batch("math", items, client, on_group=lambda m, r: got.update(r)), 0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert got["a"].answer == "93" and "b" not in got


def test_batch_gather_error_does_not_leave_siblings_running(monkeypatch):
    import harness.json_route as jr
    items = [_item("a", 100, 7, "flood"), _item("b", 50, 5, "fire")]
    done = []
    real = jr.solve_group

    async def boom(task_type, template, members, client, spec=None):
        if members[0][0].task_id == "a":
            raise RuntimeError("settle failed")
        await asyncio.sleep(0.05)
        r = await real(task_type, template, members, client, spec=spec)
        done.append("b")
        return r
    monkeypatch.setattr(jr, "solve_group", boom)
    client = FakeClient(batch_reply(SPEC, SPEC))

    async def run():
        with pytest.raises(RuntimeError):
            await solve_batch("math", items, client)
        at_raise = list(done)
        await asyncio.sleep(0.2)
        assert done == at_raise  # nothing keeps running after solve_batch has raised
    asyncio.run(run())

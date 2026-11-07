import asyncio
import json
import shutil
from pathlib import Path

import pytest

from harness.json_route import fill, request_spec, solve_group
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.tasks import Task
from harness.templates import templatize

FIX = Path(__file__).parent / "fixtures" / "examples"


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra": extra})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=5, cost=6)


def _math_task(tid: str, instruction: str) -> Task:
    return Task(tid, "math", instruction, 1.0, "exact_int", Path("."), Path("."))


def _members(tasks):
    return [(t, templatize(t.instruction)[1]) for t in tasks]


def _theft(start, loss):
    return (f"A warehouse starts with {start} boxes. It loses {loss} boxes to theft. "
            "How many boxes does it have at the end? "
            "Compute it with the bash tool (python3 -c). Work only inside the current directory.")


def test_fill_casts_whole_number_slots():
    spec = {"start": "{N1}", "ops": [["//", "{N2}"]]}
    assert fill(spec, {"N1": "10", "N2": "3"}) == {"start": 10, "ops": [["//", 3]]}


def test_fill_strings_and_substitution_and_keyerror():
    assert fill({"a": "{S1}", "b": "x {E1} y {N1}"}, {"S1": "w", "E1": "Ann Lee", "N1": "4"}) == \
        {"a": "w", "b": "x Ann Lee y 4"}
    assert fill(7, {}) == 7
    with pytest.raises(KeyError):
        fill("{N9}", {"N1": "1"})
    with pytest.raises(KeyError):
        fill("a {S2}", {"S1": "1"})


def test_one_call_per_group():
    tasks = [_math_task("m1", _theft(1000, 7)), _math_task("m2", _theft(500, 20)),
             _math_task("m3", _theft(90, 1))]
    templates = {templatize(t.instruction)[0] for t in tasks}
    assert len(templates) == 1
    template = templates.pop()
    client = FakeClient(json.dumps({"start": "{N1}", "ops": [["-", "{N2}"]]}))
    out = asyncio.run(solve_group("math", template, _members(tasks), client))
    assert len(client.calls) == 1
    assert client.calls[0]["extra"] == {"response_format": {"type": "json_object"}}
    assert client.calls[0]["max_tokens"] == 250
    assert client.calls[0]["model"] == "qwen3.6-35b-a3b-noreason"
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": template}
    assert [out[k].answer for k in ("m1", "m2", "m3")] == ["993", "480", "89"]
    assert {r.status for r in out.values()} == {"solved"}


def test_bad_json_retry_then_unparsed():
    tasks = [_math_task("m1", _theft(1, 2)), _math_task("m2", _theft(5, 6))]
    template = templatize(tasks[0].instruction)[0]
    client = FakeClient("not json", "still not json")
    out = asyncio.run(solve_group("math", template, _members(tasks), client))
    assert len(client.calls) == 2
    assert "not json" in json.dumps(client.calls[1]["messages"])
    assert {r.status for r in out.values()} == {"unparsed"}
    assert set(out) == {"m1", "m2"}


def test_bad_json_then_good_retry():
    client = FakeClient("oops", '{"start": 1, "ops": []}')
    assert asyncio.run(request_spec("math", "t", client)) == {"start": 1, "ops": []}
    assert len(client.calls) == 2


def test_non_object_json_is_invalid():
    client = FakeClient("[1]", "5")
    assert asyncio.run(request_spec("math", "t", client)) is None


def test_fenced_json_accepted():
    client = FakeClient('```json\n{"start": 2, "ops": []}\n```')
    assert asyncio.run(request_spec("math", "t", client)) == {"start": 2, "ops": []}


def test_impossible_is_infeasible():
    tasks = [_math_task("m1", _theft(1, 2))]
    template = templatize(tasks[0].instruction)[0]
    out = asyncio.run(solve_group("math", template, _members(tasks), FakeClient('{"impossible": true}')))
    assert out["m1"].status == "infeasible"


def test_unavailable_makes_group_unparsed_and_budget_propagates():
    tasks = [_math_task("m1", _theft(1, 2))]
    template = templatize(tasks[0].instruction)[0]
    out = asyncio.run(solve_group("math", template, _members(tasks), FakeClient(LLMUnavailable("boom"))))
    assert out["m1"].status == "unparsed" and "boom" in out["m1"].note
    with pytest.raises(BudgetExhausted):
        asyncio.run(solve_group("math", template, _members(tasks), FakeClient(BudgetExhausted("x"))))


def test_unsupported_task_type_makes_no_call():
    t = Task("c1", "coding", "do it", 1.0, None, Path("."), Path("."))
    client = FakeClient()
    out = asyncio.run(solve_group("coding", "do it", [(t, {})], client))
    assert out["c1"].status == "unparsed" and client.calls == []


def test_bad_fill_fails_only_itself():
    good = _math_task("m1", _theft(10, 2))
    bad = _math_task("m2", _theft(10, 2))
    template = templatize(good.instruction)[0]
    members = [(good, templatize(good.instruction)[1]), (bad, {"N1": "10"})]  # N2 missing
    client = FakeClient('{"start": "{N1}", "ops": [["-", "{N2}"]]}')
    out = asyncio.run(solve_group("math", template, members, client))
    assert out["m1"].status == "solved" and out["m1"].answer == "8"
    assert out["m2"].status == "unparsed"


def _search_task(tid: str, entity: str, ws: Path) -> Task:
    ins = ("The directory kb/ contains a small knowledge base of markdown files. "
           f"Which dog does {entity}'s kennel keep? Work only inside the current directory.")
    return Task(tid, "search", ins, 1.0, "exact_str", ws.parent, ws)


def test_one_member_fails_only_itself(tmp_path):
    ws = tmp_path / "workspace"
    shutil.copytree(FIX / "search_0001" / "workspace", ws)
    from harness.solvers.search_solver import load_kb
    kb = load_kb(ws / "kb")
    name = next(n for n, f in kb.items() if "Born" in f)
    expected = kb[name]["Born"]
    t1 = _search_task("s1", name, ws)
    t2 = _search_task("s2", "Zzyzx Nobody", ws)
    template = templatize(t1.instruction)[0]
    assert template == templatize(t2.instruction)[0]
    client = FakeClient('{"entity": "{E1}", "path": [], "field": "Born"}')
    out = asyncio.run(solve_group("search", template, _members([t1, t2]), client))
    assert out["s1"].status == "solved" and out["s1"].answer == expected
    assert out["s2"].status == "infeasible"
    assert len(client.calls) == 1


def test_data_group():
    ws = FIX / "data_analysis_0002" / "workspace"
    ins = json.loads((FIX / "data_analysis_0002" / "task.json").read_text(encoding="utf-8"))["instruction"]
    t = Task("d1", "data_analysis", ins, 1.0, "exact_int", ws.parent, ws)
    template, slots = templatize(ins)
    spec = {"file": "sales.csv", "metric": "sum_units", "product": "{S1}", "region": None,
            "month_from": "{N2}", "month_to": "{N3}"}
    client = FakeClient(json.dumps(spec))
    out = asyncio.run(solve_group("data_analysis", template, [(t, slots)], client))
    assert out["d1"].answer == "158", (out, slots)

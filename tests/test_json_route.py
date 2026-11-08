from harness import config
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
    return (f"A warehouse starts with {start} boxes. It loses {loss} boxes to a flood. "
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
    assert client.calls[0]["model"] == config.MODEL_FAST
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


def test_impossible_is_unparsed():
    tasks = [_math_task("m1", _theft(1, 2))]
    template = templatize(tasks[0].instruction)[0]
    out = asyncio.run(solve_group("math", template, _members(tasks), FakeClient('{"impossible": true}')))
    assert out["m1"].status == "unparsed"


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


def test_product_operand_list_collapsed():
    def ship(start, n, d):
        return (f"A warehouse starts with {start} boxes. It ships {n} boxes per day for {d} days. "
                "How many boxes does it have at the end? Work only inside the current directory.")
    tasks = [_math_task("m1", ship(1000, 7, 3)), _math_task("m2", ship(500, 20, 4))]
    template = templatize(tasks[0].instruction)[0]
    assert "{N2}" in template and "{N3}" in template
    spec = {"start": "{N1}", "ops": [["-", ["{N2}", "{N3}"]]]}
    out = asyncio.run(solve_group("math", template, _members(tasks), FakeClient(json.dumps(spec))))
    assert [out[k].answer for k in ("m1", "m2")] == ["979", "420"]


def test_steps_spec_with_placeholders_inside_expression():
    def ship(start, n, d):
        return (f"A warehouse starts with {start} boxes. It ships {n} boxes per day for {d} days. "
                "How many boxes does it have at the end? Work only inside the current directory.")
    tasks = [_math_task("m1", ship(1000, 7, 3)), _math_task("m2", ship(500, 20, 4))]
    template = templatize(tasks[0].instruction)[0]
    spec = {"start": "{N1}", "steps": ["x-{N2}*{N3}", "x-x//4"]}
    out = asyncio.run(solve_group("math", template, _members(tasks), FakeClient(json.dumps(spec))))
    assert [out[k].answer for k in ("m1", "m2")] == [str(979 - 979 // 4), str(420 - 420 // 4)]


def test_math_schema_has_steps_example_and_one_line_rule():
    from harness.json_route import _system_prompt
    p = _system_prompt("math")
    assert '"steps"' in p and "x-{N2}*{N3}" in p
    assert "one line" in p.lower()


def test_data_schema_lists_top_metrics():
    from harness.json_route import _system_prompt
    from harness.solvers.data_solver import _TOP_NAMES
    p = _system_prompt("data_analysis")
    for name in _TOP_NAMES:
        assert name in p


def test_data_top_metric_via_2a():
    ws = FIX / "data_analysis_0002" / "workspace"
    ins = json.loads((FIX / "data_analysis_0002" / "task.json").read_text(encoding="utf-8"))["instruction"]
    # a "which product" question is checked exact_str (gen_tasks); exact_int would fail the shape check
    t = Task("d1", "data_analysis", ins, 1.0, "exact_str", ws.parent, ws)
    spec = {"file": "sales.csv", "metric": "top_product_sum_units", "product": None, "region": None,
            "month_from": None, "month_to": None}
    out = asyncio.run(solve_group("data_analysis", templatize(ins)[0], [(t, {})], FakeClient(json.dumps(spec))))
    assert out["d1"].status == "solved" and out["d1"].answer == "widget", out


def _data_task(check="exact_int"):
    ws = FIX / "data_analysis_0002" / "workspace"
    ins = json.loads((FIX / "data_analysis_0002" / "task.json").read_text(encoding="utf-8"))["instruction"]
    return Task("d1", "data_analysis", ins, 1.0, check, ws.parent, ws), ins


def test_data_max_units_via_2a():
    t, ins = _data_task()
    spec = {"file": "sales.csv", "metric": "max_units", "product": None, "region": None,
            "month_from": None, "month_to": None}
    out = asyncio.run(solve_group("data_analysis", templatize(ins)[0], [(t, {})], FakeClient(json.dumps(spec))))
    assert out["d1"].status == "solved" and out["d1"].answer.isdigit(), out


def test_data_decimals_via_2a():
    t, ins = _data_task("exact_str")
    spec = {"file": "sales.csv", "metric": "avg_price", "product": None, "region": None,
            "month_from": None, "month_to": None, "decimals": 3}
    out = asyncio.run(solve_group("data_analysis", templatize(ins)[0], [(t, {})], FakeClient(json.dumps(spec))))
    assert out["d1"].status == "solved", out
    assert len(out["d1"].answer.split(".")[1]) == 3, out["d1"].answer


def test_data_schema_lists_new_metrics():
    from harness.json_route import _system_prompt
    p = _system_prompt("data_analysis")
    assert "max_units" in p and "min_units" in p and "decimals" in p
    assert "six keys" not in p


def test_arbitrate_models_come_from_config():
    from harness import arbitrate, config
    assert arbitrate.ARBITRATE_MODELS == config.ARBITRATE_MODELS == (config.MODEL_FAST, config.MODEL_ALT)


def test_model_names_come_from_config():
    from harness import config, json_route, llm_routes
    assert json_route.MODEL == config.MODEL_FAST
    assert llm_routes.MODEL_FAST == config.MODEL_FAST
    assert config.MODEL_ALT and config.MODEL_REASON


def test_fill_leading_zero_slot_in_expression():
    assert fill("x-{N1}*2", {"N1": "007"}) == "x-7*2"
    assert fill("{N1}", {"N1": "007"}) == 7


def test_rules_allow_placeholders_in_expressions_and_unknown_metric_impossible():
    from harness.json_route import _system_prompt
    assert "inside the expression" in _system_prompt("math")
    assert "impossible" in _system_prompt("data_analysis").split("one of:")[1]


# ------------------------------------------------------------------ search: hops schema
def _kb(tmp_path):
    ws = tmp_path / "workspace"
    shutil.copytree(FIX / "search_0001" / "workspace", ws)
    from harness.solvers.search_solver import load_kb
    return ws, load_kb(ws / "kb")


def _ask(question: str, ws: Path, tid: str = "s1") -> Task:
    ins = ("The directory kb/ contains a small knowledge base of markdown files. "
           f"{question} Work only inside the current directory.")
    return Task(tid, "search", ins, 1.0, "exact_str", ws.parent, ws)


def test_search_hops_become_path_and_field():
    reply = '{"entity": "{E1}", "hops": ["Mentor", "Employer", "Headquarters"]}'
    spec = asyncio.run(request_spec("search", "t", FakeClient(reply)))
    assert spec == {"entity": "{E1}", "path": ["Mentor", "Employer"], "field": "Headquarters"}


def test_search_single_hop_has_empty_path_and_names_are_canonical():
    spec = asyncio.run(request_spec("search", "t", FakeClient('{"entity": "{E1}", "hops": ["mentor"]}')))
    assert spec == {"entity": "{E1}", "path": [], "field": "Mentor"}


def test_search_old_path_field_reply_still_accepted():
    reply = '{"entity": "{E1}", "path": ["Employer"], "field": "Headquarters"}'
    spec = asyncio.run(request_spec("search", "t", FakeClient(reply)))
    assert spec == {"entity": "{E1}", "path": ["Employer"], "field": "Headquarters"}


@pytest.mark.parametrize("bad", ['{"entity": "{E1}", "hops": []}',
                                 '{"entity": "{E1}", "hops": ["Mentor", "Name"]}',
                                 '{"entity": "{E1}", "hops": "Mentor"}',
                                 '{"entity": "{E1}", "hops": [3]}'])
def test_search_bad_hops_are_an_invalid_spec(bad):
    client = FakeClient(bad, bad)
    assert asyncio.run(request_spec("search", "t", client)) is None
    assert len(client.calls) == 2
    assert "hops" in client.calls[1]["messages"][-1]["content"]


def test_search_bad_hops_then_good_retry():
    client = FakeClient('{"entity": "{E1}", "hops": ["Name"]}', '{"entity": "{E1}", "hops": ["Born"]}')
    assert asyncio.run(request_spec("search", "t", client)) == {"entity": "{E1}", "path": [], "field": "Born"}


def test_hops_only_for_search():
    reply = '{"start": 1, "steps": [], "hops": []}'
    assert asyncio.run(request_spec("math", "t", FakeClient(reply))) == {"start": 1, "steps": [], "hops": []}


def test_person_mentoring_x_is_one_mentor_hop(tmp_path):
    ws, kb = _kb(tmp_path)
    name = next(n for n, f in kb.items() if f.get("Mentor") in kb)
    t = _ask(f"Find the person mentoring {name}.", ws)
    template, slots = templatize(t.instruction)
    client = FakeClient('{"entity": "{E1}", "hops": ["Mentor"]}')
    out = asyncio.run(solve_group("search", template, [(t, slots)], client))
    assert out["s1"].status == "solved" and out["s1"].answer == kb[name]["Mentor"]
    assert len(client.calls) == 1


def test_mentor_of_mentor_is_two_hops(tmp_path):
    ws, kb = _kb(tmp_path)
    name = next(n for n, f in kb.items() if kb.get(f.get("Mentor"), {}).get("Mentor") in kb)
    t = _ask(f"Who is the mentor of the mentor of {name}?", ws)
    template, slots = templatize(t.instruction)
    out = asyncio.run(solve_group("search", template, [(t, slots)],
                                  FakeClient('{"entity": "{E1}", "hops": ["Mentor", "Mentor"]}')))
    assert out["s1"].answer == kb[kb[name]["Mentor"]]["Mentor"]


def test_search_prompt_speaks_hops():
    from harness.json_route import _system_prompt
    p = _system_prompt("search")
    assert '"hops"' in p and '"path"' not in p
    assert "Find the person mentoring {E1}" in p and '["Mentor"]' in p
    assert '["Mentor","Mentor"]' in p
    for word in ("mentoring", "payroll", "works for", "head office", "based", "birth", "established"):
        assert word in p


def test_search_impossible_with_hops_stays_impossible():
    spec = asyncio.run(request_spec("search", "t", FakeClient('{"impossible": true, "hops": []}')))
    assert spec == {"impossible": True, "hops": []}


def test_search_impossible_with_hops_makes_group_unparsed(tmp_path):
    ws, kb = _kb(tmp_path)
    t = _ask("Who mentors Ann Lee?", ws)
    template, slots = templatize(t.instruction, "search")
    out = asyncio.run(solve_group("search", template, [(t, slots)], FakeClient('{"impossible": true, "hops": []}')))
    assert out["s1"].status == "unparsed" and out["s1"].note == "model: spec impossible"


def test_search_prompt_allows_company_or_literal_entity():
    from harness.json_route import _system_prompt
    p = _system_prompt("search")
    assert "person or company" in p and "literal name" in p


# ------------------------------------------------------------------ search: reverse relations go to 2b
# the 7 phrasings of the final review (4 named there, 3 added by the lead) first, then more variants
REVIEW7 = ["Whom does {n} mentor?", "Who is {n} mentoring?", "Who is mentored by {n}?", "Who is {n}'s mentee?",
           "Name the person that {n} mentors.", "Which individual has {n} as their mentor?", "Who works for {c}?"]
REVERSE = REVIEW7 + ["Name the mentee of {n}.", "Which person has {c} as her employer?",
           "Which person does {n} mentor?", "Who are {n}'s mentees?", "Tell me which person {n} mentors.",
           "Who works at {c}?", "Name an employee of {c}.", "List the employees of {c}.",
           "Which person is employed by {c}?", "Whom does {c} employ?"]
FORWARD = ["Who is the mentor of {n}?", "Which individual is {n}'s mentor?", "Who does {n} have as a mentor?", "Find the person mentoring {n}.", "Which person mentors {n}?",
           "Who mentored {n}?", "Who has been mentoring {n}?", "{n} is mentored by which person?",
           "Which firm employs {n}?", "Which company does {n} work for?", "Who does {n} work for?",
           "Which organisation has {n} on its payroll?", "Where is the employer of the mentor of {n} based?",
           "Which company is {n} employed by?", "In which year was the mentor of {n} born?"]


@pytest.mark.parametrize("q", REVERSE)
def test_reverse_relation_is_unparsed_without_a_call(q, tmp_path):
    ws, kb = _kb(tmp_path)
    t = _ask(q.format(n="Ann Lee", c="Arktika Labs"), ws)
    template, slots = templatize(t.instruction, "search")
    client = FakeClient()
    out = asyncio.run(solve_group("search", template, [(t, slots)], client))
    assert out["s1"].status == "unparsed" and "reverse relation" in out["s1"].note
    assert client.calls == []


@pytest.mark.parametrize("q", FORWARD)
def test_forward_relation_is_not_flagged(q):
    from harness.json_route import _reverse_relation
    template, _ = templatize(q.format(n="Ann Lee"), "search")
    assert _reverse_relation(template) is None, template


def test_search_prompt_sends_reverse_questions_to_impossible():
    from harness.json_route import _system_prompt
    p = _system_prompt("search")
    assert "reverse" in p and "whom X mentors" in p


# ------------------------------------------------------------------ search: start entity not in the task
def test_start_entity_not_found_is_final_only_for_a_task_value(tmp_path):
    ws, kb = _kb(tmp_path)
    t = _ask("In which year was Zzyzx Nobody born?", ws)
    template, slots = templatize(t.instruction, "search")
    out = asyncio.run(solve_group("search", template, [(t, slots)],
                                  FakeClient('{"entity": "{E1}", "hops": ["Born"]}')))
    assert out["s1"].status == "infeasible" and "start entity not found" in out["s1"].note
    out = asyncio.run(solve_group("search", template, [(t, slots)],
                                  FakeClient('{"entity": "Zzyzx Nobody", "hops": ["Born"]}')))
    assert out["s1"].status == "infeasible"  # literal name, but it is the task's own value
    out = asyncio.run(solve_group("search", template, [(t, slots)],
                                  FakeClient('{"entity": "Zzyzx Nobodi", "hops": ["Born"]}')))
    assert out["s1"].status == "unparsed" and "not a task value" in out["s1"].note


@pytest.mark.parametrize("q", ["Whose mentor is {n}?", "{n} is the mentor of whom?", "{n} was a mentor of which person?",
                               "Whose mentor is the mentor of {n}?", "Whose employer is {n}?"])
def test_reverse_mentor_of_whom(q):
    from harness.json_route import _reverse_relation
    assert _reverse_relation(templatize(q.format(n="Ann Lee"), "search")[0]) is not None


@pytest.mark.parametrize("q", ["Who is the mentor of {n}?",
                               "Who is the mentor of the mentor of {n}?", "The mentor of {n} is who?"])
def test_forward_mentor_of_is_not_flagged(q):
    from harness.json_route import _reverse_relation
    assert _reverse_relation(templatize(q.format(n="Ann Lee"), "search")[0]) is None

import json
from pathlib import Path

from harness.tasks import Task
from harness.templates import group, templatize

FIX = Path(__file__).parent / "fixtures" / "examples"


def _instr(name: str) -> str:
    return json.loads((FIX / name / "task.json").read_text(encoding="utf-8"))["instruction"]


def _task(task_id: str, task_type: str, instruction: str) -> Task:
    return Task(task_id, task_type, instruction, 1.0, None, Path("."), Path("."))


def test_same_phrasing_same_template():
    t1, s1 = templatize(_instr("math_0004"))
    other = _instr("math_0004").replace("1669", "2000").replace("36", "12").replace("87", "55")
    t2, s2 = templatize(other)
    assert t1 == t2
    assert s1["N1"] == "1669"
    assert s2["N1"] == "2000"
    assert "1669" not in t1


def test_names_and_quotes_become_slots():
    t, slots = templatize(_instr("search_0005"))
    assert "{E1}" in t
    assert slots["E1"] == "Sergei Garcia"
    assert "Sergei" not in t
    t, slots = templatize(_instr("data_analysis_0002"))
    assert slots["S1"] == "widget"
    assert "widget" not in t
    assert "{S1}" in t


def test_repeated_value_shares_slot_and_decimals_stay():
    t, slots = templatize("Ann Lee meets Ann Lee and 5 of 5 at 3.5 percent in sales2.csv.")
    assert slots == {"E1": "Ann Lee", "N1": "5"}
    assert "3.5" in t and "sales2.csv" in t


def test_leading_stopword_not_part_of_entity():
    t, slots = templatize("Then Sergei Garcia is asked. The Big Cat sleeps.")
    assert slots.get("E1") == "Sergei Garcia"
    assert slots["E2"] == "Big Cat" and t.startswith("Then {E1}")


def test_apostrophe_is_not_a_quote():
    t, slots = templatize("Where is Ann Lee's employer and Bo Li's mentor?")
    assert "S1" not in slots
    assert slots == {"E1": "Ann Lee", "E2": "Bo Li"}


def test_group_keys_by_type_and_template():
    a = _task("a", "math", _instr("math_0004"))
    b = _task("b", "math", _instr("math_0004").replace("1669", "5"))
    c = _task("c", "search", _instr("search_0001"))
    g = group([a, b, c])
    assert len(g) == 2
    math_key = next(k for k in g if k[0] == "math")
    assert [t.task_id for t, _ in g[math_key]] == ["a", "b"]

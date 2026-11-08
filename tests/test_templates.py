import json
import pytest
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
    assert slots == {"E1": "Ann Lee", "N1": "5", "N2": "5"}
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


def test_equal_numbers_share_template_with_different_numbers():
    t1, s1 = templatize("A warehouse starts with 3 boxes. It ships 3 boxes per day for 3 days.")
    t2, _ = templatize("A warehouse starts with 9 boxes. It ships 7 boxes per day for 3 days.")
    assert t1 == t2 and s1 == {"N1": "3", "N2": "3", "N3": "3"}


@pytest.mark.parametrize("name", ["Nikolai-Felix Lindqvist", "Quinn D. Muller", "Karim Petrov-Sato",
                                  "Farid-Farid Gomez", "Sofia Smith"])
def test_hyphenated_names_and_initials_are_one_slot(name):
    t, slots = templatize(f"Find the person mentoring {name}. Work only inside the current directory.", "search")
    assert slots == {"E1": name}
    assert t == "Find the person mentoring {E1}."


def test_hyphen_names_share_template_but_relations_do_not():
    a, _ = templatize("Who mentors the mentor of Nikolai-Felix Lindqvist?", "search")
    b, _ = templatize("Who mentors the mentor of Ann Lee?", "search")
    c, _ = templatize("Who employs the mentor of Quinn D. Muller?", "search")
    assert a == b != c
    assert "-" not in a and "D." not in c


def test_name_before_lowercase_hyphen_word_stays_short():
    t, slots = templatize("Ann Lee-based staff and Bo Li's mentor.", "search")
    assert slots == {"E1": "Ann Lee", "E2": "Bo Li"} and t == "{E1}-based staff and {E2}'s mentor."


def _tank(target: str) -> str:
    return (f"Tank A has 5 liters and Tank B has 7 liters. Move 2 liters into {target}. What does Tank B hold? "
            "Work only inside the current directory.")


def test_letter_after_word_does_not_glue_math_tasks():
    a, b = _task("a", "math", _tank("Tank A")), _task("b", "math", _tank("Tank B"))
    g = group([a, b])
    assert len(g) == 2
    for (_, template), members in g.items():
        assert "What does" in template
        assert all("." not in v for _, slots in members for v in slots.values())


def test_letter_after_word_keeps_data_question_intact():
    ins = "Compute the total revenue of Product A. Use the file sales.csv."
    for kind in ("data_analysis", "math", "search"):
        t, slots = templatize(ins, kind)
        assert "Use the file" in t, (kind, t)
        assert all(v == "Product A" or ("." not in v and "Use" not in v) for v in slots.values()), (kind, slots)
    t, slots = templatize("Option B. costs 7.", "math")
    assert all("." not in v for v in slots.values())


def test_search_initial_only_inside_a_name():
    t, slots = templatize("Who mentors Quinn D. Muller? It was Plan B. Which one? Then Ann C. Then.", "search")
    assert slots == {"E1": "Quinn D. Muller"}
    assert "Plan B. Which one?" in t and "Ann C. Then." in t


def test_initials_and_hyphens_only_for_search():
    for kind in ("math", "data_analysis"):
        _, slots = templatize("Ask about Quinn D. Muller and Nikolai-Felix Lindqvist.", kind)
        assert all("." not in v and "-" not in v for v in slots.values()), (kind, slots)
    _, slots = templatize("Ask about Quinn D. Muller and Nikolai-Felix Lindqvist.", "search")
    assert set(slots.values()) >= {"Quinn D. Muller", "Nikolai-Felix Lindqvist"}

import json
from pathlib import Path

from harness.solvers.common import strip_boilerplate
from harness.solvers.math_solver import execute, parse, solve
from harness.tasks import Task

FIX = Path(__file__).parent / "fixtures" / "examples"


def _task(instruction: str) -> Task:
    return Task("t", "math", instruction, 1.0, "exact_int", Path("."), Path("."))


def _fixture(name: str) -> dict:
    return json.loads((FIX / name / "task.json").read_text(encoding="utf-8"))


def test_examples():
    for name in ("math_0000", "math_0004"):
        data = _fixture(name)
        res = solve(_task(data["instruction"]))
        assert res.status == "solved"
        assert res.answer == data["expected"]


def test_unknown_sentence_is_unparsed():
    instr = _fixture("math_0004")["instruction"].replace(
        "How many boxes", "It loses 5 boxes to theft. How many boxes")
    assert solve(_task(instr)).status == "unparsed"


def test_split_into_zero_is_infeasible():
    instr = ("A warehouse starts with 10 boxes. Then the stock is split equally into 0 "
             "warehouses and it keeps one share (rounded down). How many boxes does it have at the end?")
    assert solve(_task(instr)).status == "infeasible"


def test_tolerates_case_and_spacing():
    instr = "a WAREHOUSE starts with  10 boxes.  How many boxes does it have at the end?"
    res = solve(_task(instr))
    assert res.status == "solved" and res.answer == "10"


def test_parse_gives_spec():
    assert parse(_fixture("math_0004")["instruction"]) == {
        "start": 1669, "ops": [["-", 108], ["+", 174]]}


def test_missing_start_or_question_is_unparsed():
    assert solve(_task("It ships 3 boxes per day for 2 days. How many boxes does it have at the end?")).status == "unparsed"
    assert solve(_task("A warehouse starts with 5 boxes.")).status == "unparsed"


def test_execute_rejects_bad_spec():
    assert execute({"start": "x", "ops": []}).status == "unparsed"
    assert execute({"start": 5, "ops": [["**", 2]]}).status == "unparsed"
    assert execute({"start": 5, "ops": [["//", 0]]}).status == "infeasible"
    assert execute({"start": 5}).status == "unparsed"


def test_strip_boilerplate_all_task_types():
    for name in ("coding_0003", "data_analysis_0002", "math_0000", "search_0001"):
        out = strip_boilerplate(_fixture(name)["instruction"])
        for marker in ("Compute it", "Use bash", "Use the file tools", "Work only", "When you are done", "answer.txt"):
            assert marker not in out, (name, marker)
    assert strip_boilerplate(_fixture("search_0001")["instruction"]) == (
        "The directory kb/ contains a small knowledge base of markdown files. In which year was Irina Smith born?")


def test_execute_validates_whole_spec_before_running():
    assert execute({"start": 5, "ops": [["//", 0], ["foo", 1]]}).status == "unparsed"


def test_sentences_after_question_are_unparsed():
    q = "A warehouse starts with 5 boxes. How many boxes does it have at the end?"
    assert solve(_task(q + " It ships 5 boxes per day for 1 days.")).status == "unparsed"
    assert solve(_task(q + " How many boxes does it have at the end?")).status == "unparsed"


def test_solve_never_raises(monkeypatch):
    import harness.solvers.math_solver as ms

    def boom(_):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ms, "parse", boom)
    res = ms.solve(_task("anything"))
    assert res.status == "unparsed" and "kaboom" in res.note

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
        "How many boxes", "It gets repainted 5 times. How many boxes")
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
        "start": 1669, "steps": ["x-108", "x+174"], "nonneg": True}


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


# ---- expression steps
from harness.solvers.math_solver import execute_steps  # noqa: E402


def _steps(start, *steps):
    return execute_steps({"start": start, "steps": list(steps)})


def test_steps_basic_and_expressions():
    r = _steps(1000, "x-7*3", "x-x//4", "x*2", "-x+5", "(x+1)*2")
    x = 1000 - 21
    x = x - x // 4
    x = -(x * 2) + 5
    x = (x + 1) * 2
    assert r.status == "solved" and r.answer == str(x)


def test_steps_floor_division_negative():
    assert _steps(-7, "x//2").answer == "-4"


def test_steps_div_zero_infeasible():
    assert _steps(5, "x//0").status == "infeasible"
    assert _steps(5, "x-5", "10//x").status == "infeasible"


def test_steps_forbidden_nodes_unparsed():
    for bad in ("y+1", "x**2", "x/2", "x%3", "__import__('os')", "x+1.5", "abs(x)", "x if x else 1",
                "[x]", "x<2", "x+True", "lambda: 1", "x;1", "", "x.real", "x+'a'"):
        assert _steps(5, bad).status == "unparsed", bad


def test_steps_overflow_unparsed():
    assert _steps(10**9, "x*x", "x*x").status == "unparsed"
    assert _steps(10**18 + 1, "x").status == "unparsed"


def test_steps_huge_literal_unparsed():
    assert _steps(1, "9" * 199).status == "unparsed"
    assert _steps(1, "-" + "9" * 199).status == "unparsed"
    assert _steps(1, "x+1", 10**30).status == "unparsed"


def test_steps_whole_placeholder_int_sets_x():
    assert _steps(5, 9, "x+1").answer == "10"


def test_steps_bad_spec_unparsed():
    assert execute_steps({"start": "5", "steps": ["x"]}).status == "unparsed"
    assert execute_steps({"start": 5, "steps": "x"}).status == "unparsed"
    pass
    assert execute_steps({"start": True, "steps": ["x"]}).status == "unparsed"


def test_execute_dispatches_steps_and_keeps_ops():
    assert execute({"start": 10, "steps": ["x-3"]}).answer == "7"
    assert execute({"start": 10, "ops": [["-", 3]]}).answer == "7"


# ---- round-2 grammar (v2 operations and paraphrases)
import pytest  # noqa: E402

_Q = " How many boxes does it have at the end?"


def _run(*sentences, q=_Q):
    return solve(_task("A warehouse starts with 1000 boxes. " + " ".join(sentences) + q))


@pytest.mark.parametrize("sentence,expected", [
    ("Then the stock is tripled by a big merger.", "3000"),
    ("Then a big merger triples the stock.", "3000"),
    ("A merger with a sister warehouse then doubles the stock.", "2000"),
    ("It loses 90 boxes to theft.", "910"),
    ("Thieves take 90 boxes.", "910"),
    ("A leak ruins 90 boxes, which are thrown away.", "910"),
    ("Water damage destroys 90 boxes.", "910"),
    ("Then one quarter of the stock (rounded down) is moved to another site.", "750"),
    ("Then a quarter of the stock, rounded down, is transferred to another site.", "750"),
    ("Then it sells half of the boxes, rounding the number sold down.", "500"),
    ("It then sells half of the stock, with the number of boxes sold rounded down.", "500"),
    ("Then the stock grows by 25 percent (rounded down).", "1250"),
    ("A seasonal restock adds 25 percent of the current stock, rounded down.", "1250"),
    ("Then 15 percent of the stock (rounded down) spoils.", "850"),
    ("Spoilage removes 15 percent of the stock, rounded down.", "850"),
    ("Every day for 3 days, 10 boxes leave the warehouse.", "970"),
    ("For 3 days in a row, 10 boxes are shipped out daily.", "970"),
    ("4 trucks arrive, each carrying 10 boxes.", "1040"),
    ("The warehouse takes in 4 shipments of 10 boxes apiece.", "1040"),
    ("The stock is then divided evenly among 3 warehouses, and this one retains a single share (rounded down).", "333"),
])
def test_v2_sentences(sentence, expected):
    res = _run(sentence)
    assert res.status == "solved" and res.answer == expected, res


def test_rounding_semantics_with_odd_stock():
    t = "A warehouse starts with 1001 boxes. "
    assert solve(_task(t + "Then it sells half of the boxes, rounding the number sold down." + _Q)).answer == "501"
    assert solve(_task(t + "Then one quarter of the stock (rounded down) is moved to another site." + _Q)).answer == "751"
    assert solve(_task(t + "Then 15 percent of the stock (rounded down) spoils." + _Q)).answer == "851"
    assert solve(_task(t + "Then the stock grows by 25 percent (rounded down)." + _Q)).answer == "1251"


@pytest.mark.parametrize("q", ["What is the final stock level in boxes?", "How many boxes remain in the end?",
                               "How many boxes are left at the end?"])
def test_question_paraphrases(q):
    assert _run("Then the stock is doubled by merging with a sister warehouse.", q=" " + q).answer == "2000"


@pytest.mark.parametrize("sentence", [
    "Then it sells half of the boxes, rounding the number sold up.",   # wrong direction: not our rule
    "Then it sells half of the boxes.",                                 # no rounding statement
    "Then 15 percent of the stock spoils.",                             # rounding unspecified
    "Then the stock grows by 25 percent.",
    "Thieves take 90 boxes from the neighbour.",
    "Then the stock is quadrupled by a big merger.",
])
def test_unknown_or_underspecified_stays_unparsed(sentence):
    assert _run(sentence).status == "unparsed"


def test_loss_below_zero_is_not_answered():
    assert _run("Thieves take 5000 boxes.").status == "unparsed"
    assert _run("Spoilage removes 150 percent of the stock, rounded down.").status == "unparsed"


def test_mixed_chain_and_zero_split_infeasible():
    r = _run("Then it sells half of the boxes, rounding the number sold down.", "It loses 50 boxes to theft.",
             "Then the stock is tripled by a big merger.")
    assert r.answer == str((500 - 50) * 3)
    assert _run("Then it sells half of the boxes, rounding the number sold down.",
                "Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).",
                ).status == "infeasible"


def test_nonneg_negative_value_is_best_guess_not_answer():
    r = execute({"start": 3, "steps": ["x-5", "x*2"], "nonneg": True})
    assert r.status == "unparsed" and r.extra == {"best_guess": "-4"}


def test_overdrawn_shipment_is_unparsed():
    instr = ("A warehouse starts with 1000 boxes. Every day for 2 days, 1,000 boxes leave the warehouse. "
             "How many boxes does it have at the end?")
    assert solve(_task(instr)).status == "unparsed"
    ok = instr.replace("1,000 boxes leave", "10 boxes leave")
    assert solve(_task(ok)).answer == "980"

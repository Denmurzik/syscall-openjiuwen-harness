import json
from pathlib import Path

import pytest

from harness.result import SolveResult
from harness.solvers.data_solver import execute, parse, solve
from harness.tasks import Task

FIX = Path(__file__).parent / "fixtures" / "examples"
HEADER = "order_id,region,product,month,units,price\n"


def _fixture(name: str) -> dict:
    return json.loads((FIX / name / "task.json").read_text(encoding="utf-8"))


def _task(instruction: str, workspace: Path, check: str | None = "exact_int") -> Task:
    return Task("t", "data_analysis", instruction, 1.0, check, workspace.parent, workspace)


def _fix_task(name: str, instruction: str | None = None) -> Task:
    data = _fixture(name)
    return _task(instruction or data["instruction"], FIX / name / "workspace", data["check"])


def _ws(tmp_path: Path, rows: str) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_text(HEADER + rows, encoding="utf-8")
    return ws


def _q(question: str) -> str:
    return (f"The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "
            f"{question} Use bash with python3 to compute it. Work only inside the current directory. "
            f"When you are done, write ONLY the final answer to the file answer.txt.")


ROWS = ("1,west,widget,1,10,2\n2,west,widget,12,5,3\n3,east,widget,6,100,1\n4,west,gizmo,6,7,4\n")


@pytest.mark.parametrize("name", ["data_analysis_0002", "data_analysis_0006"])
def test_examples(name):
    res = solve(_fix_task(name))
    assert res.status == "solved"
    assert res.answer == _fixture(name)["expected"]


def test_unknown_product_answers_zero():
    instr = _fixture("data_analysis_0002")["instruction"].replace("widget", "doohickey")
    res = solve(_fix_task("data_analysis_0002", instr))
    assert (res.status, res.answer) == ("solved", "0")


def test_missing_file_is_infeasible(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    instr = _fixture("data_analysis_0002")["instruction"]
    assert solve(_task(instr, ws)).status == "infeasible"
    assert solve(_task(instr, tmp_path / "nope")).status == "infeasible"


def test_missing_column_is_infeasible(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_text("order_id,region,product,month,units\n1,west,widget,1,2\n")
    res = solve(_task(_q("What is the average price?"), ws))
    assert res.status == "infeasible"
    res = solve(_task(_q("What is the total revenue?"), ws))
    assert res.status == "infeasible"
    # filter columns are checked too
    (ws / "sales.csv").write_text("order_id,region,product,units,price\n1,w,a,2,3\n")
    res = solve(_task(_q("How many units were sold in month 1?"), ws))
    assert res.status == "infeasible"


def test_average_rounds_half_up(tmp_path):
    ws = _ws(tmp_path, "1,w,a,1,2,1\n2,w,a,1,3,1\n")
    res = solve(_task(_q("What is the average units?"), ws))
    assert (res.status, res.answer) == ("solved", "3")
    res = solve(_task(_q("What is the average units?"), ws, check="exact_str"))
    assert res.answer == "2.5"


def test_average_empty_answers_zero(tmp_path):
    ws = _ws(tmp_path, ROWS)
    res = solve(_task(_q("What is the average price of product 'nothing'?"), ws))
    assert (res.status, res.answer) == ("solved", "0")


def test_region_filter(tmp_path):
    ws = _ws(tmp_path, ROWS)
    q = _q("How many units of product 'widget' were sold in region 'west' in months 1 through 12 inclusive?")
    res = solve(_task(q, ws))
    assert (res.status, res.answer) == ("solved", "15")


def test_case_insensitive_values(tmp_path):
    ws = _ws(tmp_path, "1,West,Widget,3,4,1\n")
    res = solve(_task(_q("How many units of product 'widget' were sold in region 'WEST'?"), ws))
    assert res.answer == "4"


def test_revenue_orders_price(tmp_path):
    ws = _ws(tmp_path, ROWS)
    assert solve(_task(_q("What is the total revenue?"), ws)).answer == str(20 + 15 + 100 + 28)
    assert solve(_task(_q("How many orders were placed in region 'west'?"), ws)).answer == "3"
    assert solve(_task(_q("What is the average price of product 'widget'?"), ws)).answer == "2"


def test_month_filters(tmp_path):
    ws = _ws(tmp_path, ROWS)
    assert solve(_task(_q("How many units were sold in month 6?"), ws)).answer == "107"
    assert solve(_task(_q("How many units were sold in months 13 through 14 inclusive?"), ws)).answer == "0"
    assert solve(_task(_q("How many units were sold in months 5 through 2 inclusive?"), ws)).answer == "0"


def test_unrecognized_question_is_unparsed():
    assert parse(_q("What is the median units per order?")).status == "unparsed"


def test_partial_parse_is_unparsed():
    # extra unknown clause must not yield a partial answer
    assert parse(_q("How many units of product 'widget' were sold on weekdays in month 3?")).status == "unparsed"
    assert parse(_q("How many units of product 'a' of product 'b' were sold?")).status == "unparsed"
    assert parse("The file sales.csv has columns a,b. How many units were sold? Also tell a joke.").status == "unparsed"


def test_parse_gives_spec():
    spec = parse(_fixture("data_analysis_0002")["instruction"])
    assert spec == {"file": "sales.csv", "metric": "sum_units", "product": "widget",
                    "region": None, "month_from": 7, "month_to": 8}


def test_execute_validates_spec(tmp_path):
    ws = _ws(tmp_path, ROWS)
    good = {"file": "sales.csv", "metric": "sum_units", "product": None, "region": None,
            "month_from": None, "month_to": None}
    assert execute(good, ws, "exact_int").answer == "122"
    bad_specs = [
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "region"},
        {**good, "metric": "median"},
        {**good, "file": "../sales.csv"},
        {**good, "file": "/etc/passwd"},
        {**good, "product": 5},
        {**good, "month_from": 1},
        {**good, "month_from": True, "month_to": 2},
        {**good, "month_from": "1", "month_to": 2},
        "nonsense",
        None,
    ]
    for spec in bad_specs:
        res = execute(spec, ws, "exact_int")
        assert isinstance(res, SolveResult) and res.status == "unparsed", spec


def test_solve_never_raises(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_bytes(HEADER.encode() + b"1,w,a,x,y,z\n")
    res = solve(_task(_q("How many units were sold?"), ws))
    assert res.status in ("unparsed", "infeasible")


@pytest.mark.parametrize("question,expected", [
    ("What is the revenue?", "163"),
    ("What was the revenue?", "163"),
    ("What is the total revenue?", "163"),
    ("What is the total units?", "122"),
    ("How many units were ordered?", "122"),
    ("How many units were sold in months 1 to 6?", "117"),
    ("How many units were sold between months 1 and 6?", "117"),
    ("How many units were sold between months 1 and 6 inclusive?", "117"),
    ("How many units were sold from month 6 to month 12?", "112"),
    ("How many units were sold from months 6 to 12?", "112"),
    ("How many units were sold in the west region?", "22"),
    ("How many units were sold in the 'west' region?", "22"),
    ("How many units were sold in region west?", "22"),
    ("How many units were sold in region west in month 6?", "7"),
])
def test_phrasings(tmp_path, question, expected):
    ws = _ws(tmp_path, ROWS)
    res = solve(_task(_q(question), ws))
    assert (res.status, res.answer) == ("solved", expected)


def test_unquoted_region_must_exist_in_data(tmp_path):
    ws = _ws(tmp_path, ROWS)
    for q in ("How many units were sold in region mars?", "How many units were sold in the sales region?"):
        assert solve(_task(_q(q), ws)).status == "unparsed"
    # without a workspace nothing can confirm it
    assert parse(_q("How many units were sold in region west?")).status == "unparsed"


def test_conflicting_phrasings_unparsed():
    q = "How many units were sold in months 1 to 3 in month 5?"
    assert parse(_q(q)).status == "unparsed"


def test_header_case_insensitive(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_text("Order_ID,Region,Product,Month,Units,Price\n1,west,widget,2,4,5\n")
    res = solve(_task(_q("How many units of product 'widget' were sold in region 'west' in month 2?"), ws))
    assert res.answer == "4"


def test_undecodable_file_is_infeasible(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_bytes(b"order_id,region,product,month,units,price\n1,w,\xff\xfe\xfa,1,2,3\n" * 3)
    assert solve(_task(_q("How many units were sold?"), ws)).status == "infeasible"


def test_two_decimal_rounding_non_exact_int(tmp_path):
    ws = _ws(tmp_path, "1,w,a,1,1,2.345\n")
    res = solve(_task(_q("What is the average price?"), ws, check="exact_str"))
    assert res.answer == "2.35"
    (tmp_path / "b").mkdir()
    ws2 = _ws(tmp_path / "b", "1,w,a,1,1,1\n2,w,a,1,1,2\n3,w,a,1,1,2\n")
    assert solve(_task(_q("What is the average price?"), ws2, check="exact_str")).answer == "1.67"
    assert solve(_task(_q("What is the average price?"), ws2)).answer == "2"


def test_negative_zero_normalized(tmp_path):
    ws = _ws(tmp_path, "1,w,a,1,1,-0.001\n")
    res = solve(_task(_q("What is the average price?"), ws, check="exact_str"))
    assert res.answer == "0"

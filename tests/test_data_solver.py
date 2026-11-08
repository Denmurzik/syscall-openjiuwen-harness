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
    assert solve(_task(_q("How many units were sold in months 13 through 14 inclusive?"), ws)).status == "unparsed"
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


# ------------------------------------------------------- round 2: top metrics and loose parsing

from harness.solvers.data_solver import parse_loose  # noqa: E402

TOP_ROWS = ("1,west,widget,1,10,2\n2,west,widget,2,5,3\n3,east,gizmo,6,12,1\n4,east,gizmo,6,1,40\n")


def _spec(metric, **kw):
    base = {"file": "sales.csv", "metric": metric, "product": None, "region": None,
            "month_from": None, "month_to": None}
    return {**base, **kw}


@pytest.mark.parametrize("metric,expected", [
    ("top_product_sum_units", "widget"),      # widget 15 vs gizmo 13
    ("top_product_sum_revenue", "gizmo"),     # gizmo 12+40 vs widget 20+15
    ("top_region_count_orders", None),        # 2 vs 2: tie
    ("top_region_sum_units", "west"),
])
def test_execute_top(tmp_path, metric, expected):
    ws = _ws(tmp_path, TOP_ROWS)
    res = execute(_spec(metric), ws, "exact_str")
    if expected is None:
        assert res.status == "unparsed"
    else:
        assert res == SolveResult("solved", expected)


def test_execute_top_respects_filter_and_keeps_csv_case(tmp_path):
    ws = _ws(tmp_path, "1,West,Widget,1,10,2\n2,East,Gizmo,1,3,2\n3,East,Widget,2,4,2\n")
    assert execute(_spec("top_product_sum_units", month_from=1, month_to=1), ws, "exact_str") \
        == SolveResult("solved", "Widget")


def test_execute_rejects_unknown_top_metric(tmp_path):
    ws = _ws(tmp_path, TOP_ROWS)
    assert execute(_spec("top_month_sum_units"), ws, "exact_str").status == "unparsed"
    assert execute(_spec("top_product_avg_price"), ws, "exact_str").status == "unparsed"


@pytest.mark.parametrize("question,metric", [
    ("Which product sold the most units overall?", "top_product_sum_units"),
    ("Which product had the highest revenue?", "top_product_sum_revenue"),
    ("Which region had the most orders?", "top_region_count_orders"),
])
def test_parse_top_strict(question, metric):
    assert parse(_q(question))["metric"] == metric


LOOSE_CASES = [
    ("What was the total number of 'widget' units sold between month 1 and month 2 (both included)?",
     _spec("sum_units", product="widget", month_from=1, month_to=2)),
    ("What is the total revenue (units multiplied by price) for region 'east'?",
     _spec("sum_revenue", region="east")),
    ("How many orders were placed in month 6?", _spec("count_orders", month_from=6, month_to=6)),
    ("Which product sold the most units overall?", _spec("top_product_sum_units")),
    ("Which region generated the highest revenue?", _spec("top_region_sum_revenue")),
]


@pytest.mark.parametrize("question,spec", LOOSE_CASES)
def test_parse_loose(tmp_path, question, spec):
    ws = _ws(tmp_path, TOP_ROWS)
    assert parse_loose(_q(question), ws) == spec


@pytest.mark.parametrize("question", [
    "How many units of product 'doohickey' were sold?",                  # value not in the CSV
    "How many units of 'widget' were sold in 'east' to 'gizmo' fans?",   # stray words
    "How many units of 'widget' and 'gizmo' were sold?",                 # two values, one column
    "What is the revenue and the number of orders?",                     # two metrics
    "How many units were sold in month 1 and also in month 3?",                  # two month expressions
    "Which product in region 'east' sold the most units per capita?",    # stray word
    "Which product was sold?",                                           # no measure
])
def test_parse_loose_fails_closed(tmp_path, question):
    ws = _ws(tmp_path, TOP_ROWS)
    assert isinstance(parse_loose(_q(question), ws), SolveResult)


def test_solve_loose_end_to_end(tmp_path):
    ws = _ws(tmp_path, ROWS)
    res = solve(_task(_q("What was the total number of 'widget' units sold between month 1 and "
                         "month 6 (both included)?"), ws))
    assert res == SolveResult("solved", "110")
    top = solve(_task(_q("Which product sold the most units overall?"), ws, "exact_str"))
    assert top == SolveResult("solved", "widget")


def test_solve_top_tie_is_unparsed(tmp_path):
    ws = _ws(tmp_path, "1,west,widget,1,5,2\n2,east,gizmo,1,5,2\n")
    assert solve(_task(_q("Which product sold the most units overall?"), ws, "exact_str")).status == "unparsed"


@pytest.mark.parametrize("question", [
    "How many units were sold in months 1 and 3?",
    "How many units were sold in month 1 and month 3?",
])
def test_parse_loose_and_is_not_a_range_without_between(tmp_path, question):
    ws = _ws(tmp_path, TOP_ROWS)
    assert isinstance(parse_loose(_q(question), ws), SolveResult)


def test_parse_loose_between_and_is_a_range(tmp_path):
    ws = _ws(tmp_path, TOP_ROWS)
    res = parse_loose(_q("How many units were sold between months 1 and 3?"), ws)
    assert (res["month_from"], res["month_to"]) == (1, 3)


def test_top_ignores_rows_with_empty_key(tmp_path):
    ws = _ws(tmp_path, "1,west,widget,1,5,2\n2,west,,1,50,2\n3,east,gizmo,1,3,2\n")
    assert execute(_spec("top_product_sum_units"), ws, "exact_str") == SolveResult("solved", "widget")


# ---------------------------------------------------------------- bench grammar (task 17)

def _solve_q(tmp_path, question, check=None):
    return solve(_task(_q(question), _ws(tmp_path, ROWS), check))


@pytest.mark.parametrize("question,answer", [
    ("What is the largest number of units in a single order?", "100"),
    ("What is the smallest number of units in a single order?", "5"),
    ("What is the largest number of units in a single order in region 'west'?", "10"),
    ("What is the smallest number of units in a single order for product 'widget' in region 'west'?", "5"),
    ("What is the largest number of units in a single order in months 6 through 6 inclusive?", "100"),
    ("What is the smallest number of units in a single order in month 6 for product 'gizmo'?", "7"),
    ("What is the average price, rounded to 2 decimal places?", "2.50"),
    ("What is the average number of units per order, rounded to 2 decimal places?", "30.50"),
    ("What is the average price in region 'west' in months 1 through 12 inclusive, rounded to 2 decimal places?",
     "3.00"),
    ("What is the average price in region 'west', rounded to 3 decimal places?", "3.000"),
    ("What is the average number of units per order for product 'widget', rounded to 1 decimal places?", "38.3"),
    ("What is the average number of units per order for product 'widget', rounded to 0 decimal places?", "38"),
])
def test_bench_grammar(tmp_path, question, answer):
    res = _solve_q(tmp_path, question)
    assert (res.status, res.answer) == ("solved", answer)


def test_rounding_places_other_than_two(tmp_path):
    ws = _ws(tmp_path, "1,west,widget,1,1,1\n2,west,widget,1,2,1\n3,west,widget,1,2,1\n")
    res = solve(_task(_q("What is the average number of units per order, rounded to 3 decimal places?"),
                      ws, None))
    assert res.answer == "1.667"


@pytest.mark.parametrize("question", [
    "What is the total units sold, rounded to 1 decimal places?",        # rounding a count/sum of units
    "How many orders were placed, rounded to 2 decimal places?",
    "What is the average price, rounded to 9 decimal places?",           # absurd precision
    "What is the average price, rounded to 2 decimal places, rounded to 3 decimal places?",
    "What is the largest number of units in a single order, rounded to 3 decimal places?",
])
def test_rounding_fails_closed(tmp_path, question):
    assert _solve_q(tmp_path, question).status == "unparsed"


def test_extreme_over_empty_month_selection_is_unparsed(tmp_path):
    res = _solve_q(tmp_path, "What is the largest number of units in a single order in month 11?")
    assert res.status == "unparsed"


def test_execute_rejects_bad_decimals(tmp_path):
    ws = _ws(tmp_path, ROWS)
    base = {"file": "sales.csv", "metric": "avg_price", "product": None, "region": None,
            "month_from": None, "month_to": None}
    assert execute({**base, "decimals": 3}, ws, None).status == "solved"
    for bad in (-1, 7, True, "2", 2.0):
        assert execute({**base, "decimals": bad}, ws, None).status == "unparsed"
    assert execute({**base, "metric": "sum_units", "decimals": 3}, ws, None).status == "unparsed"
    assert execute({**base, "metric": "max_units"}, ws, None).answer == "100"


@pytest.mark.parametrize("question", [
    "What is the largest and smallest number of units in a single order?",
    "What is the largest number of units in orders?",                     # no 'single'
    "Which product had the largest number of units in a single order?",
])
def test_extreme_loose_fails_closed(tmp_path, question):
    assert _solve_q(tmp_path, question).status == "unparsed"


def test_extreme_loose_paraphrase(tmp_path):
    res = _solve_q(tmp_path, "What was the maximum number of units in one single order for widget?")
    assert (res.status, res.answer) == ("solved", "100")


def test_explicit_rounding_keeps_trailing_zeros_but_plain_question_does_not(tmp_path):
    ws = _ws(tmp_path, "1,w,a,1,2,56\n2,w,a,1,2,56\n")
    assert solve(_task(_q("What is the average price, rounded to 2 decimal places?"), ws, "exact_str")).answer == "56.00"
    assert solve(_task(_q("What is the average price?"), ws, "exact_str")).answer == "56"
    assert solve(_task(_q("What is the average price, rounded to 2 decimal places?"), ws, "exact_int")).answer == "56"


@pytest.mark.parametrize("i,q", enumerate([
    "What is the total discount given?", "What is the total discount given in region 'west'?",
    "What is the total tax in months 1 through 3 inclusive for product 'widget'?"]))
def test_absent_column_is_infeasible(tmp_path, i, q):
    assert _solve_q(tmp_path, q).status == "infeasible"


def test_present_column_is_not_declared_absent(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sales.csv").write_text("order_id,region,product,month,units,price,discount\n1,w,a,1,2,3,1\n")
    assert solve(_task(_q("What is the total discount given?"), ws)).status == "unparsed"


def test_revenue_parenthetical(tmp_path):
    res = _solve_q(tmp_path, "What is the total revenue (units multiplied by price) in region 'west'?")
    assert (res.status, res.answer) == ("solved", "63")


def test_extreme_of_unknown_product_is_infeasible_but_empty_month_is_not_declared_so(tmp_path):
    assert _solve_q(tmp_path, "What is the largest number of units in a single order for product 'doohickey'?"
                    ).status == "infeasible"
    ws = tmp_path / "w2"
    ws.mkdir()
    res = _solve_q(ws, "What is the largest number of units in a single order in month 11?")
    assert res.status == "unparsed"


def test_top_tie_carries_first_in_file_order_as_best_guess(tmp_path):
    ws = _ws(tmp_path, "1,west,widget,1,5,2\n2,east,gizmo,1,5,2\n")
    r = solve(_task(_q("Which product sold the most units overall?"), ws, "exact_str"))
    assert r.status == "unparsed" and r.extra == {"best_guess": "widget"}


def test_total_cost_is_not_declared_absent(tmp_path):
    assert _solve_q(tmp_path, "What is the total cost?").status != "infeasible"


def test_execute_treats_null_decimals_as_absent(tmp_path):
    ws = _ws(tmp_path, ROWS)
    base = {"file": "sales.csv", "metric": "sum_units", "product": None, "region": None,
            "month_from": None, "month_to": None}
    assert execute({**base, "decimals": None}, ws, None).status == "solved"
    assert execute({**base, "decimals": None}, ws, None).answer == execute(base, ws, None).answer
    avg = {**base, "metric": "avg_price", "decimals": None}
    assert execute(avg, ws, None).answer == execute({k: v for k, v in avg.items() if k != "decimals"}, ws, None).answer


def test_loose_path_tie_keeps_best_guess(tmp_path):
    ws = _ws(tmp_path, "1,west,widget,1,5,2\n2,east,gizmo,1,5,2\n")
    r = solve(_task(_q("Which product sold the most units in total?"), ws, "exact_str"))
    assert r.status == "unparsed" and r.extra == {"best_guess": "widget"}


def test_strict_months_outside_1_12_are_unparsed():
    for q in ("How many units were sold in month 13?",
              "How many units were sold in months 0 to 5?",
              "How many units were sold between months 3 and 13?"):
        assert parse(_q(q)).status == "unparsed", q
    assert parse(_q("How many units were sold in month 12?"))["month_from"] == 12

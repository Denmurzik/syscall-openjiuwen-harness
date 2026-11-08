"""Round 3 / Task 20: exact-outcome tests for parser fixes (silent-wrong cases and plausible phrasings).

The adversarial corpus only checks the acceptable set; here every case names the exact outcome (a value, or
"unparsed" where the question must stay with the model)."""
from pathlib import Path

import pytest

from tests.test_adversarial_phrasings import (D_HEAD, S_INTRO, U, outcome, ref)
from tools import gen_tasks

SFX = {"data": gen_tasks.SUFFIX_DATA, "search": gen_tasks.SUFFIX_SEARCH, "math": gen_tasks.SUFFIX_MATH}


def _run(kind, text, tmp_path):
    case = {"id": "x", "kind": kind, "text": text + SFX[kind], "check": "exact_str" if kind == "search" else "exact_int"}
    return outcome(case, tmp_path)


def _d(q, head=D_HEAD):
    return ("data", head + q)


def _s(q):
    return ("search", S_INTRO + q)


def _m(sentences, n=1000):
    return ("math", f"A warehouse starts with {n} boxes. " + " ".join(sentences))


MQ = "How many boxes does it have at the end?"
CASES = {
    # ---- data: one-sided month bounds
    "through-month": (_d("How many units were sold through month 5?"), ref("units", lo=1, hi=5)),
    "through-revenue": (_d("What is the total revenue through month 6?"), ref("revenue", lo=1, hi=6)),
    "up-to-month": (_d("How many units were sold up to month 5?"), ref("units", lo=1, hi=5)),
    "until-month": (_d("How many units were sold until month 5?"), ref("units", lo=1, hi=5)),
    "from-onwards": (_d("How many units were sold from month 5 onwards?"), ref("units", lo=5, hi=12)),
    "months-onwards": (_d("How many units were sold in months 5 onwards?"), ref("units", lo=5, hi=12)),
    "on-product-1": (_d("What was the revenue in month 2 on product gadget?"), U),
    "on-product-2": (_d("How many units were sold in month 4 on gizmo?"), U),
    "on-product-3": (_d("Which region sold the most units in month 2 on gadget?"), U),
    "on-product-4": (_d("What is the average number of units in month 3 on widget?"), U),
    "on-product-5": (_d("How many orders were placed in month 3 on widget?"), U),
    "from-month-on": (_d("How many units were sold from month 5 on?"), U),
    "ordinal-13": (_d("How many units were sold in the 13th month?"), U),
    "range-13": (_d("How many units were sold in months 3 through 13?"), U),
    "by-month": (_d("How many units were sold by month 5?"), U),
    "from-month": (_d("How many units were sold from month 5?"), U),
    "after-month": (_d("How many units were sold after month 5?"), U),
    "through-with-range": (_d("How many units were sold through months 3 to 5?"), U),
    "in-month-onwards-dup": (_d("How many units were sold in month 5 onwards from month 6?"), U),
    # ---- data: two measures
    "units-and-revenue": (_d("How many units of product 'widget' were sold in months 3 through 7 inclusive, "
                             "and what is the revenue?"), U),
    "top-both": (_d("Which product had both the most units and the most revenue?"), U),
    "revenue-of-units": (_d("What is the total revenue from units sold?"), U),
    # ---- data: plausible phrasings
    "for-region": (_d("What is the total revenue (units multiplied by price) for region 'west'?"),
                   ref("revenue", region="west")),
    "total-number-units": (_d("What is the total number of units sold for product 'widget' in region 'west'?"),
                           ref("units", "widget", "west")),
    "month-name": (_d("How many units of product 'widget' were sold in July?"), ref("units", "widget", lo=7, hi=7)),
    "month-name-sept": (_d("How many orders were placed in September?"), ref("orders", lo=9, hi=9)),
    "month-ordinal": (_d("How many units of product 'widget' were sold in the 7th month?"),
                      ref("units", "widget", lo=7, hi=7)),
    "month-ordinal-1st": (_d("How many units were sold in the 1st month?"), ref("units", lo=1, hi=1)),
    "month-name-and-number": (_d("How many units were sold in July in month 7?"), U),
    "round-nearest-int": (_d("What is the average price, rounded to the nearest integer?"),
                          ref("avg_price", decimals=0)),
    "round-whole": (_d("What is the average number of units per order, rounded to the nearest whole number?"),
                    ref("avg_units", decimals=0)),
    "rows-comma": (_d("How many units were sold?", head=D_HEAD.replace("(12 rows)", "(1,200 rows)")),
                   ref("units")),
    "no-question-mark": (_d("How many units were sold."), ref("units")),
    "no-question-mark-rev": (_d("What is the total revenue."), ref("revenue")),
    # ---- search
    "city-mentor-work": (_s("In which city does the mentor of Alice Garcia work?"), "Tbilisi"),
    "city-mentor-poss": (_s("In which city does Alice Garcia's mentor work?"), "Tbilisi"),
    "birth-year-poss-mentor": (_s("What is the birth year of Alice Garcia's mentor?"), "1965"),
    "birth-year-the-mentor-of": (_s("What is the birth year of the mentor of Alice Garcia?"), "1965"),
    "birth-year-chain": (_s("What is the birth year of the mentor of the mentor of Alice Garcia?"), "1950"),
    "founded-company-employs": (_s("When was the company that employs Alice Garcia founded?"), "1999"),
    "works-for-which": (_s("Alice Garcia works for which company?"), "Arktika Labs"),
    "which-firm-employs": (_s("Which firm employs Alice Garcia?"), "Arktika Labs"),
    "find-birth-year": (_s("Find the birth year of Alice Garcia."), "1980"),
    "where-work-ambiguous": (_s("Where does Alice Garcia work?"), U),
    "curly-apostrophe-kb-name": (_s("In which year was Patrick O’Brien born?"), "1969"),
    "curly-apostrophe-poss": (_s("What is the birth year of Anne van Dijk’s mentor?"), "1962"),
    # ---- math
    "half-article": (_m(["It sells a half of the stock, with the number of boxes sold rounded down."], 1001),
                     "501"),
    "fifth-article": (_m(["It sells a fifth of the boxes, rounding the number sold down."], 1001), "801"),
    "tenth-one": (_m(["Then it sells one tenth of the boxes, rounding the number sold down."], 1001), "901"),
    "half-moved": (_m(["Half of the stock (rounded down) is moved to another site."], 1001), "501"),
    "comma-start": (("math", "A warehouse starts with 1,752 boxes. It ships 30 boxes per day for 4 days."), "1632"),
    "singular-box": (_m(["It receives 1 delivery of 1 box each."], 1), "2"),
    "singular-loss": (_m(["It loses 1 box to theft."], 5), "4"),
    "singular-ship": (_m(["It ships 1 box per day for 1 day."], 5), "4"),
    "comma-delivery": (_m(["It receives 2 deliveries of 1,500 boxes each."], 10), "3010"),
    "comma-bad": (_m(["It receives 2 deliveries of 1,50 boxes each."], 10), U),
}


@pytest.mark.parametrize("name", list(CASES))
def test_exact(name, tmp_path):
    (kind, text), want = CASES[name]
    if kind == "math" and "How many boxes" not in text:
        text += " " + MQ
    assert _run(kind, text, tmp_path) == want



@pytest.mark.parametrize("q", ["How many units were sold", "What is the total revenue", "How many orders were placed",
                               "Which product sold the most units"])
def test_strict_parse_without_final_punctuation(q):
    from harness.solvers import data_solver
    spec = data_solver.parse(D_HEAD + q)
    assert isinstance(spec, dict), spec


@pytest.mark.parametrize("q", ["How many units were sold!?", "How many units were solds", "How many units were sold?x"])
def test_strict_parse_rejects_garbage_tail(q):
    from harness.solvers import data_solver
    assert not isinstance(data_solver.parse(D_HEAD + q), dict)

"""Hand-written adversarial corpus for the phase-1 parsers (math / data_analysis / search).

A wrong phase-1 answer is final; `unparsed` only costs tokens. Every case lists the ACCEPTABLE outcomes
(an exact answer, "unparsed" or "infeasible") and the ideal one (`want`). A test fails only on a silent wrong
outcome (a value or verdict outside the acceptable set). Outcomes that are acceptable but not ideal are
"misses": `python -m tests.test_adversarial_phrasings` prints them (organizer-plausible phrasings the parsers
do not handle yet). Known wrong outcomes are marked `xfail(strict=True)` via XFAIL below.
"""
import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from harness.solvers import data_solver, math_solver, search_solver
from harness.tasks import Task
from tools import gen_tasks

U, I = "unparsed", "infeasible"

# case id -> one-line reason (strict xfail: the solver gives a silent wrong outcome today)
XFAIL: dict[str, str] = {}

CASES: list[dict] = []


def _case(kind, cid, text, ok, want=None, check="exact_int"):
    ok = set(ok)
    CASES.append({"id": cid, "kind": kind, "text": text, "ok": ok, "want": want if want is not None else
                  (next(iter(ok)) if len(ok) == 1 else None), "check": check})


# ============================================================================ math
M_START = "A warehouse starts with {n} boxes."
M_Q = "How many boxes does it have at the end?"


def _m(cid, sentences, ok, want=None, q=M_Q, start="A warehouse starts with 1000 boxes."):
    text = " ".join([start, *sentences, q] if q else [start, *sentences])
    _case("math", cid, text + gen_tasks.SUFFIX_MATH, ok, want)


def _mv(cid, sentences, value, start_n=1000, **kw):
    """Answer expected (or unparsed): `value` is computed independently by the caller."""
    _m(cid, sentences, {str(value), U}, str(value), start=f"A warehouse starts with {start_n} boxes.", **kw)


def _mu(cid, sentences, extra=(), **kw):
    _m(cid, sentences, {U, *extra}, U, **kw)


# canonical phrasings, chains of length 0..4
_mv("m-chain0", [], 1000)
_mv("m-ship", ["It ships 30 boxes per day for 4 days."], 1000 - 120)
_mv("m-ship2", ["Every day for 5 days, 7 boxes leave the warehouse."], 965)
_mv("m-ship3", ["For 3 days in a row, 11 boxes are shipped out daily."], 967)
_mv("m-recv", ["It receives 3 deliveries of 40 boxes each."], 1120)
_mv("m-recv2", ["6 trucks arrive, each carrying 25 boxes."], 1150)
_mv("m-recv3", ["The warehouse takes in 2 shipments of 15 boxes apiece."], 1030)
_mv("m-loss", ["It loses 33 boxes to theft."], 967)
_mv("m-loss2", ["Thieves take 12 boxes."], 988)
_mv("m-loss3", ["A leak ruins 50 boxes, which are thrown away."], 950)
_mv("m-loss4", ["Water damage destroys 70 boxes."], 930)
_mv("m-split", ["The stock is split equally into 3 warehouses and it keeps one share (rounded down)."], 333)
_mv("m-split3", ["The stock is divided evenly among 6 warehouses, and this one retains a single share (rounded down)."],
    166)
_mv("m-double", ["The stock is doubled."], 2000)
_mv("m-triple", ["The stock is tripled by a big merger."], 3000)
_mv("m-merger", ["A big merger with a sister warehouse then doubles the stock."], 2000)
_mv("m-merger2", ["A merger triples the stock."], 3000)
_mv("m-quarter", ["A quarter of the stock (rounded down) is moved to another site."], 751, start_n=1001)
_mv("m-half-sell", ["It sells a half of the stock, with the number of boxes sold rounded down."], 501, start_n=1001)
_mv("m-third", ["A third of the stock, rounded down, is transferred to another site."], 1001 - 333, start_n=1001)
_mv("m-tenth", ["Then it sells one tenth of the boxes, rounding the number sold down."], 901, start_n=1001)
_mv("m-fifth", ["It sells a fifth of the boxes, rounding the number sold down."], 801, start_n=1001)
_mv("m-pct-up", ["The stock grows by 15 percent (rounded down)."], 1150)
_mv("m-pct-up2", ["A seasonal restock adds 33 percent of the current stock, rounded down."], 1330)
_mv("m-pct-dn", ["25 percent of the stock (rounded down) spoils."], 750)
_mv("m-pct-dn2", ["Spoilage removes 7 percent of the stock, rounded down."], 930)
_mv("m-pct-up-floor", ["The stock grows by 15 percent (rounded down)."], 999 + 15 * 999 // 100, start_n=999)
_mv("m-pct-dn-floor", ["25 percent of the stock (rounded down) spoils."], 999 - 999 * 25 // 100, start_n=999)
_mv("m-chain2", ["It ships 10 boxes per day for 5 days.", "The stock is doubled."], 1900)
_mv("m-chain3", ["Then the stock is doubled.", "Then it loses 100 boxes to theft.",
                 "Then the stock is tripled."], 5700)
_mv("m-chain4", ["The stock is split equally into 4 warehouses and it keeps one share (rounded down).",
                 "It receives 2 deliveries of 10 boxes each.", "The stock grows by 10 percent (rounded down).",
                 "Thieves take 5 boxes."], 292)  # 1000 -> 250 -> 270 -> 297 -> 292
_mv("m-order-matters", ["The stock is doubled.", "It loses 500 boxes to theft."], 1500)
_mv("m-order-matters2", ["It loses 500 boxes to theft.", "The stock is doubled."], 1000)
_mv("m-divide-after-floor", ["The stock is split equally into 3 warehouses and it keeps one share (rounded down).",
                              "The stock is split equally into 3 warehouses and it keeps one share (rounded down)."],
    111)
_mv("m-zero-days", ["It ships 5 boxes per day for 0 days."], 1000)
_mv("m-ship-exact-zero", ["It loses 1000 boxes to theft."], 0)
_mv("m-big", ["It receives 9 deliveries of 123456789 boxes each."], 1000 + 9 * 123456789)

# formatting noise
_mv("m-noise-ws", ["It   ships 30  boxes per day for 4 days."], 880)
_mv("m-noise-nbsp", ["It ships 30 boxes per day for 4 days."], 880)
_mv("m-noise-newline", ["\nIt ships 30 boxes per day\nfor 4 days."], 880)
_mv("m-noise-case", ["IT SHIPS 30 BOXES PER DAY FOR 4 DAYS."], 880)
_mv("m-noise-then", ["Then it ships 30 boxes per day for 4 days."], 880)

# near-misses that must NOT be computed
_mu("m-halved", ["The stock is halved."])
_mu("m-quadrupled", ["The stock is quadrupled."])
_mu("m-merger-quad", ["A merger with a rival warehouse quadruples the stock."])
_mu("m-round-up", ["It sells a third of the stock, with the number of boxes sold rounded up."])
_mu("m-round-none", ["It sells a third of the boxes."])
_mu("m-round-nearest", ["The stock grows by 15 percent (rounded to the nearest box)."])
_mu("m-pct-sign", ["The stock grows by 15% (rounded down)."])
_mu("m-pct-decimal", ["The stock grows by 12.5 percent (rounded down)."])
_mu("m-pct-word", ["The stock grows by fifteen percent (rounded down)."])
_mu("m-num-words", ["It ships three boxes per day for 4 days."])
_m("m-comma-start", ["The stock is doubled."], {U, "3504"}, "3504", start="A warehouse starts with 1,752 boxes.")
_m("m-comma-op", ["It receives 2 deliveries of 1,500 boxes each."], {U, "4000"}, "4000")
_mu("m-negation", ["It does not lose any boxes."])
_mu("m-negation2", ["It loses no boxes to theft."])
_mu("m-atleast", ["It ships at least 5 boxes per day for 3 days."])
_mu("m-atmost", ["It ships at most 5 boxes per day for 3 days."])
_mu("m-except", ["It ships 5 boxes per day for 3 days, except on day 2."])
_mu("m-unit-week", ["It ships 5 boxes per week for 3 days."])
_mu("m-per-day-no-days", ["It ships 5 boxes per day."])
_mu("m-demerger", ["A demerger doubles the stock."])
_mu("m-not-double", ["A merger does not double the stock."])
_mu("m-halfway", ["A half of the stock is moved to another site."])  # no rounding clause
_mu("m-q-start", [], q="How many boxes does it have at the start?")
_mu("m-q-other", [], q="How many boxes are in the warehouse?")
_mu("m-q-twice", [], q=M_Q + " " + M_Q)
_mu("m-no-q", [], q=None)
_mu("m-q-no-mark", [], q="How many boxes does it have at the end")
_mu("m-op-after-q", [], q=M_Q + " It loses 5 boxes to theft.")
_mu("m-two-starts", ["A warehouse starts with 5 boxes."])
_mu("m-unknown-sentence", ["The warehouse is renovated."])
_mu("m-sentence-union", ["It ships 5 boxes per day for 3 days and loses 7 boxes to theft."])
_mu("m-neg-stock", ["It loses 2000 boxes to theft."], extra=set())
_m("m-div-zero", ["The stock is split equally into 0 warehouses and it keeps one share (rounded down)."],
   {I, U}, I)
_m("m-start-singular", ["It loses 1 boxes to theft."], {U, "0"}, "0", start="A warehouse starts with 1 box.")
_m("m-percent-of-days", ["It ships 10 percent of the boxes per day for 3 days."], {U}, U)
_m("m-ordinal", ["On the second day it loses 10 boxes to theft."], {U}, U)
_m("m-currency", ["It receives 2 deliveries of $50 boxes each."], {U}, U)

# ============================================================================ data
ROWS = [  # order_id, region, product, month, units, price
    (1, "north", "widget", 1, 10, 5), (2, "north", "gadget", 2, 3, 20), (3, "south", "widget", 3, 7, 5),
    (4, "south", "gizmo", 4, 12, 8), (5, "east", "widget", 5, 4, 6), (6, "east", "gadget", 6, 9, 20),
    (7, "west", "sprocket", 7, 2, 30), (8, "west", "widget", 7, 5, 5), (9, "north", "gizmo", 8, 6, 8),
    (10, "south", "gadget", 9, 1, 20), (11, "east", "gizmo", 10, 11, 8), (12, "west", "widget", 12, 8, 5),
]
D_HEAD = "The file sales.csv has columns order_id,region,product,month,units,price (12 rows). "


def ref(metric, product=None, region=None, lo=None, hi=None, decimals=None):
    rows = [r for r in ROWS if (product is None or r[2] == product) and (region is None or r[1] == region)
            and (lo is None or lo <= r[3] <= hi)]
    if metric.startswith("top_"):
        _, col, base = metric.split("_")
        idx = {"product": 2, "region": 1}[col]
        g: dict = {}
        for r in rows:
            g[r[idx]] = g.get(r[idx], 0) + (1 if base == "orders" else r[4] if base == "units" else r[4] * r[5])
        best = sorted(g.values(), reverse=True)
        assert len(best) < 2 or best[0] != best[1], "tie in fixture query"
        return max(g, key=g.get)
    units = [r[4] for r in rows]
    if metric == "units":
        return str(sum(units))
    if metric == "revenue":
        v = Decimal(sum(r[4] * r[5] for r in rows))
    elif metric == "orders":
        return str(len(rows))
    elif metric == "max":
        return str(max(units))
    elif metric == "min":
        return str(min(units))
    elif metric == "avg_units":
        v = Decimal(sum(units)) / len(units)
    elif metric == "avg_price":
        v = Decimal(sum(r[5] for r in rows)) / len(rows)
    else:
        raise ValueError(metric)
    d = 2 if decimals is None else decimals
    out = format(v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP), "f")
    if decimals is None and "." in out:
        out = out.rstrip("0").rstrip(".")
    return out


def _d(cid, q, ok, want=None, check="exact_int"):
    _case("data", cid, D_HEAD + q + gen_tasks.SUFFIX_DATA, ok, want, check)


def _da(cid, q, metric, check=None, **kw):
    """The question has one unambiguous answer; unparsed is only a miss."""
    v = ref(metric, **kw)
    _d(cid, q, {v, U}, v, check or ("exact_int" if metric in ("units", "revenue", "orders", "max", "min")
                                    and kw.get("decimals") is None else None))


def _du(cid, q, extra=()):
    _d(cid, q, {U, *extra}, U)


# canonical
_da("d-units-range", "How many units of product 'widget' were sold in months 3 through 7 inclusive?", "units",
    product="widget", lo=3, hi=7)
_da("d-units-all", "How many units were sold?", "units")
_da("d-units-total", "What is the total units sold?", "units")
_da("d-revenue", "What is the total revenue (units multiplied by price)?", "revenue")
_da("d-revenue-region", "What is the total revenue in region 'north'?", "revenue", region="north")
_da("d-orders", "How many orders were placed?", "orders")
_da("d-orders-month", "How many orders were placed in month 7?", "orders", lo=7, hi=7)
_da("d-avg-price", "What is the average price?", "avg_price")
_da("d-avg-units", "What is the average number of units per order?", "avg_units")
_da("d-max", "What is the largest number of units in a single order?", "max")
_da("d-min", "What is the smallest number of units in a single order?", "min")
_da("d-top-prod-units", "Which product sold the most units overall?", "top_product_units")
_da("d-top-prod-rev", "Which product generated the highest revenue?", "top_product_revenue")
_da("d-top-reg-rev", "Which region generated the highest revenue?", "top_region_revenue")
_da("d-top-prod-orders", "Which product had the most orders?", "top_product_orders")
_da("d-top-reg-units", "Which region sold the most units?", "top_region_units")
_da("d-top-in-month", "Which product sold the most units in months 1 through 5 inclusive?", "top_product_units",
    lo=1, hi=5)
# ranges
_da("d-range-between", "How many units of product 'widget' were sold between months 3 and 7?", "units",
    product="widget", lo=3, hi=7)
_da("d-range-from", "How many units of product 'widget' were sold from month 3 to month 7?", "units",
    product="widget", lo=3, hi=7)
_da("d-range-to", "How many units of product 'widget' were sold in months 3 to 7?", "units", product="widget",
    lo=3, hi=7)
_da("d-range-hyphen", "How many units of product 'widget' were sold in months 3-7?", "units", product="widget",
    lo=3, hi=7)
_da("d-range-month-single", "How many units of product 'widget' were sold in month 7?", "units", product="widget",
    lo=7, hi=7)
_da("d-range-month-13", "How many units of product 'widget' were sold in month 13?", "units", product="widget",
    lo=13, hi=13)
_da("d-range-month-zero-pad", "How many units were sold in month 07?", "units", lo=7, hi=7)
_d("d-range-reversed", "How many units were sold in months 7 through 3 inclusive?", {U, "0"}, U)
_d("d-range-exclusive", "How many units were sold in months 3 through 7 exclusive?", {U}, U)
_d("d-range-exclusive2", "How many units were sold between months 3 and 7 (exclusive)?", {U}, U)
_d("d-range-two", "How many units were sold in months 1 through 2 and in months 5 through 6?", {U}, U)
_d("d-month-name", "How many units of product 'widget' were sold in July?", {U, ref("units", "widget", lo=7, hi=7)},
   ref("units", "widget", lo=7, hi=7))
_d("d-month-ordinal", "How many units of product 'widget' were sold in the 7th month?",
   {U, ref("units", "widget", lo=7, hi=7)}, ref("units", "widget", lo=7, hi=7))
_d("d-month-sept", "How many orders were placed in September?", {U, ref("orders", lo=9, hi=9)},
   ref("orders", lo=9, hi=9))
_d("d-quarter", "How many units were sold in the first quarter?", {U}, U)
_d("d-first-months", "How many units were sold in the first 3 months?", {U}, U)
_d("d-months-through-only", "How many units were sold through month 5?", {U, ref("units", lo=1, hi=5)}, ref("units", lo=1, hi=5))
_d("d-after-month", "How many units were sold after month 5?", {U}, U)
_d("d-before-month", "How many units were sold before month 5?", {U}, U)
_d("d-month-1000", "How many units were sold in month 1,000?", {U, "0"}, U)
# filters / quoting
_da("d-prod-region", "How many units of product 'widget' were sold in region 'north'?", "units", product="widget",
    region="north")
_da("d-region-the", "What is the total revenue in the 'east' region?", "revenue", region="east")
_da("d-region-unquoted", "What is the total revenue in region east?", "revenue", region="east")
_da("d-region-bare", "How many orders were placed in the south?", "orders", region="south")
_da("d-prod-upper", "How many units of product 'Widget' were sold?", "units", product="widget")
_da("d-prod-unquoted", "How many units of widget were sold?", "units", product="widget")
_da("d-prod-double-quote", 'How many units of product "widget" were sold?', "units", product="widget")
_da("d-prod-curly", "How many units of product ‘widget’ were sold?", "units", product="widget")
_du("d-two-products", "How many units of product 'widget' and 'gadget' were sold?")
_du("d-two-products-or", "How many units of product 'widget' or 'gadget' were sold?")
_du("d-two-products-sep", "How many units of product 'widget' and product 'gadget' were sold?")
_du("d-two-regions", "How many orders were placed in region 'north' and region 'south'?")
_du("d-product-and-region-conflict", "How many units of product 'widget' were sold in region 'widget'?", extra={"0"})
# negation / exclusion / thresholds
_du("d-except", "How many units were sold, except for product 'widget'?")
_du("d-other-than", "How many units were sold in regions other than 'north'?")
_du("d-excluding", "What is the total revenue excluding product 'widget'?")
_du("d-excluding-region", "How many orders were placed, excluding region 'north'?")
_du("d-not", "How many units of product 'widget' were not sold in region 'north'?")
_du("d-without", "What is the total revenue without region 'north'?")
_du("d-at-least", "How many orders had at least 5 units?")
_du("d-at-most", "How many orders had at most 5 units?")
_du("d-more-than", "How many orders had more than 5 units?")
_du("d-less-than", "How many units were sold in orders of fewer than 5 units?")
_du("d-over-price", "How many orders had a price over 10?")
# metric near-misses
_du("d-fewest", "Which product sold the fewest units?")
_du("d-least-revenue", "Which product generated the least revenue?")
_du("d-two-top", "Which two products sold the most units?")
_du("d-top-both", "Which product and region sold the most units?")
_du("d-median", "What is the median price?")
_du("d-distinct", "How many distinct products were sold?")
_du("d-per-unit", "What is the average price per unit?")
_du("d-order-value", "What is the average order value?")
_du("d-percent", "What percent of orders were placed in region 'north'?")
_du("d-share", "What share of the total revenue came from product 'widget'?")
_du("d-doubled", "What would the revenue be if all prices were doubled?")
_du("d-growth", "By how much did units grow from month 1 to month 12?")
_du("d-possessive", "What is widget's total revenue?", extra={ref("revenue", "widget")})
_du("d-two-questions", "How many units were sold? What is the total revenue?")
_du("d-plural-orders-singular", "How many order were placed?", extra={ref("orders")})
_du("d-per-region", "What is the total revenue per region?")
_du("d-which-no-measure", "Which product was best?")
_d("d-discount", "What is the total discount given?", {I, U}, I)
_d("d-profit", "What is the total profit?", {I, U}, I)
# rounding / numbers
_da("d-round-2", "What is the average price, rounded to 2 decimal places?", "avg_price", decimals=2)
_da("d-round-1", "What is the average number of units per order, rounded to 1 decimal place?", "avg_units", decimals=1)
_da("d-round-0", "What is the average price, rounded to 0 decimal places?", "avg_price", decimals=0)
_da("d-round-rev", "What is the total revenue, rounded to 2 decimal places?", "revenue", decimals=2)
_d("d-round-up", "What is the average price, rounded up to 2 decimal places?", {U}, U)
_d("d-round-down", "What is the average price, rounded down to the nearest integer?", {U}, U)
_d("d-round-nearest", "What is the average price, rounded to the nearest integer?", {U, ref("avg_price", decimals=0)},
   ref("avg_price", decimals=0))
_d("d-round-ceil", "What is the average price (rounded up)?", {U}, U)
_d("d-round-sig", "What is the average price, rounded to 3 significant figures?", {U}, U)
_d("d-rows-comma", "How many units were sold?", {U, ref("units")}, ref("units"))
CASES[-1]["text"] = CASES[-1]["text"].replace("(12 rows)", "(1,200 rows)")
_d("d-extra-ws", "How  many   units of   product 'widget'  were sold  in months 3 through 7 inclusive?",
   {U, ref("units", "widget", lo=3, hi=7)}, ref("units", "widget", lo=3, hi=7))
_d("d-no-mark", "How many units were sold", {U, ref("units")}, ref("units"))
_d("d-wrong-file", "How many units were sold?", {I, U}, I)
CASES[-1]["text"] = CASES[-1]["text"].replace("sales.csv", "orders.csv")

# ============================================================================ search
KB = {
    "Alice Garcia": {"Born": "1980", "Employer": "Arktika Labs", "Mentor": "Boris Petrov"},
    "Boris Petrov": {"Born": "1965", "Employer": "Bluefin Systems", "Mentor": "Chen Wang"},
    "Chen Wang": {"Born": "1950", "Employer": "Cobalt Works", "Mentor": "Dmitri Sato"},
    "Dmitri Sato": {"Born": "1940", "Employer": "Dune Robotics"},
    "Jean-Luc Moreau": {"Born": "1971", "Employer": "Ember Bio", "Mentor": "Alice Garcia"},
    "Jonas T. Dubois": {"Born": "1975", "Employer": "Fjord Data", "Mentor": "Boris Petrov"},
    "Maria de la Cruz": {"Born": "1962", "Employer": "Arktika Labs", "Mentor": "Jonas T. Dubois"},
    "Anne van Dijk": {"Born": "1988", "Employer": "Bluefin Systems", "Mentor": "Maria de la Cruz"},
    "Patrick O'Brien": {"Born": "1969", "Employer": "Cobalt Works", "Mentor": "Anne van Dijk"},
    "Arktika Labs": {"Founded": "1999", "Headquarters": "Medan"},
    "Bluefin Systems": {"Founded": "2005", "Headquarters": "Tbilisi"},
    "Cobalt Works": {"Founded": "2010", "Headquarters": "Hanoi"},
    "Dune Robotics": {"Founded": "1990", "Headquarters": "Osaka"},
    "Ember Bio": {"Founded": "2015", "Headquarters": "Quito"},
    "Fjord Data": {"Founded": "2001", "Headquarters": "Lyon"},
}
S_INTRO = "The directory kb/ contains a small knowledge base of markdown files. "


def _s(cid, q, ok, want=None):
    _case("search", cid, S_INTRO + q + gen_tasks.SUFFIX_SEARCH, ok, want, "exact_str")


def _sa(cid, q, value):
    _s(cid, q, {value, U}, value)


def _su(cid, q, extra=()):
    _s(cid, q, {U, *extra}, U)


def _sn(cid, q):
    _s(cid, q, {I, U}, I)


# direct facts
_sa("s-born", "In which year was Alice Garcia born?", "1980")
_sa("s-born-what", "What year was Alice Garcia born?", "1980")
_sa("s-born-tell", "Can you tell me the year Alice Garcia was born?", "1980")
_sa("s-born-birthyear", "What is the birth year of Alice Garcia?", "1980")
_sa("s-mentor", "Who is the mentor of Alice Garcia?", "Boris Petrov")
_sa("s-mentor-poss", "Who is Alice Garcia's mentor?", "Boris Petrov")
_sa("s-mentor-poss-curly", "Who is Alice Garcia’s mentor?", "Boris Petrov")
_sa("s-mentor-name", "Name the mentor of Alice Garcia.", "Boris Petrov")
_sa("s-mentor-who-mentors", "Who mentors Alice Garcia?", "Boris Petrov")
_sa("s-employer", "Which company is the employer of Alice Garcia?", "Arktika Labs")
_sa("s-employer-work", "For which company does Alice Garcia work?", "Arktika Labs")
_sa("s-employer-work2", "Which company does Alice Garcia work for?", "Arktika Labs")
_sa("s-employer-poss", "Who is Alice Garcia's employer?", "Arktika Labs")
_sa("s-founded", "In which year was Arktika Labs founded?", "1999")
_sa("s-founded-when", "When was Arktika Labs founded?", "1999")
_sa("s-hq", "In which city is the headquarters of Arktika Labs?", "Medan")
_sa("s-hq-where", "Where is Arktika Labs headquartered?", "Medan")
# chains of length 1..4
_sa("s-chain1-born", "In which year was the mentor of Alice Garcia born?", "1965")
_sa("s-chain1-poss", "In which year was Alice Garcia's mentor born?", "1965")
_sa("s-chain2-poss", "In which year was Alice Garcia's mentor's mentor born?", "1950")
_sa("s-chain2-the", "In which year was the mentor of the mentor of Alice Garcia born?", "1950")
_sa("s-chain3-the", "In which year was the mentor of the mentor of the mentor of Alice Garcia born?", "1940")
_sa("s-chain2-employer-hq", "In which city is the headquarters of Alice Garcia's mentor's employer?", "Tbilisi")
_sa("s-chain1-hq", "In which city is the headquarters of Alice Garcia's employer?", "Medan")
_sa("s-chain1-founded", "In which year was the employer of Boris Petrov founded?", "2005")
_sa("s-chain1-hq-the", "In which city does the employer of Boris Petrov have its headquarters?", "Tbilisi")
_sa("s-chain-mentor-employer", "Which company is the employer of the mentor of Alice Garcia?", "Bluefin Systems")
_sa("s-chain-mentor-of-employer-name", "Who is the mentor of Jean-Luc Moreau's mentor?", "Boris Petrov")
_sa("s-chain3-poss", "Who is Jean-Luc Moreau's mentor's mentor's mentor?", "Chen Wang")
_sn("s-chain4-too-long", "In which year was the mentor of the mentor of the mentor of the mentor of Alice Garcia born?")
_sn("s-chain-mentor-of-company", "Who is the mentor of Arktika Labs?")
_sn("s-chain-no-mentor", "Who is the mentor of Dmitri Sato?")
_sn("s-unknown-person", "In which year was Zed Nobody born?")
_sn("s-unknown-company", "In which year was Nonexistent Corp founded?")
_sn("s-person-company-field", "In which city is the headquarters of Alice Garcia?")
# names
_sa("s-name-initial", "In which year was Jonas T. Dubois born?", "1975")
_sa("s-name-initial-mentor", "Who is the mentor of Jonas T. Dubois?", "Boris Petrov")
_sa("s-name-initial-poss", "Who is Jonas T. Dubois's mentor?", "Boris Petrov")
_sa("s-name-initial-chain", "In which year was Maria de la Cruz's mentor born?", "1975")
_sa("s-name-initial-chain2", "Who is the mentor of the mentor of Maria de la Cruz?", "Boris Petrov")
_sa("s-name-hyphen", "In which year was Jean-Luc Moreau born?", "1971")
_sa("s-name-hyphen-poss", "Who is Jean-Luc Moreau's employer?", "Ember Bio")
_sa("s-name-particle", "In which year was Maria de la Cruz born?", "1962")
_sa("s-name-particle2", "Who is the mentor of Anne van Dijk?", "Maria de la Cruz")
_sa("s-name-particle-poss", "In which year was Anne van Dijk's mentor born?", "1962")
_sa("s-name-apostrophe", "In which year was Patrick O'Brien born?", "1969")
_sa("s-name-apostrophe-mentor", "Who is the mentor of Patrick O'Brien?", "Anne van Dijk")
_sa("s-name-apostrophe-poss", "Who is Patrick O'Brien's mentor?", "Anne van Dijk")
_sa("s-name-apostrophe-curly", "In which year was Patrick O’Brien born?", "1969")
_sa("s-name-apostrophe-chain", "In which year was Patrick O'Brien's mentor's mentor born?", "1962")
_sa("s-name-lower", "In which year was alice garcia born?", "1980")
_sa("s-name-upper", "In which year was ALICE GARCIA born?", "1980")
_sa("s-name-extra-ws", "In  which year   was Alice   Garcia born?", "1980")
_s("s-name-partial", "In which year was Alice born?", {U, I}, I)
_sn("s-name-surname-only", "In which year was Garcia born?")
# punctuation
_sa("s-punct-none", "In which year was Alice Garcia born", "1980")
_sa("s-punct-dot", "In which year was Alice Garcia born.", "1980")
_sa("s-punct-double", "In which year was Alice Garcia born??", "1980")
_sa("s-punct-space", "In which year was Alice Garcia born ?", "1980")
_sa("s-punct-ellipsis", "In which year was Alice Garcia born...", "1980")
_sa("s-punct-quoted", 'In which year was "Alice Garcia" born?', "1980")
_sa("s-punct-leading-ws", "   In which year was Alice Garcia born?", "1980")
# two names / multi-question / negation / plural
_su("s-two-names", "Who is the mentor of Alice Garcia and Boris Petrov?")
_su("s-two-names-compare", "Was Alice Garcia born before Boris Petrov?")
_su("s-two-names-both", "In which year were Alice Garcia and Boris Petrov born?")
_su("s-two-names-years", "In which year was Alice Garcia born, and where does Boris Petrov work?")
_su("s-two-names-or", "Who is the mentor of Alice Garcia or Boris Petrov?")
_su("s-two-questions", "In which year was Alice Garcia born? Who is her mentor?")
_su("s-pronoun", "Who is her mentor?")
_su("s-negation", "Which company does Alice Garcia not work for?")
_su("s-negation2", "Who is not the mentor of Alice Garcia?")
_su("s-former", "Who was the former mentor of Alice Garcia?")
_su("s-former-employer", "Which company was the previous employer of Alice Garcia?")
_su("s-mentee", "Who is the mentee of Boris Petrov?")
_su("s-plural-mentors", "Who are the mentors of Alice Garcia?", extra={"Boris Petrov"})
_su("s-plural-employers", "Which companies employ Alice Garcia?", extra={"Arktika Labs"})
_su("s-all", "Which people work for Arktika Labs?")
_su("s-count", "How many people work for Arktika Labs?")
_su("s-which-older", "Who is older, Alice Garcia or Boris Petrov?")
_su("s-age", "How old is Alice Garcia?")
_su("s-age-diff", "How many years older is Boris Petrov than Alice Garcia?")
_su("s-before-after", "Which company was founded first, Arktika Labs or Bluefin Systems?")
_su("s-hq-of-both", "Where are Arktika Labs and Bluefin Systems headquartered?")
_su("s-month", "In which month was Alice Garcia born?")
_su("s-country", "In which country is the headquarters of Arktika Labs?")
_su("s-founder", "Who founded Arktika Labs?")
_su("s-company-of-mentor-founded-when", "How long ago was the employer of Alice Garcia founded?")
_su("s-unknown-field", "What is the phone number of Alice Garcia?")
# readings: mentor(employer(Alice)) has no answer; employer(mentor(Alice)) = Bluefin Systems (judgment call)
_su("s-ambiguous-scope", "Who is the mentor of Alice Garcia's employer?", extra={"Bluefin Systems", I})
_su("s-other-than", "Which company other than Arktika Labs employs Maria de la Cruz's mentor?")
_su("s-except", "Who is the mentor of everyone except Alice Garcia?")
_su("s-at-least", "Which companies were founded at least in 2005?")
_su("s-between", "Which company was founded between 2000 and 2010?")


# one-sided month bounds, compound questions (found by probing the loose parsers)
_d("d-by-month", "How many units were sold by month 5?", {U, ref("units", lo=1, hi=5)}, ref("units", lo=1, hi=5))
_d("d-from-month", "How many units were sold from month 5?", {U, ref("units", lo=5, hi=12)}, ref("units", lo=5, hi=12))
_d("d-through-revenue", "What is the total revenue through month 6?", {U, ref("revenue", lo=1, hi=6)},
   ref("revenue", lo=1, hi=6))
_d("d-compound-units-revenue",
   "How many units of product 'widget' were sold in months 3 through 7 inclusive, and what is the revenue?", {U}, U)
_d("d-top-both-measures", "Which product had both the most units and the most revenue?", {U}, U)
_du("d-up-to-month", "How many units were sold up to month 5?", extra={ref("units", lo=1, hi=5)})
_du("d-until-month", "How many units were sold until month 5?", extra={ref("units", lo=1, hi=5)})
_du("d-month-onwards", "How many units were sold in months 5 onwards?", extra={ref("units", lo=5, hi=12)})
_du("d-months-and-list", "How many units were sold in months 3, 4 and 5?", extra={ref("units", lo=3, hi=5)})
_du("d-months-two-single", "How many units were sold in month 3 and month 5?")
_du("d-versus", "How many units were sold in month 5 versus month 6?")
_du("d-plus", "What is the total revenue in month 5 plus month 6?")
_du("d-each", "How many units were sold in each month?")
_du("d-time-2", "What is the total revenue of widget times 2?")
_da("d-from-through-month", "How many units were sold from month 5 through month 8?", "units", lo=5, hi=8)
_da("d-between-month-and-month", "How many units were sold between month 5 and month 8?", "units", lo=5, hi=8)
_da("d-included", "How many units were sold in months 3 through 5 included?", "units", lo=3, hi=5)
_da("d-prod-top-in-region", "Which product sold the most units in region north?", "top_product_units",
    region="north")
_da("d-region-top-for-prod", "Which region has the most orders for widget?", "top_region_orders", product="widget")
_da("d-the-widget-total", "How many units of the widget were sold in total?", "units", product="widget")
_da("d-prod-region-month", "How many units of widget were sold in the north region in month 1?", "units",
    product="widget", region="north", lo=1, hi=1)
_da("d-single-order-region", "What is the largest number of units ordered in a single order in region north?", "max",
    region="north")
_sa("s-who-does-work", "Who does Alice Garcia's mentor work for?", "Bluefin Systems")
_sa("s-born-in-which", "Alice Garcia's mentor was born in which year?", "1965")
_sa("s-year-of-birth", "What is the year of birth of Boris Petrov?", "1965")
_sa("s-poss-year-of-birth", "What is Alice Garcia's mentor's year of birth?", "1965")
_sa("s-whom", "Whom does Alice Garcia work for?", "Arktika Labs")
_sa("s-mentor-poss3", "Who is the mentor of Patrick O'Brien's mentor's mentor?", "Jonas T. Dubois")
_su("s-two-fields", "In which year was Alice Garcia's employer founded and where is it headquartered?")
_su("s-city-founded", "In which city was Alice Garcia's employer founded?")
_su("s-yes-no", "Is Boris Petrov the mentor of Alice Garcia?")
_su("s-mentee-appositive", "Who is the mentor of Alice Garcia, Boris Petrov's mentee?")
_su("s-and-employer", "Who is the employer of Boris Petrov and Alice Garcia?")
_su("s-boss", "Who is the boss of Alice Garcia?")
_mu("m-and-then", ["The stock is doubled and then tripled."])
_mu("m-and-also", ["It loses 10 boxes to theft and 5 boxes to fire."])
_mu("m-pct-over-100", ["150 percent of the stock (rounded down) spoils."], extra={"0", "-500"})
_mu("m-two-shares", ["The stock is split equally into 3 warehouses and it keeps two shares (rounded down)."])
_mu("m-split-up", ["The stock is split equally into 3 warehouses and it keeps one share (rounded up)."])
_mv("m-days-1", ["Every day for 1 days, 5 boxes leave the warehouse."], 995)
_mv("m-zero-deliveries", ["It receives 0 deliveries of 5 boxes each."], 1000)
_mv("m-pct-100", ["The stock grows by 100 percent (rounded down)."], 2000)

# ============================================================================ harness
def _write_ws(root: Path, kind: str) -> Path:
    ws = root / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    if kind == "data":
        with (ws / "sales.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["order_id", "region", "product", "month", "units", "price"])
            w.writerows(ROWS)
    elif kind == "search":
        kb = ws / "kb"
        kb.mkdir(exist_ok=True)
        for name, fields in KB.items():
            slug = "".join(c if c.isalnum() else "_" for c in name.casefold())
            (kb / f"{slug}.md").write_text(
                f"# {name}\n\n" + "".join(f"{k}: {v}\n" for k, v in fields.items()), encoding="utf-8")
    return ws


_SOLVER = {"math": math_solver, "data": data_solver, "search": search_solver}
_TYPE = {"math": "math", "data": "data_analysis", "search": "search"}


def outcome(case: dict, root: Path) -> str:
    ws = _write_ws(root, case["kind"])
    task = Task(case["id"], _TYPE[case["kind"]], case["text"], 1.0, case["check"], root, ws)
    res = _SOLVER[case["kind"]].solve(task)
    if res.status == "solved":
        return str(res.answer)
    return {"unparsed": U, "infeasible": I}[res.status]


def _params():
    out = []
    for c in CASES:
        marks = [pytest.mark.xfail(strict=True, reason=XFAIL[c["id"]])] if c["id"] in XFAIL else []
        out.append(pytest.param(c, id=c["id"], marks=marks))
    return out


def test_corpus_size():
    assert len(CASES) >= 120
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["kind"] for c in CASES} == {"math", "data", "search"}
    assert set(XFAIL) <= {c["id"] for c in CASES}


@pytest.mark.parametrize("case", _params())
def test_no_silent_wrong(case, tmp_path):
    got = outcome(case, tmp_path)
    assert got in case["ok"], f"{got!r} not in {sorted(case['ok'])}: {case['text'][:300]}"


def misses(tmp_root: Path) -> list[tuple[dict, str]]:
    """Acceptable-but-not-ideal outcomes (organizer-plausible phrasings the parsers miss)."""
    out = []
    for i, c in enumerate(CASES):
        d = tmp_root / f"m{i}"
        d.mkdir()
        got = outcome(c, d)
        if got in c["ok"] and c["want"] is not None and got != c["want"]:
            out.append((c, got))
    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        for c, got in misses(Path(t)):
            print(f"{c['id']}: want {c['want']!r} got {got!r}: {c['text'].split(' Use ')[0].split(' Compute it')[0][-170:]}")

"""Phase-1 instruction normalizer (task 29): free wording is rewritten to the canonical form the solvers parse;
anything that could change the meaning stays as it is and the solver answers `unparsed`."""
import pytest

from harness.result import SolveResult
from harness.solvers import data_solver, math_solver, search_solver
from harness.solvers.common import normalize_instruction, split_sentences

M_SUF = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, "
         "write ONLY the final answer to the file answer.txt in the current directory.")
D_SUF = (" Use bash with python3 to compute it. Work only inside the current directory. When you are done, "
         "write ONLY the final answer to the file answer.txt in the current directory.")
S_SUF = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. "
         "When you are done, write ONLY the final answer to the file answer.txt in the current directory.")

FLAT_M = ("A warehouse starts with 1000 boxes. It loses 5 boxes to theft. Then the stock is doubled by merging "
          "with a sister warehouse. How many boxes does it have at the end?")


def _unparsed(r):
    return isinstance(r, SolveResult) and r.status == "unparsed"


# ------------------------------------------------------------------ generic form

def test_bullets_become_sentences():
    text = ("- A warehouse starts with 1000 boxes.\n* It loses 5 boxes to theft\n• Then the stock is doubled.\n"
            "1. Done\nHow many?")
    out = normalize_instruction(text, "math")
    assert split_sentences(out)[:2] == ["A warehouse starts with 1000 boxes.", "It loses 5 boxes to theft."]
    assert "- " not in out and "•" not in out


def test_canonical_input_is_unchanged():
    assert normalize_instruction(FLAT_M + M_SUF, "math") == FLAT_M


def test_idempotent():
    text = ("- Quick note on inventory: A warehouse starts with 1000 boxes.\n- It loses 5 boxes to theft.\n"
            "How many boxes does it have at the end?")
    once = normalize_instruction(text, "math")
    assert normalize_instruction(once, "math") == once


# ------------------------------------------------------------------ math

def test_math_bullets_parse_like_flat():
    text = ("- A warehouse starts with 1000 boxes.\n- It loses 5 boxes to theft.\n"
            "- Then the stock is doubled by merging with a sister warehouse.\n"
            "How many boxes does it have at the end?" + M_SUF)
    assert isinstance(math_solver.parse(text), dict)
    assert math_solver.parse(text) == math_solver.parse(FLAT_M + M_SUF)


@pytest.mark.parametrize("pre", ["Quick note: ", "FYI: ", "Consider the following. ", "Please review the following task. ",
                                 "Here is a task: "])
def test_math_filler_preamble_dropped(pre):
    assert isinstance(math_solver.parse(pre + FLAT_M + M_SUF), dict)
    assert math_solver.parse(pre + FLAT_M + M_SUF) == math_solver.parse(FLAT_M + M_SUF)


@pytest.mark.parametrize("pre", ["Imagine a logistics center managing inventory. ", "Quick note on inventory: ",
                                 "Before that, ", "Finally, "])
def test_math_domain_words_and_ordering_are_not_filler(pre):
    text = FLAT_M.replace("It loses 5 boxes to theft.", pre + "it loses 5 boxes to theft.") if pre.endswith(", ") else pre + FLAT_M
    r = math_solver.parse(text + M_SUF)
    assert _unparsed(r) or pre == "Finally, "


@pytest.mark.parametrize("pre", ["Ignore the shipments. ", "Note that 5 boxes are missing. ",
                                 "Only count the first 3 steps. ", "Do not apply the theft. ",
                                 "Quick note on the 'north' warehouse: "])
def test_math_meaningful_preamble_kept(pre):
    assert _unparsed(math_solver.parse(pre + FLAT_M + M_SUF))


@pytest.mark.parametrize("start", ["Initial stock: 1000 boxes.", "Starting inventory: 1000 boxes.",
                                   "The warehouse starts with 1000 boxes.", "Our warehouse begins with 1000 boxes."])
def test_math_start_variants(start):
    text = FLAT_M.replace("A warehouse starts with 1000 boxes.", start)
    assert isinstance(math_solver.parse(text + M_SUF), dict)
    assert math_solver.parse(text + M_SUF) == math_solver.parse(FLAT_M + M_SUF)


# ------------------------------------------------------------------ data

CANON_D = ("The file sales.csv has columns order_id,region,product,month,units,price (636 rows). "
           "What is the average price in month 8, rounded to 2 decimal places?")


def _same_d(text):
    want = data_solver.parse(CANON_D + D_SUF)
    return isinstance(want, dict) and data_solver.parse(text + D_SUF) == want


def test_data_canonical_parses():
    assert isinstance(data_solver.parse(CANON_D + D_SUF), dict)


@pytest.mark.parametrize("text", [
    "Here is a brief overview of the dataset structure. " + CANON_D,
    "- File: sales.csv\n- Columns: order_id, region, product, month, units, price\n- Row count: 636\n\n"
    "What is the average price in month 8, rounded to 2 decimal places?",
    "Looking at sales.csv (order_id, region, product, month, units, price; 636 rows), what's the mean price in month 8, "
    "rounded to 2 decimal places?",
    "Quick query on the sales.csv file (columns: order_id,region,product,month,units,price; 636 rows). "
    "What is the average price in month 8, rounded to 2 decimal places?",
    "The dataset sales.csv includes 636 rows with columns order_id, region, product, month, units, and price. "
    "What is the average price in month 8, rounded to 2 decimal places?",
    "Please review the file sales.csv, which includes 636 rows and columns order_id,region,product,month,units,price. "
    "Given this context, what is the average price in month 8, rounded to 2 decimal places?",
    "Given that the file sales.csv contains 636 rows with columns order_id, region, product, month, units, and price, "
    "could you calculate the average price in month 8, rounded to 2 decimal places?",
    "Regarding sales.csv (636 rows, cols: order_id,region,product,month,units,price), could you please calculate the "
    "average price in month 8 rounded to 2 decimal places?",
])
def test_data_variants_equal_canonical(text):
    assert _same_d(text)


def test_data_units_times_price():
    text = ("Looking at sales.csv (order_id, region, product, month, units, price; 1740 rows), "
            "what's the total revenue (units * price)?" + D_SUF)
    spec = data_solver.parse(text)
    assert isinstance(spec, dict) and spec["metric"] == "sum_revenue"


def test_data_total_number_of_orders():
    text = ("Please check the sales.csv file (1787 rows, columns: order_id,region,product,month,units,price) "
            "and tell me the total number of orders placed." + D_SUF)
    spec = data_solver.parse(text)
    assert isinstance(spec, dict) and spec["metric"] == "count_orders"


def test_data_restating_sentences_are_kept_so_unparsed():
    text = ("- The file sales.csv has columns order_id,region,product,month,units,price (327 rows).\n"
            "- Focus on 'sprocket' units in month 4.\n- Compute the mean value.\n"
            "What is the average number of units per order in month 4 for product 'sprocket', "
            "rounded to 2 decimal places?")
    assert _unparsed(data_solver.parse(text + D_SUF))


@pytest.mark.parametrize("filler", [
    "We are focusing on product 'sprocket' in months 11 through 11 inclusive.",  # the filter is only in the filler
    "Focus on the north region.",
    "Only count orders above 10 units.",
    "Ignore month 3.",
    "We are focusing on product 'gizmo'.",  # conflicts with the question
])
def test_data_filler_with_a_filter_is_never_dropped(filler):
    text = ("- The file sales.csv has columns order_id,region,product,month,units,price (747 rows).\n- " + filler +
            "\nWhat is the largest number of units in a single order for product 'sprocket'?")
    assert _unparsed(data_solver.parse(text + D_SUF))


def test_data_two_files_unparsed():
    text = ("The file a.csv has columns order_id,region,product,month,units,price (5 rows). "
            "The file b.csv has columns order_id,region,product,month,units,price (6 rows). "
            "How many orders were placed?")
    assert _unparsed(data_solver.parse(text + D_SUF))


def test_data_trailing_clause_after_file_kept():
    text = ("The file sales.csv has columns order_id,region,product,month,units,price (5 rows) but only for the north "
            "region. How many orders were placed?")
    assert _unparsed(data_solver.parse(text + D_SUF))


# ------------------------------------------------------------------ search

def test_search_intro_variants():
    want = search_solver.parse("In which year was Petra Ivanova's mentor born?" + S_SUF)
    assert isinstance(want, dict)
    for intro in ["The directory kb/ contains a small knowledge base of markdown files.",
                  "Check out the kb/ folder for some markdown files.",
                  "The kb/ directory holds a small knowledge base in markdown format.",
                  "FYI, the directory kb/ contains a small knowledge base of markdown files."]:
        assert search_solver.parse(intro + " In which year was Petra Ivanova's mentor born?" + S_SUF) == want, intro
    assert search_solver.parse("- The kb/ directory contains a small knowledge base of markdown files.\n"
"\n"
                               "In which year was Petra Ivanova's mentor born?" + S_SUF) == want


def test_search_sentence_after_question_is_never_dropped():
    q = "In which year was Petra Ivanova's mentor born?"
    note = "(Note: The directory kb/ contains a small knowledge base of markdown files.)"
    assert _unparsed(search_solver.parse(q + " " + note + S_SUF))
    assert _unparsed(search_solver.parse(q + " Then find the employer of that mentor." + S_SUF))


def test_search_polite_prefix_and_imperative():
    want = search_solver.parse("In which year was Petra Ivanova's mentor born?" + S_SUF)
    assert search_solver.parse("The kb/ directory holds a small knowledge base of markdown files. Given this context, "
                               "in which year was Petra Ivanova's mentor born?" + S_SUF) == want
    spec = search_solver.parse("- The kb/ directory contains a small knowledge base of markdown files.\n"
"\n"
                               "- Identify the mentor of the mentor of Lena Ito." + S_SUF)
    assert isinstance(spec, dict)
    assert spec == search_solver.parse("Who is the mentor of the mentor of Lena Ito?" + S_SUF)
    spec = search_solver.parse("Please tell me which company employs the mentor of Bruno-Dmitri Moreau." + S_SUF)
    assert isinstance(spec, dict)
    assert spec == search_solver.parse("Which company employs the mentor of Bruno-Dmitri Moreau?" + S_SUF)


def test_search_step_sentences_are_kept_so_unparsed():
    q = "What year was the employer of the mentor of Rosa Haddad founded?"
    base = "The kb/ directory contains a small knowledge base of markdown files.\n"
    for step in ["Identify Rosa Haddad's mentor.", "Identify Boris Garcia's mentor.", "Ignore anyone who left.",
                 "Only use the file kb/people.md."]:
        assert _unparsed(search_solver.parse(base + step + "\n" + q + S_SUF)), step


def test_search_wrong_directory_not_dropped():
    assert _unparsed(search_solver.parse("The data/ directory contains a small knowledge base of markdown files. "
                                         "In which year was Petra Ivanova's mentor born?" + S_SUF))


def test_long_comma_list_is_fast():
    import time
    text = "A warehouse starts with " + ", ".join(["x"] * 5000) + " boxes. How many boxes does it have at the end?"
    t = time.time()
    math_solver.parse(text + M_SUF)
    data_solver.parse(text + D_SUF)
    search_solver.parse(text + S_SUF)
    assert time.time() - t < 1.0


# ------------------------------------------------------------------ adversarial gate (84 probes from the review)

import json
from pathlib import Path

from harness.tasks import Task

_FIX = Path(__file__).parent / "fixtures" / "probes_r6"
_PROBES = json.loads((_FIX / "probes.json").read_text(encoding="utf-8"))
_SUFFIX = {"data": D_SUF, "math": M_SUF, "search": S_SUF}
_MOD = {"data": (data_solver, "data_analysis"), "math": (math_solver, "math"), "search": (search_solver, "search")}
_WS = {"data": _FIX / "data", "north": _FIX / "north", "search": _FIX / "search"}


@pytest.mark.parametrize("probe", _PROBES, ids=[p["id"] for p in _PROBES])
def test_probe_never_confidently_wrong(probe, tmp_path):
    mod, typ = _MOD[probe["dom"]]
    ws = _WS[probe["ws"]]
    r = mod.solve(Task(probe["id"], typ, probe["text"] + _SUFFIX[probe["dom"]], 1.0, "exact_str", ws, ws))
    if r.status == "solved":
        assert r.answer in probe["allowed"], (probe["note"], r.answer)

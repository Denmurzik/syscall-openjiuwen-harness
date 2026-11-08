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


@pytest.mark.parametrize("pre", ["Quick note on inventory: ", "FYI on stock levels: ",
                                 "Imagine a logistics center managing inventory. ",
                                 "Consider the logistics of a busy warehouse. ", "Let's look at a distribution update. "])
def test_math_filler_preamble_dropped(pre):
    assert isinstance(math_solver.parse(pre + FLAT_M + M_SUF), dict)
    assert math_solver.parse(pre + FLAT_M + M_SUF) == math_solver.parse(FLAT_M + M_SUF)


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
    "- The file sales.csv has columns order_id,region,product,month,units,price (636 rows).\n"
    "- Calculate the mean value.\nWhat is the average price in month 8, rounded to 2 decimal places?",
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


def test_data_redundant_filler_with_contained_values_is_dropped():
    text = ("- The file sales.csv has columns order_id,region,product,month,units,price (327 rows).\n"
            "- Focus on 'sprocket' units in month 4.\n- Compute the mean value.\n"
            "What is the average number of units per order in month 4 for product 'sprocket', "
            "rounded to 2 decimal places?")
    spec = data_solver.parse(text + D_SUF)
    assert isinstance(spec, dict) and spec["product"] == "sprocket" and spec["month_from"] == 4


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
                  "You can find a collection of markdown files in the kb/ directory.",
                  "Check out the kb/ folder for some markdown files.",
                  "The kb/ directory holds a small knowledge base in markdown format.",
                  "FYI, the directory kb/ contains a small knowledge base of markdown files."]:
        assert search_solver.parse(intro + " In which year was Petra Ivanova's mentor born?" + S_SUF) == want, intro
    assert search_solver.parse("- The kb/ directory contains a small knowledge base of markdown files.\n"
                               "In which year was Petra Ivanova's mentor born?" + S_SUF) == want
    assert search_solver.parse("In which year was Petra Ivanova's mentor born? (Note: The directory kb/ contains a "
                               "small knowledge base of markdown files.)" + S_SUF) == want


def test_search_polite_prefix_and_imperative():
    want = search_solver.parse("In which year was Petra Ivanova's mentor born?" + S_SUF)
    assert search_solver.parse("The kb/ directory holds a small knowledge base of markdown files. Given this context, "
                               "in which year was Petra Ivanova's mentor born?" + S_SUF) == want
    spec = search_solver.parse("- The kb/ directory contains a small knowledge base of markdown files.\n"
                               "- Identify the mentor of the mentor of Lena Ito." + S_SUF)
    assert isinstance(spec, dict)
    assert spec == search_solver.parse("Who is the mentor of the mentor of Lena Ito?" + S_SUF)
    spec = search_solver.parse("Please tell me which company employs the mentor of Bruno-Dmitri Moreau." + S_SUF)
    assert isinstance(spec, dict)
    assert spec == search_solver.parse("Which company employs the mentor of Bruno-Dmitri Moreau?" + S_SUF)


def test_search_redundant_steps_dropped_only_when_contained():
    q = "What year was the employer of the mentor of Rosa Haddad founded?"
    steps = ("- Identify Rosa Haddad's mentor.\n- Determine the employer of that mentor.\n"
             "- Find the founding year of that employer.\n")
    base = "- Locate the knowledge base in the kb/ directory, which consists of markdown files.\n"
    want = search_solver.parse(q + S_SUF)
    assert isinstance(want, dict)
    assert search_solver.parse(base + steps + q + S_SUF) == want
    assert _unparsed(search_solver.parse(base + "- Identify Boris Garcia's mentor.\n" + q + S_SUF))
    assert _unparsed(search_solver.parse(base + "- Ignore anyone who left the company.\n" + q + S_SUF))
    assert _unparsed(search_solver.parse(base + "- Only use the file kb/people.md.\n" + q + S_SUF))


def test_search_wrong_directory_not_dropped():
    assert _unparsed(search_solver.parse("The data/ directory contains a small knowledge base of markdown files. "
                                         "In which year was Petra Ivanova's mentor born?" + S_SUF))

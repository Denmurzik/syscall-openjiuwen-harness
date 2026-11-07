"""Synthetic task generator in the organizers' format (math / search / data_analysis / coding).

Only for end-to-end runs and robustness checks: its feasible templates match the solver grammars,
so phase-1 accuracy measured on it proves nothing.
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

TYPES = ["math", "search", "data_analysis", "coding"]

SUFFIX_MATH = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. "
               "When you are done, write ONLY the final answer to the file answer.txt in the current "
               "directory (for example with bash: echo 42 > answer.txt).")
SUFFIX_SEARCH = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current "
                 "directory. When you are done, write ONLY the final answer to the file answer.txt in the "
                 "current directory (for example with bash: echo 42 > answer.txt).")
SUFFIX_DATA = (" Use bash with python3 to compute it. Work only inside the current directory. When you are "
               "done, write ONLY the final answer to the file answer.txt in the current directory "
               "(for example with bash: echo 42 > answer.txt).")
CODING_INSTRUCTION = (
    "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
    "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are "
    "done, write ONLY the word DONE to the file answer.txt in the current directory (for example with "
    "bash: echo 42 > answer.txt).")


# ---------------------------------------------------------------- math

def gen_math(rng, novel, infeasible):
    start = rng.randint(200, 2500)
    stock = start
    sentences = [f"A warehouse starts with {start} boxes."]
    n_ops = rng.randint(2, 6)
    kinds = ["ship", "receive", "split", "double"]
    novel_kinds = ["theft", "ruined", "tripled", "moved"]
    plan = [rng.choice(kinds) for _ in range(n_ops)]
    if novel:
        plan[rng.randrange(n_ops)] = rng.choice(novel_kinds)
    bad_split = rng.randrange(n_ops) if infeasible else -1
    if infeasible:
        plan[bad_split] = "split"
    feasible = True
    for i, kind in enumerate(plan):
        if kind == "ship":
            days = rng.randint(2, 5)
            per = rng.randint(5, 60)
            if per * days > stock:
                per = max(1, stock // days)
            stock -= per * days
            sentences.append(f"It ships {per} boxes per day for {days} days.")
        elif kind == "receive":
            k, b = rng.randint(2, 4), rng.randint(20, 120)
            stock += k * b
            sentences.append(f"It receives {k} deliveries of {b} boxes each.")
        elif kind == "split":
            k = 0 if i == bad_split else rng.randint(2, 4)
            if k == 0:
                feasible = False
            else:
                stock //= k
            sentences.append(f"Then the stock is split equally into {k} warehouses and it keeps one "
                             f"share (rounded down).")
        elif kind == "double":
            stock *= 2
            sentences.append("Then the stock is doubled by merging with a sister warehouse.")
        elif kind == "theft":
            n = min(rng.randint(5, 80), stock)
            stock -= n
            sentences.append(f"It loses {n} boxes to theft.")
        elif kind == "ruined":
            n = min(rng.randint(5, 80), stock)
            stock -= n
            sentences.append(f"A leak ruins {n} boxes, which are thrown away.")
        elif kind == "tripled":
            stock *= 3
            sentences.append("Then the stock is tripled by a big merger.")
        elif kind == "moved":
            stock -= stock // 4
            sentences.append("Then one quarter of the stock (rounded down) is moved to another site.")
    if novel:
        question = rng.choice(["What is the final stock level in boxes?", "How many boxes remain in the end?"])
    else:
        question = "How many boxes does it have at the end?"
    return {
        "instruction": " ".join(sentences) + " " + question + SUFFIX_MATH,
        "expected": str(stock) if feasible else None,
        "check": "exact_int",
    }


# ---------------------------------------------------------------- data analysis

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]


def _sales(rng, n):
    return [(i + 1, rng.choice(REGIONS), rng.choice(PRODUCTS), rng.randint(1, 12),
             rng.randint(1, 40), rng.randint(5, 99)) for i in range(n)]


def gen_data(rng, novel, infeasible, ws):
    n = rng.choice([20, 300])
    while True:
        rows = _sales(rng, n)
        by_prod = {}
        for r in rows:
            by_prod[r[2]] = by_prod.get(r[2], 0) + r[4]
        top = sorted(by_prod.values(), reverse=True)
        if len(by_prod) == len(PRODUCTS) and top[0] != top[1]:
            break
    (ws / "sales.csv").write_text(
        "order_id,region,product,month,units,price\n"
        + "".join(",".join(map(str, r)) + "\n" for r in rows), encoding="utf-8", newline="\n")
    head = f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). "
    a = rng.randint(1, 11)
    b = rng.randint(a, min(12, a + 3))
    product = rng.choice(PRODUCTS)
    region = rng.choice(REGIONS)
    month = rng.randint(1, 12)
    if infeasible:
        product = "doohickey"
    if not novel or infeasible:
        q = f"How many units of product '{product}' were sold in months {a} through {b} inclusive?"
        exp = None if infeasible else str(sum(r[4] for r in rows if r[2] == product and a <= r[3] <= b))
        check = "exact_int"
    else:
        v = rng.randrange(4)
        if v == 0:
            q = (f"What was the total number of '{product}' units sold between month {a} and "
                 f"month {b} (both included)?")
            exp, check = str(sum(r[4] for r in rows if r[2] == product and a <= r[3] <= b)), "exact_int"
        elif v == 1:
            q = f"What is the total revenue (units multiplied by price) for region '{region}'?"
            exp, check = str(sum(r[4] * r[5] for r in rows if r[1] == region)), "exact_int"
        elif v == 2:
            q = f"How many orders were placed in month {month}?"
            exp, check = str(sum(1 for r in rows if r[3] == month)), "exact_int"
        else:
            q = "Which product sold the most units overall?"
            exp, check = max(by_prod, key=by_prod.get), "exact_str"
    return {"instruction": head + q + SUFFIX_DATA, "expected": exp, "check": check}


# ---------------------------------------------------------------- search

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya",
         "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Garcia", "Haddad", "Ivanova", "Kowalski", "Muller", "Novak", "Petrov", "Sato", "Smith", "Wang"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Medan", "Tbilisi", "Hanoi", "Osaka", "Quito", "Lyon", "Porto", "Tartu", "Bergen", "Kobe"]


def gen_search(rng, novel, infeasible, ws):
    all_names = [f"{f} {l}" for f in FIRST for l in LAST]
    people = rng.sample(all_names, rng.randint(34, 40))
    absent = [x for x in all_names if x not in people]
    comp = {c: {"city": rng.choice(CITIES), "founded": rng.randint(1950, 2020)} for c in COMPANIES}
    info = {}
    for p in people:
        info[p] = {"born": rng.randint(1950, 2005), "employer": rng.choice(COMPANIES),
                   "mentor": rng.choice([x for x in people if x != p])}
    kb = ws / "kb"
    kb.mkdir(parents=True)
    for p, d in info.items():
        (kb / ("person_" + p.lower().replace(" ", "_") + ".md")).write_text(
            f"# {p}\n\nBorn: {d['born']}\nEmployer: {d['employer']}\nMentor: {d['mentor']}\n",
            encoding="utf-8", newline="\n")
    for c, d in comp.items():
        (kb / ("company_" + c.lower().replace(" ", "_") + ".md")).write_text(
            f"# {c}\n\nHeadquarters: {d['city']}\nFounded: {d['founded']}\n", encoding="utf-8", newline="\n")
    p = rng.choice(absent) if infeasible else rng.choice(people)
    d = info.get(p)
    mentor = info[d["mentor"]] if d else None
    if not novel:
        if rng.random() < 0.5:
            q, exp = f"In which year was {p} born?", d and str(d["born"])
        else:
            q, exp = (f"In which city is the headquarters of {p}'s employer?",
                      d and comp[d["employer"]]["city"])
    else:
        v = rng.randrange(5)
        if v == 0:
            q, exp = f"Who is the mentor of {p}?", d and d["mentor"]
        elif v == 1:
            q, exp = f"In which year was {p}'s employer founded?", d and str(comp[d["employer"]]["founded"])
        elif v == 2:
            q, exp = f"In which city does the mentor of {p} work?", d and comp[mentor["employer"]]["city"]
        elif v == 3:
            q, exp = f"What is the birth year of {p}'s mentor?", d and str(mentor["born"])
        else:
            q, exp = f"Can you tell me the year {p} was born?", d and str(d["born"])
    return {"instruction": "The directory kb/ contains a small knowledge base of markdown files. "
                           + q + SUFFIX_SEARCH,
            "expected": exp, "check": "exact_str"}


# ---------------------------------------------------------------- coding

# name -> (correct source, 4 argument strings; each gives one assert)
LIB = {
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              ["[1, 2, 3]", "[]", "[10, -4, 7]", "[5, 5, 5, 5]"]),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n"
                "            best = x\n    return best\n",
                ["[3, 9, 2]", "[-5, -2, -9]", "[7]", "[1, 8, 8, 3]"]),
    "minimum": ("def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n"
                "            best = x\n    return best\n",
                ["[3, 9, 2]", "[-5, -2, -9]", "[7]", "[4, 1, 1, 6]"]),
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      ["'c a b'", "'one two'", "'solo'", "'x y z w'"]),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   ["[1, 2, 4, 6, 7]", "[]", "[3, 5, 9]", "[0, 10, 11, 12]"]),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                ["[2, 4, 6]", "[10]", "[1, 2, 3, 4]", "[5, 5, 8, 2]"]),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                ["[5, 6, 7, 8], 3", "[1, 2], 5", "[9, 8, 7], 1", "[4, 3, 2, 1], 4"]),
    "last_n": ("def last_n(xs, n):\n    return xs[len(xs) - n:]\n",
               ["[5, 6, 7, 8], 2", "[1, 2, 3], 3", "[9, 8, 7, 6, 5], 1", "[4, 3, 2, 1], 4"]),
    "is_palindrome": ("def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n",
                      ["'Level'", "'abc'", "'noon'", "'ab'"]),
    "factorial": ("def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
                  ["5", "0", "1", "7"]),
    "fibonacci": ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
                  "    return a\n", ["6", "0", "1", "10"]),
    "gcd": ("def gcd(a, b):\n    for d in range(min(a, b), 0, -1):\n        if a % d == 0 and b % d == 0:\n"
            "            return d\n    return 1\n", ["12, 18", "7, 13", "20, 5", "9, 27"]),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
                     ["'Hello World'", "'xyz'", "'AEIOU'", "'banana'"]),
    "clamp": ("def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n"
              "    return x\n", ["5, 0, 10", "-3, 0, 10", "15, 0, 10", "7, 7, 9"]),
    "dedupe": ("def dedupe(xs):\n    seen = []\n    for x in xs:\n        if x not in seen:\n"
               "            seen.append(x)\n    return seen\n",
               ["[1, 2, 1, 3, 2]", "[]", "[4, 4, 4]", "[5, 6, 7]"]),
    "sum_of_squares": ("def sum_of_squares(xs):\n    s = 0\n    for x in xs:\n        s += x * x\n    return s\n",
                       ["[1, 2, 3]", "[]", "[4]", "[-2, 3]"]),
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, n):\n"
                 "        if n % d == 0:\n            return False\n    return True\n",
                 ["7", "9", "2", "1"]),
    "sort_desc": ("def sort_desc(xs):\n    return sorted(xs, reverse=True)\n",
                  ["[3, 1, 2]", "[]", "[5, 5, 1]", "[9, 8, 10]"]),
    "second_largest": ("def second_largest(xs):\n    vals = sorted(set(xs))\n    return vals[-2]\n",
                       ["[3, 9, 2]", "[1, 5, 5, 4]", "[10, 20]", "[7, 3, 8, 8]"]),
    "capitalize_words": ("def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
                         ["'hello big world'", "'a'", "'mIxEd cAsE'", "'one two'"]),
}

# bug injection: operator swaps, comparison flips, parity flips, sorted <-> reversed, +-1 on constants/slices
_MUT_RULES = [
    (r"\+=", "-="), (r"-=", "+="),
    (r" \+ ", " - "), (r" - ", " + "),
    (r" < ", " > "), (r" > ", " < "),
    (r"== 0", "== 1"), (r"== 1", "== 0"),
    (r"\bsorted\(", "reversed("), (r"\breversed\(", "sorted("),
    (r"\[:(\w+)\]", r"[:\1 - 1]"), (r"\[:(\w+)\]", r"[:\1 + 1]"),
    (r"\b(\d+)\b", None),
]


def _call(src, name, args):
    ns = {}
    exec(compile(src, "<lib>", "exec"), ns)
    return eval(f"f({args})", {"f": ns[name]})


def _mutants(src):
    """All distinct single-site mutations of `src`."""
    out = []
    for pat, rep in _MUT_RULES:
        for m in re.finditer(pat, src):
            reps = [str(int(m.group(1)) + 1), str(int(m.group(1)) - 1)] if rep is None else [m.expand(rep)]
            for r in reps:
                new = src[:m.start()] + r + src[m.end():]
                if new != src and new not in out:
                    out.append(new)
    return out


def correct_library(names):
    return "\n\n".join(LIB[n][0].rstrip("\n") + "\n" for n in names)


def _expected(name, args):
    return _call(LIB[name][0], name, args)


def _killing_mutant(rng, name):
    """A mutant failing at least two cases (so the visible one and a hidden one both catch it),
    else one failing a single case: (source, indices of failing cases) or None."""
    cases = LIB[name][1]
    good = [_expected(name, a) for a in cases]
    cands = _mutants(LIB[name][0])
    rng.shuffle(cands)
    fallback = None
    for c in cands:
        killed = []
        for i, a in enumerate(cases):
            try:
                same = _call(c, name, a) == good[i]
            except Exception:
                same = False
            if not same:
                killed.append(i)
        if len(killed) >= 2:
            return c, killed
        if killed and fallback is None:
            fallback = (c, killed)
    return fallback


def _contradict(v):
    if v is None:  # e.g. max_key({}) legitimately returns None
        return 0
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "!"
    return list(v) + [0]


def gen_coding(rng, novel, infeasible, ws):
    names = rng.sample(sorted(LIB), rng.randint(3, 5))
    mutable = [n for n in names if _mutants(LIB[n][0])]
    buggy = set(rng.sample(mutable, min(len(mutable), rng.randint(1, 2))))
    sources, visible_idx = {}, {}
    for n in names:
        m = _killing_mutant(rng, n) if n in buggy else None
        if m:
            sources[n] = m[0]
            visible_idx[n] = rng.choice(m[1])
        else:
            sources[n] = LIB[n][0]
            visible_idx[n] = rng.randrange(4)
    (ws / "mylib.py").write_text("\n\n".join(sources[n].rstrip("\n") + "\n" for n in names),
                                 encoding="utf-8", newline="\n")
    contra = rng.choice(names) if infeasible else None
    head = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    vis, hid = list(head), list(head)
    for n in names:
        cases = LIB[n][1]
        vi = visible_idx[n]
        vis += [f"    def test_{n}(self):",
                f"        self.assertEqual(m.{n}({cases[vi]}), {_expected(n, cases[vi])!r})"]
        if n == contra:
            vis.append(f"        self.assertEqual(m.{n}({cases[vi]}), "
                       f"{_contradict(_expected(n, cases[vi]))!r})")
        vis.append("")
        hid.append(f"    def test_{n}(self):")
        hid += [f"        self.assertEqual(m.{n}({a}), {_expected(n, a)!r})"
                for i, a in enumerate(cases) if i != vi]
        hid.append("")
    tail = ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    (ws / "test_mylib.py").write_text("\n".join(vis + tail), encoding="utf-8", newline="\n")
    return {"instruction": CODING_INSTRUCTION, "expected": None if infeasible else "tests_pass",
            "check": "unittest", "_hidden": None if infeasible else "\n".join(hid + tail)}


# ================================================================ v2: medium / hard generators
# Conventions: "sells half (rounded down)" = the NUMBER SOLD is floor(s/2), so the remaining stock is
# s - s//2 (the odd box stays). Percent growth/loss: the changed amount is floor(s*p/100).
# Every v2 generator draws only from the per-task `rng` it is given, so a task never depends on
# its neighbours and a set of any size is generated with the same per-task content for a seed.

DIFFS = ("easy", "medium", "hard")

# ---------------------------------------------------------------- math v2

_M_PHRASES = {
    "ship": ("It ships {per} boxes per day for {days} days.",
             ["Every day for {days} days, {per} boxes leave the warehouse.",
              "For {days} days in a row, {per} boxes are shipped out daily."]),
    "receive": ("It receives {k} deliveries of {b} boxes each.",
                ["{k} trucks arrive, each carrying {b} boxes.",
                 "The warehouse takes in {k} shipments of {b} boxes apiece."]),
    "split": ("Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).",
              ["The stock is then divided evenly among {k} warehouses, and this one retains a single "
               "share (rounded down)."]),
    "double": ("Then the stock is doubled by merging with a sister warehouse.",
               ["A merger with a sister warehouse then doubles the stock."]),
    "triple": ("Then the stock is tripled by a big merger.", ["Then a big merger triples the stock."]),
    "theft": ("It loses {n} boxes to theft.", ["Thieves take {n} boxes."]),
    "ruined": ("A leak ruins {n} boxes, which are thrown away.", ["Water damage destroys {n} boxes."]),
    "moved": ("Then one quarter of the stock (rounded down) is moved to another site.",
              ["Then a quarter of the stock, rounded down, is transferred to another site."]),
    "sell_half": ("Then it sells half of the boxes, rounding the number sold down.",
                  ["It then sells half of the stock, with the number of boxes sold rounded down."]),
    "add_pct": ("Then the stock grows by {p} percent (rounded down).",
                ["A seasonal restock adds {p} percent of the current stock, rounded down."]),
    "lose_pct": ("Then {p} percent of the stock (rounded down) spoils.",
                 ["Spoilage removes {p} percent of the stock, rounded down."]),
}
_M_QUESTIONS = ["How many boxes does it have at the end?", "What is the final stock level in boxes?",
                "How many boxes remain in the end?", "How many boxes are left at the end?"]


def gen_math_v2(rng, diff, novel, infeasible):
    hard = diff == "hard"
    start = rng.randint(2000, 60000) if hard else rng.randint(500, 9000)
    n_ops = rng.randint(4, 8) if hard else rng.randint(3, 6)
    p_para = 0.35 if hard else 0.15
    kinds = ["ship", "receive", "split", "double", "triple", "theft", "ruined", "moved", "sell_half",
             "add_pct", "lose_pct"]
    plan = [rng.choice(kinds) for _ in range(n_ops)]
    bad_split = rng.randrange(n_ops) if infeasible else -1
    if infeasible:
        plan[bad_split] = "split"
    stock = start
    sentences = [f"A warehouse starts with {start} boxes."]
    feasible = True
    forced_para = rng.randrange(n_ops) if novel else -1
    for i, kind in enumerate(plan):
        v: dict = {}
        if kind == "ship":
            v["days"] = rng.randint(2, 9 if hard else 6)
            if stock < v["days"]:
                kind = "receive"
            else:
                v["per"] = min(rng.randint(5, 400 if hard else 80), stock // v["days"])
                stock -= v["per"] * v["days"]
        if kind == "receive":
            v = {"k": rng.randint(2, 9 if hard else 5), "b": rng.randint(20, 900 if hard else 150)}
            stock += v["k"] * v["b"]
        elif kind == "split":
            v["k"] = 0 if i == bad_split else rng.randint(2, 6 if hard else 4)
            if v["k"] == 0:
                feasible = False
            else:
                stock //= v["k"]
        elif kind == "double":
            stock *= 2
        elif kind == "triple":
            stock *= 3
        elif kind in ("theft", "ruined"):
            v["n"] = min(rng.randint(5, 500 if hard else 90), stock)
            stock -= v["n"]
        elif kind == "moved":
            stock -= stock // 4
        elif kind == "sell_half":
            stock -= stock // 2
        elif kind == "add_pct":
            v["p"] = rng.randint(5, 75)
            stock += stock * v["p"] // 100
        elif kind == "lose_pct":
            v["p"] = rng.randint(5, 60)
            stock -= stock * v["p"] // 100
        canon, paras = _M_PHRASES[kind]
        if i == forced_para or rng.random() < p_para:
            sentences.append(rng.choice(paras).format(**v))
        else:
            sentences.append(canon.format(**v))
    q = rng.choice(_M_QUESTIONS) if (novel or rng.random() < p_para) else _M_QUESTIONS[0]
    return {"instruction": " ".join(sentences) + " " + q + SUFFIX_MATH,
            "expected": str(stock) if feasible else None, "check": "exact_int"}


# ---------------------------------------------------------------- data v2

CSV_NAMES = ["sales.csv"] * 7 + ["orders.csv", "sales_data.csv", "transactions.csv"]
COLS = "order_id,region,product,month,units,price"


def _div_round(a, b):
    """a / b rounded half up to 2 decimals, as a string, using integers only."""
    q = (200 * a + b) // (2 * b)
    return f"{q // 100}.{q % 100:02d}"


def gen_data_v2(rng, diff, novel, infeasible, ws):
    hard = diff == "hard"
    n = rng.randint(300, 2000) if hard else rng.randint(20, 500)
    rows = [(i + 1, rng.choice(REGIONS), rng.choice(PRODUCTS), rng.randint(1, 12),
             rng.randint(1, 60 if hard else 40), rng.randint(5, 250 if hard else 99)) for i in range(n)]
    fname = rng.choice(CSV_NAMES)
    metric = rng.choice(["units", "revenue", "orders", "avg_price", "avg_units", "max_units", "min_units"])
    clauses, preds = [], []  # clauses: (tag, prepositional phrase)
    nf = rng.choice([0, 1, 1, 2, 2, 3]) if hard else rng.choice([0, 1, 1, 2])
    for f in rng.sample(["product", "region", rng.choice(["month_range", "month"])], min(nf, 3)):
        if f == "product":
            x = rng.choice(PRODUCTS)
            clauses.append((f, f"for product '{x}'"))
            preds.append(lambda r, x=x: r[2] == x)
        elif f == "region":
            x = rng.choice(REGIONS)
            clauses.append((f, f"in region '{x}'"))
            preds.append(lambda r, x=x: r[1] == x)
        elif f == "month_range":
            a = rng.randint(1, 11)
            b = rng.randint(a, min(12, a + rng.randint(1, 5)))
            clauses.append((f, f"in months {a} through {b} inclusive"))
            preds.append(lambda r, a=a, b=b: a <= r[3] <= b)
        else:
            m = rng.randint(1, 12)
            clauses.append((f, f"in month {m}"))
            preds.append(lambda r, m=m: r[3] == m)
    bad = ""
    if infeasible:
        bad = rng.choice(["product", "region", "column", "file"])
        if bad == "product":
            clauses = [c for c in clauses if c[0] != "product"] + [("product", "for product 'doohickey'")]
        elif bad == "region":
            clauses = [c for c in clauses if c[0] != "region"] + [("region", "in region 'central'")]
        if bad in ("product", "region"):
            preds = [lambda r: False]
    sel = [r for r in rows if all(p(r) for p in preds)]
    where = "".join(" " + c[1] for c in clauses)
    check = "exact_int"
    exp = None
    if metric == "units":
        q = f"How many units were sold{where}?" if novel or rng.random() < 0.5 else \
            f"What is the total number of units sold{where}?"
        exp = str(sum(r[4] for r in sel))
    elif metric == "revenue":
        q = f"What is the total revenue (units multiplied by price){where}?"
        exp = str(sum(r[4] * r[5] for r in sel))
    elif metric == "orders":
        q = f"How many orders were placed{where}?"
        exp = str(len(sel))
    elif metric == "avg_price":
        q = f"What is the average price{where}, rounded to 2 decimal places?"
        exp, check = (_div_round(sum(r[5] for r in sel), len(sel)) if sel else None), "exact_str"
    elif metric == "avg_units":
        q = f"What is the average number of units per order{where}, rounded to 2 decimal places?"
        exp, check = (_div_round(sum(r[4] for r in sel), len(sel)) if sel else None), "exact_str"
    elif metric == "max_units":
        q = f"What is the largest number of units in a single order{where}?"
        exp = str(max(r[4] for r in sel)) if sel else None
    else:
        q = f"What is the smallest number of units in a single order{where}?"
        exp = str(min(r[4] for r in sel)) if sel else None
    if bad == "column":
        q = f"What is the total discount given{where}?"
        exp, check = None, "exact_str"
    if not sel and exp is not None and bad in ("product", "region"):
        exp = None  # a filter that matches nothing known to the file has no answer
    if bad in ("product", "region") and metric in ("units", "revenue", "orders"):
        exp = None
    if not infeasible and exp is None:  # empty selection by chance: ask the plain total instead
        q, exp, check = f"How many orders were placed{where}?", str(len(sel)), "exact_int"
    if bad != "file":
        (ws / fname).write_text(
            COLS + "\n" + "".join(",".join(map(str, r)) + "\n" for r in rows), encoding="utf-8", newline="\n")
    else:
        exp = None
        fname = "inventory.csv" if fname == "sales.csv" else "sales.csv"
    head = f"The file {fname} has columns {COLS} ({n} rows). "
    return {"instruction": head + q + SUFFIX_DATA, "expected": exp, "check": check}


# ---------------------------------------------------------------- search v2

FIRST2 = FIRST + ["Anna", "Bruno", "Carla", "Dario", "Elena", "Felix", "Greta", "Hugo", "Ivan", "Julia",
                  "Karim", "Lena", "Marco", "Nadia", "Omar", "Petra", "Rafael", "Sofia", "Tomas", "Uma",
                  "Viktor", "Wanda", "Yusuf", "Zoya"]
LAST2 = LAST + ["Almeida", "Bianchi", "Chowdhury", "Dubois", "Eriksen", "Fischer", "Gomez", "Horvat",
                "Ito", "Jansen", "Khan", "Lindqvist", "Moreau", "Nguyen", "Okafor", "Popescu", "Rossi",
                "Silva", "Tanaka", "Volkov"]
MIDDLE = ["A.", "B.", "C.", "D.", "E.", "J.", "K.", "L.", "M.", "R.", "S.", "T."]
CO_A = ["Arktika", "Bluefin", "Cobalt", "Dune", "Ember", "Fjord", "Granite", "Helix", "Iris", "Juniper",
        "Kestrel", "Lumen", "Meridian", "Nimbus", "Onyx", "Pioneer", "Quartz", "Raven", "Sable", "Tundra",
        "Umber", "Vector", "Willow", "Zenith"]
CO_B = ["Labs", "Systems", "Works", "Robotics", "Bio", "Data", "AI", "Motors", "Optics", "Foods", "Energy",
        "Logistics", "Media", "Networks"]
CITIES2 = CITIES + ["Cork", "Ghent", "Kaunas", "Lviv", "Nantes", "Oulu", "Plovdiv", "Rennes", "Split",
                    "Turku", "Utrecht", "Vilnius", "Zagreb", "Brno", "Graz", "Leeds", "Malmo", "Riga"]


def _slug(name):
    return re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "_"))


def _person_names(rng, k):
    names, slugs = [], set()
    while len(names) < k:
        f, l = rng.choice(FIRST2), rng.choice(LAST2)
        r = rng.random()
        if r < 0.10:
            nm = f"{f} {rng.choice(MIDDLE)} {l}"            # initials
        elif r < 0.17:
            nm = f"{f}-{rng.choice(FIRST2)} {l}"           # hyphenated first name
        elif r < 0.23:
            nm = f"{f} {l}-{rng.choice(LAST2)}"            # hyphenated last name
        else:
            nm = f"{f} {l}"
        if _slug(nm) not in slugs:
            slugs.add(_slug(nm))
            names.append(nm)
    return names


# attribute -> (lookups it costs on top of reaching the person, question template over a person phrase)
_S_ATTRS = {
    "born": (1, "In which year was {P} born?"),
    "mentor": (1, "Who is the mentor of {P}?"),
    "employer": (1, "Which company is the employer of {P}?"),
    "hq": (2, "In which city is the headquarters of the employer of {P}?"),
    "founded": (2, "In which year was the employer of {P} founded?"),
}
_S_ALT = {
    "born": ["Can you tell me the year {P} was born?", "What is the birth year of {P}?"],
    "hq": ["In which city does the employer of {P} have its headquarters?"],
    "founded": ["What year was the employer of {P} founded?"],
    "mentor": ["Name the mentor of {P}."],
    "employer": ["For which company does {P} work?"],
}


def gen_search_v2(rng, diff, novel, infeasible, ws):
    hard = diff == "hard"
    n_people = rng.randint(60, 150) if hard else rng.randint(20, 80)
    n_comp = rng.randint(20, 40) if hard else rng.randint(10, 25)
    people = _person_names(rng, n_people)
    cnames = []
    while len(cnames) < n_comp:
        c = f"{rng.choice(CO_A)} {rng.choice(CO_B)}"
        if rng.random() < 0.1:
            c = c.replace(" ", "-", 1) if rng.random() < 0.5 else c + " Co."
        if _slug(c) not in {_slug(x) for x in cnames}:
            cnames.append(c)
    comp = {c: {"city": rng.choice(CITIES2), "founded": rng.randint(1950, 2020)} for c in cnames}
    info = {p: {"born": rng.randint(1950, 2005), "employer": rng.choice(cnames),
                "mentor": rng.choice([x for x in people if x != p])} for p in people}
    hops = rng.randint(2, 4) if hard else rng.randint(1, 3)
    attrs = [a for a, (c, _) in _S_ATTRS.items() if 0 <= hops - c <= 3]
    attr = rng.choice(attrs)
    depth = hops - _S_ATTRS[attr][0]
    start = rng.choice(people)
    # walk the mentor chain, then resolve the attribute
    chain = [start]
    for _ in range(depth):
        chain.append(info[chain[-1]]["mentor"])
    last = chain[-1]
    d = info[last]
    ans = {"born": str(d["born"]), "mentor": d["mentor"], "employer": d["employer"],
           "hq": comp[d["employer"]]["city"], "founded": str(comp[d["employer"]]["founded"])}[attr]
    phrase = start
    for _ in range(depth):
        phrase = f"the mentor of {phrase}"
    tmpl = _S_ATTRS[attr][1]
    if (novel or rng.random() < 0.15) and attr in _S_ALT:
        tmpl = rng.choice(_S_ALT[attr])
    q = tmpl.format(P=phrase)
    drop_mentor, drop_company = None, None
    if infeasible:
        kind = rng.choice(["person", "mentor_link", "company"])
        if kind == "person" or (kind == "mentor_link" and depth == 0 and attr != "mentor"):
            kind = "person"
        if kind == "company" and attr not in ("hq", "founded"):
            kind = "person"
        if kind == "person":
            ghost = _person_names(rng, 1)[0]
            while ghost in info:
                ghost = _person_names(rng, 1)[0]
            q = tmpl.format(P=ghost if depth == 0 else
                            "the mentor of " * (depth - 1) + f"the mentor of {ghost}")
        elif kind == "mentor_link":
            drop_mentor = chain[depth - 1] if depth > 0 else last  # a person on the path loses the link
        else:
            drop_company = d["employer"]
        ans = None
    kb = ws / "kb"
    kb.mkdir(parents=True)
    for p, dd in info.items():
        lines = f"# {p}\n\nBorn: {dd['born']}\nEmployer: {dd['employer']}\n"
        if p != drop_mentor:
            lines += f"Mentor: {dd['mentor']}\n"
        (kb / ("person_" + _slug(p) + ".md")).write_text(lines, encoding="utf-8", newline="\n")
    for c, dd in comp.items():
        if c == drop_company:
            continue
        (kb / ("company_" + _slug(c) + ".md")).write_text(
            f"# {c}\n\nHeadquarters: {dd['city']}\nFounded: {dd['founded']}\n", encoding="utf-8", newline="\n")
    return {"instruction": "The directory kb/ contains a small knowledge base of markdown files. "
                           + q + SUFFIX_SEARCH, "expected": ans, "check": "exact_str"}


# ---------------------------------------------------------------- coding v2

def _load_lib():
    txt = (Path(__file__).parent / "_lib_src.txt").read_text(encoding="utf-8")
    lib = {}
    for block in txt.split("\n## ")[0:]:
        block = block.lstrip("#").lstrip()
        if not block.strip():
            continue
        head, _, rest = block.partition("\n")
        name, _, deps = head.partition("|")
        src, _, cases = rest.partition("\n--\n")
        lib[name.strip()] = {"src": src.rstrip("\n") + "\n", "deps": [d.strip() for d in deps.split(",") if d.strip()],
                             "cases": [c for c in cases.split("\n") if c.strip()]}
    return lib


LIB2 = _load_lib()
_MUT_V2 = _MUT_RULES[:-1] + [
    (r" <= ", " < "), (r" >= ", " > "), (r" and ", " or "), (r" or ", " and "),
    (r" != ", " == "), (r"\bmax\(", "min("), (r"\bmin\(", "max("), (r" // ", " / "),
    (r"\b(\d+)\b", None),
]
_FULL_NS: dict = {}
_EXPECTED2: dict = {}


def _full_ns():
    if not _FULL_NS:
        exec(compile("\n\n".join(v["src"] for v in LIB2.values()), "<lib2>", "exec"), _FULL_NS)
    return _FULL_NS


def _case_expr(name, case):
    return case[1:] if case.startswith("=") else f"m.{name}({case})"


def _eval_expr(ns, name, case):
    from types import SimpleNamespace
    return eval(_case_expr(name, case), {"m": SimpleNamespace(**{k: v for k, v in ns.items()
                                                                  if not k.startswith("__")})})


def _expected2(name):
    if name not in _EXPECTED2:
        ns = _full_ns()
        _EXPECTED2[name] = [_eval_expr(ns, name, c) for c in LIB2[name]["cases"]]
    return _EXPECTED2[name]


def _mutants2(src):
    out = []
    for pat, rep in _MUT_V2:
        for m in re.finditer(pat, src):
            reps = [str(int(m.group(1)) + 1), str(int(m.group(1)) - 1)] if rep is None else [m.expand(rep)]
            for r in reps:
                new = src[:m.start()] + r + src[m.end():]
                if new != src and new not in out:
                    try:
                        compile(new, "<m>", "exec")
                    except SyntaxError:
                        continue
                    out.append(new)
    return out


def _killing_mutant2(rng, name):
    good = _expected2(name)
    cands = _mutants2(LIB2[name]["src"])
    rng.shuffle(cands)
    fallback = None
    for c in cands[:40]:
        ns = dict(_full_ns())
        try:
            exec(compile(c, "<mut>", "exec"), ns)
        except Exception:
            continue
        killed = []
        for i, case in enumerate(LIB2[name]["cases"]):
            try:
                same = _eval_expr(ns, name, case) == good[i]
            except Exception:
                same = False
            if not same:
                killed.append(i)
        if len(killed) >= 2:  # visible case + at least one hidden case must catch the bug
            return c, killed
    return None


def gen_coding_v2(rng, diff, novel, infeasible, ws):
    hard = diff == "hard"
    names = rng.sample(sorted(LIB2), rng.randint(10, 22) if hard else rng.randint(6, 12))
    n_bugs = rng.randint(1, 3) if hard else rng.randint(1, 2)
    buggy = rng.sample(names, len(names))  # try functions in random order until n_bugs are placed
    bug_set = set()
    sources, visible_idx, nb = {}, {}, 0
    for n in buggy:
        if nb >= n_bugs:
            break
        m = _killing_mutant2(rng, n)
        if m:
            bug_set.add(n)
            sources[n] = m[0]
            visible_idx[n] = rng.choice(m[1])
            nb += 1
    for n in names:
        if n not in bug_set:
            sources[n] = LIB2[n]["src"]
            visible_idx[n] = rng.randrange(len(LIB2[n]["cases"]))
    helpers = []
    for n in names:
        for dep in LIB2[n]["deps"]:
            if dep not in names and dep not in helpers:
                helpers.append(dep)
    order = list(names) + helpers
    contra = rng.choice(names) if infeasible else None
    missing_file = infeasible and rng.random() < 0.3
    if missing_file:
        contra = None
    if not missing_file:
        (ws / "mylib.py").write_text(
            "\n\n".join((sources.get(n) or LIB2[n]["src"]).rstrip("\n") + "\n" for n in order),
            encoding="utf-8", newline="\n")
    head = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    vis, hid = list(head), list(head)
    for n in names:
        cases = LIB2[n]["cases"]
        good = _expected2(n)
        vi = visible_idx[n]
        vis += [f"    def test_{n}(self):",
                f"        self.assertEqual({_case_expr(n, cases[vi])}, {good[vi]!r})"]
        if n == contra:
            vis.append(f"        self.assertEqual({_case_expr(n, cases[vi])}, {_contradict(good[vi])!r})")
        vis.append("")
        hid.append(f"    def test_{n}(self):")
        hid += [f"        self.assertEqual({_case_expr(n, c)}, {good[i]!r})"
                for i, c in enumerate(cases) if i != vi]
        hid.append("")
    tail = ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    (ws / "test_mylib.py").write_text("\n".join(vis + tail), encoding="utf-8", newline="\n")
    return {"instruction": CODING_INSTRUCTION, "expected": None if infeasible else "tests_pass",
            "check": "unittest", "_hidden": None if infeasible else "\n".join(hid + tail)}


# ---------------------------------------------------------------- driver

def _gen_one(typ, diff, rng, nov, inf, ws):
    if typ == "math":
        return gen_math(rng, nov, inf) if diff == "easy" else gen_math_v2(rng, diff, nov, inf)
    if typ == "search":
        return gen_search(rng, nov, inf, ws) if diff == "easy" else gen_search_v2(rng, diff, nov, inf, ws)
    if typ == "data_analysis":
        return gen_data(rng, nov, inf, ws) if diff == "easy" else gen_data_v2(rng, diff, nov, inf, ws)
    return gen_coding(rng, nov, inf, ws) if diff == "easy" else gen_coding_v2(rng, diff, nov, inf, ws)


def task_difficulty(seed: int, i: int, difficulty: str) -> str:
    if difficulty != "mixed":
        return difficulty
    return random.Random(f"{seed}/{i}/diff").choices(DIFFS, weights=[2, 3, 3])[0]


def generate(out: Path, n: int, seed: int, infeasible: float = 0.2, novel: float = 0.2,
             difficulty: str = "easy") -> None:
    out = Path(out)
    for sub in ("tasks", "hidden"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)
    rng = random.Random(seed)
    inf_set = set(rng.sample(range(n), round(n * infeasible)))
    nov_set = set(rng.sample(range(n), round(n * novel)))
    diffs = {}
    for i in range(n):
        typ = TYPES[i % 4]
        tid = f"{typ}_{i:04d}"
        tdir = out / "tasks" / tid
        ws = tdir / "workspace"
        ws.mkdir(parents=True)
        inf, nov = i in inf_set, i in nov_set
        d = task_difficulty(seed, i, difficulty)
        diffs[tid] = d
        if difficulty == "easy":  # byte-for-byte the original stream
            spec = _gen_one(typ, "easy", rng, nov, inf, ws)
            reward = rng.randint(1, 5)
        else:
            trng = random.Random(f"{seed}/{i}")
            spec = _gen_one(typ, d, trng, nov, inf, ws)
            reward = trng.randint(1, 5)
        hidden = spec.pop("_hidden", None)
        task = {"task_id": tid, "task_type": typ, "instruction": spec["instruction"],
                "reward": reward, "expected": spec["expected"], "check": spec["check"]}
        (tdir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")
        if hidden:
            (out / "hidden" / tid).mkdir()
            (out / "hidden" / tid / "test_hidden.py").write_text(hidden, encoding="utf-8", newline="\n")
    if difficulty != "easy":
        (out / "difficulty.json").write_text(json.dumps(diffs, indent=1) + "\n", encoding="utf-8", newline="\n")
    elif (out / "difficulty.json").exists():
        (out / "difficulty.json").unlink()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infeasible", type=float, default=0.2)
    ap.add_argument("--novel", type=float, default=0.2)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="easy")
    a = ap.parse_args(argv)
    generate(a.out, a.n, a.seed, a.infeasible, a.novel, a.difficulty)


if __name__ == "__main__":
    main()

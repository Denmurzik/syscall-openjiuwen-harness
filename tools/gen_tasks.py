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


# ---------------------------------------------------------------- driver

def generate(out: Path, n: int, seed: int, infeasible: float = 0.2, novel: float = 0.2) -> None:
    out = Path(out)
    for sub in ("tasks", "hidden"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)
    rng = random.Random(seed)
    inf_set = set(rng.sample(range(n), round(n * infeasible)))
    nov_set = set(rng.sample(range(n), round(n * novel)))
    for i in range(n):
        typ = TYPES[i % 4]
        tid = f"{typ}_{i:04d}"
        tdir = out / "tasks" / tid
        ws = tdir / "workspace"
        ws.mkdir(parents=True)
        inf, nov = i in inf_set, i in nov_set
        if typ == "math":
            spec = gen_math(rng, nov, inf)
        elif typ == "search":
            spec = gen_search(rng, nov, inf, ws)
        elif typ == "data_analysis":
            spec = gen_data(rng, nov, inf, ws)
        else:
            spec = gen_coding(rng, nov, inf, ws)
        hidden = spec.pop("_hidden", None)
        task = {"task_id": tid, "task_type": typ, "instruction": spec["instruction"],
                "reward": rng.randint(1, 5), "expected": spec["expected"], "check": spec["check"]}
        (tdir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")
        if hidden:
            (out / "hidden" / tid).mkdir()
            (out / "hidden" / tid / "test_hidden.py").write_text(hidden, encoding="utf-8", newline="\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infeasible", type=float, default=0.2)
    ap.add_argument("--novel", type=float, default=0.2)
    a = ap.parse_args(argv)
    generate(a.out, a.n, a.seed, a.infeasible, a.novel)


if __name__ == "__main__":
    main()

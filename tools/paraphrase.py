"""Deterministic paraphrases of generated tasks, built to push phase 1 into `unparsed`.

    python -m tools.paraphrase --in runs/x/set --out runs/x/set_para --rate 1.0 --seed 7

Copies a generated task set (tasks/ + hidden/ + the rest) and rewrites the instruction of math, data and
search tasks with template paraphrases (no model). A template is accepted only if, at tool time, the
installed phase-1 solver returns `unparsed` for the rewritten task; otherwise the next template is tried,
and if none works the original instruction is kept and counted. Numbers, names, quoted strings, `expected`
and `check` are never touched; coding tasks are copied as they are. Summary: <out>/paraphrase.json.
"""
import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from harness.solvers import data_solver, math_solver, search_solver
from harness.tasks import Task
from tools.gen_tasks import _M_PHRASES, _S_ALT, _S_ATTRS, SUFFIX_DATA, SUFFIX_MATH, SUFFIX_SEARCH

PARAPHRASED_TYPES = ("math", "data_analysis", "search")
_SOLVERS = {"math": math_solver.solve, "data_analysis": data_solver.solve, "search": search_solver.solve}
_SUFFIX = {"math": SUFFIX_MATH, "data_analysis": SUFFIX_DATA, "search": SUFFIX_SEARCH}
N_TEMPLATES = 8


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ------------------------------------------------------------------ math

_M_START_RX = re.compile(r"A warehouse starts with (?P<s>\d+) boxes\.")
_M_START_ALTS = [
    "The warehouse's opening stock is {s} boxes.",
    "At the outset a warehouse holds {s} boxes.",
    "Initially there are {s} boxes in a warehouse.",
    "To begin with, a warehouse contains {s} boxes.",
]
_M_OP_ALTS = {
    "ship": ["Over {days} days, it sends out {per} boxes each day.",
             "On each of {days} consecutive days, {per} boxes are dispatched.",
             "A daily dispatch of {per} boxes continues for {days} days.",
             "{per} boxes go out every day, and this lasts {days} days."],
    "receive": ["{k} deliveries come in, each with {b} boxes.",
                "It gets {k} consignments, {b} boxes in each.",
                "Each of {k} incoming shipments brings {b} boxes.",
                "{k} lorries unload {b} boxes apiece."],
    "split": ["Next, the stock is shared out equally between {k} warehouses; this warehouse keeps just one "
              "share, rounded down.",
              "After that the stock is distributed in equal parts across {k} warehouses, and this one holds "
              "a single part (rounded down).",
              "The stock is then cut into {k} equal parts, one per warehouse, and this warehouse retains "
              "one part, rounded down.",
              "Afterwards {k} warehouses divide the stock equally and this warehouse keeps one portion "
              "(rounded down)."],
    "double": ["Next, the stock is doubled when a sister warehouse is merged in.",
               "Absorbing a sister warehouse makes the stock twice as large.",
               "The stock becomes twice its size after a sister warehouse joins.",
               "Joining forces with a sister warehouse doubles what is in stock."],
    "triple": ["Next, the stock is multiplied by three through a big merger.",
               "A large merger makes the stock three times as large.",
               "The stock becomes three times its size after a big merger.",
               "A big merger leaves three times as many boxes in stock."],
    "theft": ["{n} boxes are stolen.",
              "Theft removes {n} boxes from the stock.",
              "{n} boxes disappear in a burglary.",
              "Burglars make off with {n} boxes."],
    "ruined": ["{n} boxes are spoiled by a leak and discarded.",
               "A leak damages {n} boxes and they are thrown out.",
               "{n} boxes get soaked by a leak and are binned.",
               "Because of a leak, {n} boxes are discarded."],
    "moved": ["Next, a fourth of the stock (rounded down) goes to another site.",
              "One quarter of the stock, rounded down, is sent to a different site.",
              "A quarter of the boxes (rounded down) are relocated elsewhere.",
              "Then a different site receives one quarter of the stock, rounded down."],
    "sell_half": ["Next, half of the boxes are sold, where the number sold is rounded down.",
                  "It sells off a half of the stock; the quantity sold is rounded down.",
                  "Half the stock goes to customers, the number of boxes sold being rounded down.",
                  "Customers buy half of the boxes, with the sold count rounded down."],
    "add_pct": ["The stock rises by {p} percent of its current level (rounded down).",
                "{p} percent of the current stock, rounded down, is added through a seasonal restock.",
                "A seasonal restock increases the stock by {p} percent, rounded down.",
                "Stock grows by {p} percent (the increase is rounded down) thanks to a seasonal restock."],
    "lose_pct": ["{p} percent of the current stock (rounded down) goes bad.",
                 "The stock shrinks by {p} percent, rounded down, due to spoilage.",
                 "Spoilage takes away {p} percent of the stock, with the amount rounded down.",
                 "Rounded down, {p} percent of the stock is lost to spoilage."],
}
_M_QUESTION_ALTS = [
    "Finally, how many boxes are in stock?",
    "After all of these events, what is the number of boxes in the warehouse?",
    "Give the resulting number of boxes.",
    "Determine the closing stock, in boxes.",
    "What quantity of boxes is on hand once everything has happened?",
    "Tell me how many boxes the warehouse holds after the last event.",
    "How many boxes are in the warehouse when all of this is over?",
    "Find the number of boxes that are left after the final step.",
]
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _phrase_rx(phrase: str) -> re.Pattern:
    parts = re.split(r"\{(\w+)\}", phrase)
    rx = "".join(re.escape(p) if i % 2 == 0 else rf"(?P<{p}>\d+)" for i, p in enumerate(parts))
    return re.compile(rx)


_M_OP_RXS = [(kind, _phrase_rx(p)) for kind, (canon, paras) in _M_PHRASES.items() for p in [canon, *paras]]


def _math_prepare(body: str):
    sentences = _SENT_SPLIT.split(body.strip())
    if len(sentences) < 3 or not sentences[-1].endswith("?"):
        return None
    ops = []
    for i, s in enumerate(sentences[:-1]):
        if i == 0:
            m = _M_START_RX.fullmatch(s)
            if not m:
                return None
            ops.append(("start", {"s": m.group("s")}))
            continue
        for kind, rx in _M_OP_RXS:
            m = rx.fullmatch(s)
            if m:
                ops.append((kind, m.groupdict()))
                break
        else:
            return None
    return ops


def _math_candidates(body: str):
    ops = _math_prepare(body)
    if ops is None:
        return None
    def make(i: int) -> str:
        style = i % 4
        out = []
        for kind, vals in ops:
            alts = _M_START_ALTS if kind == "start" else _M_OP_ALTS[kind]
            out.append(alts[style].format(**vals))
        out.append(_M_QUESTION_ALTS[i])
        return " ".join(out)
    return [(f"math:q{i}", (lambda i=i: make(i))) for i in range(N_TEMPLATES)]


# ------------------------------------------------------------------ data

_D_PROD = re.compile(r"\s+(?:(?:of|for) )?product '(\w+)'")
_D_PROD_V1 = re.compile(r"'(\w+)' (?=units)")
_D_REG = re.compile(r"\s+(?:in|for) region '(\w+)'")
_D_MONTHS = [re.compile(r"\s+in months (\d+) through (\d+) inclusive"),
             re.compile(r"\s+between month (\d+) and month (\d+) \(both included\)")]
_D_MONTH = re.compile(r"\s+in month (\d+)")
_D_ROUND = re.compile(r",?\s+rounded to (\d+) decimal places?")
_D_METRICS = [
    ("units", re.compile(r"(?:how many units were sold|what (?:is|was) the total number of units sold)\?", re.I)),
    ("revenue", re.compile(r"what is the total revenue \(units multiplied by price\)\?", re.I)),
    ("orders", re.compile(r"how many orders were placed\?", re.I)),
    ("avg_price", re.compile(r"what is the average price\?", re.I)),
    ("avg_units", re.compile(r"what is the average number of units per order\?", re.I)),
    ("max", re.compile(r"what is the largest number of units in a single order\?", re.I)),
    ("min", re.compile(r"what is the smallest number of units in a single order\?", re.I)),
    ("absent", re.compile(r"what is the total (\w+) given\?", re.I)),
]
# metric -> text, per metric style (lower-case first letter, except "I need ...")
_D_STYLES = [
    ({"units": "report the sum of the units column", "revenue": "report the sum of units times price",
      "orders": "report the number of order rows", "avg_price": "report the mean price",
      "avg_units": "report the mean number of units per order",
      "max": "report the largest units value found in any one order",
      "min": "report the smallest units value found in any one order",
      "absent": "report the overall {c} granted"}, "."),
    ({"units": "what do the units add up to", "revenue": "what does units times price add up to",
      "orders": "how many order rows are there", "avg_price": "what does the price average out to",
      "avg_units": "what do the units per order average out to",
      "max": "which is the greatest units value in any one order",
      "min": "which is the least units value in any one order",
      "absent": "what is the overall {c} granted"}, "?"),
    ({"units": "I need the grand total of units sold", "revenue": "I need the grand total of units multiplied by price",
      "orders": "I need the count of order rows", "avg_price": "I need the typical price (arithmetic mean)",
      "avg_units": "I need the typical units per order (arithmetic mean)",
      "max": "I need the top units value of a single order",
      "min": "I need the lowest units value of a single order",
      "absent": "I need the overall {c} granted"}, "."),
    ({"units": "please work out the aggregate of the units column",
      "revenue": "please work out the aggregate of units times price",
      "orders": "please work out how many order rows exist", "avg_price": "please work out the mean price",
      "avg_units": "please work out the mean units per order",
      "max": "please work out the biggest units value of one order",
      "min": "please work out the smallest units value of one order",
      "absent": "please work out the overall {c} granted"}, "."),
]


def _data_parse(q: str):
    f: dict = {}
    for key, rx in (("product", _D_PROD), ("product", _D_PROD_V1), ("region", _D_REG)):
        m = rx.search(q)
        if m:
            if key in f:
                return None
            f[key] = m.group(1)
            q = q[:m.start()] + q[m.end():]
    for rx in _D_MONTHS:
        m = rx.search(q)
        if m:
            f["months"] = (m.group(1), m.group(2))
            q = q[:m.start()] + q[m.end():]
    m = _D_MONTH.search(q)
    if m and "months" not in f:
        f["month"] = m.group(1)
        q = q[:m.start()] + q[m.end():]
    m = _D_ROUND.search(q)
    if m:
        f["round"] = m.group(1)
        q = q[:m.start()] + q[m.end():]
    q = " ".join(q.split())
    for name, rx in _D_METRICS:
        m = rx.fullmatch(q)
        if m:
            f["metric"] = name
            f["col"] = m.group(1) if name == "absent" else None
            return f
    return None


def _data_render(f: dict, i: int) -> str:
    texts, end = _D_STYLES[i % 4]
    leading = i >= 4
    metric = texts[f["metric"]].format(c=f["col"])
    if f.get("round"):
        metric += f", rounded to {f['round']} decimal places"
    months = f.get("months")
    if leading:
        conds = []
        if "product" in f:
            conds.append(f"the product is '{f['product']}'")
        if "region" in f:
            conds.append(f"the region is '{f['region']}'")
        if months:
            conds.append(f"the month lies between {months[0]} and {months[1]} inclusive")
        if "month" in f:
            conds.append(f"the month is {f['month']}")
        prefix = f"Among the rows where {' and '.join(conds)}, " if conds else ""
        return (prefix + metric if prefix else _cap(metric)) + end
    where = []
    if "product" in f:
        where.append(f"restricted to product '{f['product']}'")
    if "region" in f:
        where.append(f"with the region being '{f['region']}'")
    if months:
        where.append(f"limited to months {months[0]} through {months[1]}, both ends included")
    if "month" in f:
        where.append(f"during month {f['month']}")
    # the rounding clause stays last
    rnd = ""
    if f.get("round"):
        metric = metric[:metric.rindex(", rounded to")]
        rnd = f", rounded to {f['round']} decimal places"
    return _cap(metric) + "".join(", " + w for w in where) + rnd + end


def _data_candidates(body: str):
    m = re.fullmatch(r"(The file \S+ has columns [\w,]+ \(\d+ rows\)\. )(.+)", body, re.S)
    if not m:
        return None
    f = _data_parse(m.group(2))
    if f is None:
        return None
    return [(f"data:s{i % 4}{'L' if i >= 4 else 'T'}", (lambda i=i: m.group(1) + _data_render(f, i)))
            for i in range(N_TEMPLATES)]


# ------------------------------------------------------------------ search

_S_INTRO = "The directory kb/ contains a small knowledge base of markdown files. "
_S_PARAS = {
    "born": ["Which calendar year did {P} come into the world?",
             "Please determine the birth year recorded for {P}.",
             "Which year does the knowledge base list as the year {P} was born in?",
             "State the year in which {P} came into the world.",
             "Dig up the birth year of {P}.",
             "In what year did {P} enter the world?",
             "Find the year of birth recorded for {P}.",
             "{P} entered the world in which year?"],
    "mentor": ["Name the person who acts as a mentor to {P}.",
               "Find the person mentoring {P}.",
               "Which person takes on the role of mentor for {P}?",
               "Look up the individual assigned as the mentor of {P}.",
               "Identify the mentor recorded for {P}.",
               "Who, according to the knowledge base, mentors {P}?",
               "Tell me which person is mentoring {P}.",
               "Give the name of the mentor of {P}."],
    "employer": ["Which organisation has {P} on its payroll?",
                 "Identify the company {P} is employed by.",
                 "Name the firm that {P} is employed at.",
                 "For which firm does {P} do their job?",
                 "Find the company recorded as the employer for {P}.",
                 "Which company pays the salary of {P}?",
                 "Look up the workplace company of {P}.",
                 "State which firm employs {P}."],
    "hq": ["Which city hosts the head office of the company employing {P}?",
           "Name the city where the employer of {P} is headquartered.",
           "Find the city that the employer of {P} has as its base of operations.",
           "Tell me the home city of the head office of the employer of {P}.",
           "In what city is the main office of the employer of {P} located?",
           "State the city of the headquarters belonging to the employer of {P}.",
           "Look up the city housing the headquarters of the employer of {P}.",
           "The employer of {P} keeps its headquarters in which city?"],
    "founded": ["In what year did the employer of {P} come into being?",
                "Find the founding year of the employer of {P}.",
                "Which year saw the foundation of the employer of {P}?",
                "State the year the employer of {P} was established.",
                "Look up the year in which the employer of {P} began operating.",
                "The employer of {P} was created in which year?",
                "Tell me the year of foundation of the employer of {P}.",
                "Identify the year that the employer of {P} was set up."],
}


def _tmpl_rx(tmpl: str) -> re.Pattern:
    a, b = tmpl.split("{P}")
    return re.compile(re.escape(a) + r"(?P<P>.+)" + re.escape(b), re.S)


_S_RXS = [(attr, _tmpl_rx(t), lambda p: p)
          for attr, (_, t) in _S_ATTRS.items()] + \
         [(attr, _tmpl_rx(t), lambda p: p) for attr, ts in _S_ALT.items() for t in ts] + [
    ("hq", re.compile(r"In which city is the headquarters of (?P<P>.+)'s employer\?", re.S), lambda p: p),
    ("founded", re.compile(r"In which year was (?P<P>.+)'s employer founded\?", re.S), lambda p: p),
    ("hq", re.compile(r"In which city does the mentor of (?P<P>.+) work\?", re.S),
     lambda p: f"the mentor of {p}"),
    ("born", re.compile(r"What is the birth year of (?P<P>.+)'s mentor\?", re.S),
     lambda p: f"the mentor of {p}"),
]


def _search_candidates(body: str):
    if not body.startswith(_S_INTRO):
        return None
    q = body[len(_S_INTRO):]
    for attr, rx, build in _S_RXS:
        m = rx.fullmatch(q)
        if m:
            p = build(m.group("P"))
            return [(f"search:{attr}:{i}", (lambda i=i: _S_INTRO + _cap(_S_PARAS[attr][i].format(P=p))
                                            if not _S_PARAS[attr][i].startswith("{P}")
                                            else _S_INTRO + _S_PARAS[attr][i].format(P=_cap(p))))
                    for i in range(N_TEMPLATES)]
    return None


_CANDIDATES = {"math": _math_candidates, "data_analysis": _data_candidates, "search": _search_candidates}


# ------------------------------------------------------------------ driver

def paraphrase_task(task_type: str, instruction: str, make_task, seed_key: str):
    """-> (new_instruction, template_id) or (None, reason). `make_task(instruction)` builds the Task that
    the phase-1 check runs on."""
    suffix = _SUFFIX[task_type]
    if not instruction.endswith(suffix):
        return None, "unrecognized"
    cands = _CANDIDATES[task_type](instruction[:-len(suffix)])
    if cands is None:
        return None, "unrecognized"
    start = random.Random(seed_key).randrange(len(cands))
    for j in range(len(cands)):
        tid, build = cands[(start + j) % len(cands)]
        new = build() + suffix
        if new != instruction and _SOLVERS[task_type](make_task(new)).status == "unparsed":
            return new, tid
    return None, "phase1_parses"


def paraphrase(src: Path, dst: Path, rate: float = 1.0, seed: int = 0) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tasks_root = dst / "tasks" if (dst / "tasks").is_dir() else dst
    by_type: dict[str, Counter] = {t: Counter() for t in PARAPHRASED_TYPES}
    templates: Counter = Counter()
    changed: dict[str, str] = {}
    for tdir in sorted(tasks_root.iterdir()):
        tj = tdir / "task.json"
        if not tj.is_file():
            continue
        data = json.loads(tj.read_text(encoding="utf-8"))
        typ = data["task_type"]
        if typ not in by_type:
            continue
        c = by_type[typ]
        c["tasks"] += 1
        if random.Random(f"{seed}/{data['task_id']}/rate").random() >= rate:
            c["skipped_rate"] += 1
            continue

        def make_task(instr, data=data, tdir=tdir):
            return Task(task_id=str(data["task_id"]), task_type=typ, instruction=instr,
                        reward=float(data["reward"]), check=data.get("check"), dir=tdir,
                        workspace=tdir / "workspace")

        new, info = paraphrase_task(typ, data["instruction"], make_task, f"{seed}/{data['task_id']}")
        if new is None:
            c["kept_" + info] += 1
            continue
        data["instruction"] = new
        tj.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        c["paraphrased"] += 1
        templates[info] += 1
        changed[data["task_id"]] = info
    summary = {"seed": seed, "rate": rate, "source": str(src),
               "by_type": {t: dict(c) for t, c in by_type.items()},
               "templates": dict(sorted(templates.items())), "tasks": changed}
    (dst / "paraphrase.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = paraphrase(a.src, a.out, a.rate, a.seed)
    for t, c in s["by_type"].items():
        print(t, c)


if __name__ == "__main__":
    main()

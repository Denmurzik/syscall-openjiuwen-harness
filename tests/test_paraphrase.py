import dataclasses
import json
import re
from pathlib import Path

import pytest

from harness.tasks import load_tasks
from tools import gen_tasks, paraphrase
from tools.paraphrase import PARAPHRASED_TYPES, paraphrase_task


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    gen_tasks.generate(d, 40, 81, 0.2, 0.2, "mixed")
    return d


@pytest.fixture(scope="module")
def para(src, tmp_path_factory):
    out = tmp_path_factory.mktemp("para") / "out"
    summary = paraphrase.paraphrase(src, out, 1.0, 5)
    return out, summary


def _load(root):
    return {t.task_id: t for t in load_tasks(root / "tasks")}


def _raw(root):
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (root / "tasks").glob("*/task.json")}


def _digits(s):
    return sorted(re.findall(r"\d+", s))


def _quoted(s):
    return sorted(re.findall(r"'[^']*'", s))


def test_template_counts():
    for attr, ts in paraphrase._S_PARAS.items():
        assert len(ts) >= 8, attr
    assert len(paraphrase._M_QUESTION_ALTS) >= 8
    assert paraphrase.N_TEMPLATES >= 8
    assert len(paraphrase._D_STYLES) * 2 >= 8


def test_expected_check_and_workspace_unchanged(src, para):
    out, _ = para
    a, b = _raw(src), _raw(out)
    assert a.keys() == b.keys()
    for tid in a:
        for k in a[tid]:
            if k != "instruction":
                assert a[tid][k] == b[tid][k], (tid, k)
        wa = sorted(p.relative_to(src) for p in (src / "tasks" / tid).rglob("*") if p.is_file())
        wb = sorted(p.relative_to(out) for p in (out / "tasks" / tid).rglob("*") if p.is_file())
        assert [str(x) for x in wa] == [str(x) for x in wb]
    assert (out / "hidden").is_dir() and sorted(p.name for p in (src / "hidden").iterdir()) == \
        sorted(p.name for p in (out / "hidden").iterdir())


def test_coding_untouched(src, para):
    out, _ = para
    a, b = _raw(src), _raw(out)
    for tid in a:
        if a[tid]["task_type"] == "coding":
            assert a[tid] == b[tid]


def test_numbers_names_quotes_preserved(src, para):
    out, summary = para
    a, b = _raw(src), _raw(out)
    assert summary["tasks"]
    for tid in summary["tasks"]:
        ia, ib = a[tid]["instruction"], b[tid]["instruction"]
        assert ia != ib
        assert _digits(ia) == _digits(ib), tid
        assert _quoted(ia) == _quoted(ib), tid
        if a[tid]["task_type"] == "search":
            names = set(re.findall(r"[A-Z][\w.-]*(?: [A-Z][\w.-]*)+", ia.split("files. ", 1)[1]))
            for n in names:
                assert n in ib, (tid, n)


def test_paraphrased_tasks_are_unparsed_in_phase1(para):
    out, summary = para
    tasks = _load(out)
    for tid in summary["tasks"]:
        t = tasks[tid]
        res = paraphrase._SOLVERS[t.task_type](t)
        assert res.status == "unparsed", (tid, t.instruction)


def test_summary_counts(para):
    out, summary = para
    assert json.loads((out / "paraphrase.json").read_text(encoding="utf-8")) == summary
    assert set(summary["by_type"]) == set(PARAPHRASED_TYPES)
    total = 0
    for t, c in summary["by_type"].items():
        assert c["tasks"] == c.get("paraphrased", 0) + sum(v for k, v in c.items() if k.startswith("kept_"))
        total += c.get("paraphrased", 0)
    assert total == len(summary["tasks"]) == sum(summary["templates"].values())
    assert total > 0


def test_deterministic(src, tmp_path):
    s1 = paraphrase.paraphrase(src, tmp_path / "a", 1.0, 3)
    s2 = paraphrase.paraphrase(src, tmp_path / "b", 1.0, 3)
    assert s1["tasks"] == s2["tasks"]
    assert _raw(tmp_path / "a") == _raw(tmp_path / "b")


def test_rate_zero_changes_nothing(src, tmp_path):
    s = paraphrase.paraphrase(src, tmp_path / "z", 0.0, 1)
    assert s["tasks"] == {}
    assert _raw(src) == _raw(tmp_path / "z")


def test_rejects_template_when_phase1_parses(monkeypatch, src):
    """Mechanism: a candidate that phase 1 still parses is skipped; none left -> original kept and counted."""
    from harness.result import SolveResult
    t = next(t for t in load_tasks(src / "tasks") if t.task_type == "math")
    seen = []
    mk = lambda i: dataclasses.replace(t, instruction=i)

    def fake(task):
        seen.append(task.instruction)
        return SolveResult("solved", answer="1") if len(seen) < 3 else SolveResult("unparsed")

    monkeypatch.setitem(paraphrase._SOLVERS, "math", fake)
    new, tid = paraphrase_task("math", t.instruction, mk, "k")
    assert new == seen[2] and tid.startswith("math:")
    monkeypatch.setitem(paraphrase._SOLVERS, "math", lambda task: SolveResult("solved", answer="1"))
    assert paraphrase_task("math", t.instruction, mk, "k") == (None, "phase1_parses")


def test_unrecognized_instruction_kept():
    assert paraphrase_task("math", "something odd", lambda i: None, "k") == (None, "unrecognized")


def test_every_template_renders_meaning_preserving_math():
    from tools.gen_tasks import SUFFIX_MATH
    body = ("A warehouse starts with 1000 boxes. It ships 10 boxes per day for 3 days. "
            "Thieves take 5 boxes. How many boxes are left at the end?")
    cands = paraphrase._math_candidates(body)
    assert len(cands) >= 8
    for tid, build in cands:
        assert _digits(build()) == _digits(body), tid


def test_every_template_renders_data_and_search():
    body = ("The file sales.csv has columns order_id,region,product,month,units,price (50 rows). "
            "What is the average price for product 'widget' in region 'north' in months 2 through 5 inclusive, "
            "rounded to 2 decimal places?")
    c = paraphrase._data_candidates(body)
    assert len(c) >= 8
    for tid, build in c:
        out = build()
        assert _digits(out) == _digits(body) and _quoted(out) == _quoted(body), (tid, out)
    sb = paraphrase._S_INTRO + "In which city is the headquarters of the employer of the mentor of Ana B. Rossi?"
    c = paraphrase._search_candidates(sb)
    assert len(c) >= 8
    for tid, build in c:
        assert "the mentor of Ana B. Rossi" in build(), tid


# ---- math: every operation x every style, against a hand-computed table (start 1001)
_M_CASES = {  # kind: (canonical sentence, hand-computed stock after it from 1001, must keep "rounded down")
    "ship": ("It ships 7 boxes per day for 3 days.", 1001 - 21, False),
    "receive": ("It receives 3 deliveries of 40 boxes each.", 1001 + 120, False),
    "split": ("Then the stock is split equally into 4 warehouses and it keeps one share (rounded down).",
              250, True),
    "double": ("Then the stock is doubled by merging with a sister warehouse.", 2002, False),
    "triple": ("Then the stock is tripled by a big merger.", 3003, False),
    "theft": ("It loses 11 boxes to theft.", 990, False),
    "ruined": ("A leak ruins 12 boxes, which are thrown away.", 989, False),
    "moved": ("Then one quarter of the stock (rounded down) is moved to another site.", 1001 - 250, True),
    "sell_half": ("Then it sells half of the boxes, rounding the number sold down.", 1001 - 500, True),
    "add_pct": ("Then the stock grows by 33 percent (rounded down).", 1001 + 330, True),
    "lose_pct": ("Then 33 percent of the stock (rounded down) spoils.", 1001 - 330, True),
}


@pytest.mark.parametrize("kind", sorted(_M_CASES))
def test_math_styles_agree_with_hand_table(kind):
    from harness.tasks import Task
    from tools.gen_tasks import SUFFIX_MATH
    sentence, expected, rounded = _M_CASES[kind]
    body = f"A warehouse starts with 1001 boxes. {sentence} How many boxes does it have at the end?"
    t = Task("m", "math", body + SUFFIX_MATH, 1.0, "exact_int", Path("."), Path("."))
    assert paraphrase.math_solver.solve(t).answer == str(expected)  # the table matches the real semantics
    for i in range(4):
        out = paraphrase._math_candidates(body)[i][1]()
        assert _digits(out) == _digits(body)
        if rounded:
            assert "rounded down" in out.split("1001 boxes", 1)[-1] or "rounded down" in out or \
                "rounding" in out, (kind, i, out)
    # explicit guard: the percent-loss wording must say the AMOUNT LOST is rounded down
    if kind == "lose_pct":
        for alt in paraphrase._M_OP_ALTS["lose_pct"]:
            if "shrinks" in alt or "costs" in alt:
                assert "amount lost is rounded down" in alt


def test_revenue_styles_are_not_sum_units_times_price():
    for texts, _ in paraphrase._D_STYLES:
        r = texts["revenue"]
        assert "over all rows" in r or "per-row" in r, r

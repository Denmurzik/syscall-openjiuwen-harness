"""Generator v2 (difficulty levels): determinism, independent re-verification of `expected`, bench smoke."""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import bench, gen_tasks

DIFFS = ["easy", "medium", "hard", "mixed"]


def _tasks(root):
    out = []
    for d in sorted((Path(root) / "tasks").iterdir()):
        out.append((d, json.loads((d / "task.json").read_text(encoding="utf-8"))))
    return out


def _digest(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    base = tmp_path_factory.mktemp("sets")
    for d in ("medium", "hard"):
        gen_tasks.main(["--out", str(base / d), "--n", "80", "--seed", "11", "--difficulty", d,
                        "--infeasible", "0.25"])
    return base


@pytest.mark.parametrize("diff", DIFFS)
def test_determinism_by_seed(tmp_path, diff):
    for name in ("a", "b"):
        gen_tasks.main(["--out", str(tmp_path / name), "--n", "16", "--seed", "5", "--difficulty", diff])
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    gen_tasks.main(["--out", str(tmp_path / "c"), "--n", "16", "--seed", "6", "--difficulty", diff])
    assert _digest(tmp_path / "a") != _digest(tmp_path / "c")


def test_easy_default_equals_explicit_easy(tmp_path):
    gen_tasks.main(["--out", str(tmp_path / "a"), "--n", "12", "--seed", "2"])
    gen_tasks.main(["--out", str(tmp_path / "b"), "--n", "12", "--seed", "2", "--difficulty", "easy"])
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    assert not (tmp_path / "a" / "difficulty.json").exists()


def test_task_prefix_independent_of_n(tmp_path):
    # v2 tasks depend only on (seed, index, flags), so a bigger set extends a smaller one
    gen_tasks.main(["--out", str(tmp_path / "a"), "--n", "8", "--seed", "4", "--difficulty", "hard",
                    "--infeasible", "0", "--novel", "0"])
    gen_tasks.main(["--out", str(tmp_path / "b"), "--n", "16", "--seed", "4", "--difficulty", "hard",
                    "--infeasible", "0", "--novel", "0"])
    for i in range(8):
        t = ["math", "search", "data_analysis", "coding"][i % 4] + f"_{i:04d}"
        assert (tmp_path / "a/tasks" / t / "task.json").read_bytes() == (tmp_path / "b/tasks" / t / "task.json").read_bytes()


def test_format_matches_organizer_examples(sets):
    for d, meta in _tasks(sets / "hard"):
        assert set(meta) == {"task_id", "task_type", "instruction", "reward", "expected", "check"}
        if meta["task_type"] == "coding":
            assert meta["instruction"] == gen_tasks.CODING_INSTRUCTION
        else:
            assert meta["instruction"].endswith(
                {"math": gen_tasks.SUFFIX_MATH, "search": gen_tasks.SUFFIX_SEARCH,
                 "data_analysis": gen_tasks.SUFFIX_DATA}[meta["task_type"]])


# ---- math: re-simulate the instruction with regexes built from the phrase templates

def _math_regexes():
    out = []
    for kind, (canon, paras) in gen_tasks._M_PHRASES.items():
        for t in [canon] + paras:
            rx = re.escape(t).replace(r"\{", "{").replace(r"\}", "}")
            rx = re.sub(r"\{(\w+)\}", r"(?P<\1>\\d+)", rx)
            out.append((kind, re.compile("^" + rx + "$")))
    return out


def _solve_math(text):
    body = text[:text.index(" Compute it with")]
    m = re.match(r"A warehouse starts with (\d+) boxes\. (.*\.) ([^.]*\?)$", body)
    stock, rest = int(m.group(1)), m.group(2)
    rxs = _math_regexes()
    for sent in re.split(r"(?<=\.) (?=[A-Z0-9])", rest):
        for kind, rx in rxs:
            mm = rx.match(sent.strip())
            if mm:
                v = {k: int(x) for k, x in mm.groupdict().items()}
                break
        else:
            raise AssertionError(f"unparsed sentence: {sent!r}")
        if kind == "ship":
            stock -= v["per"] * v["days"]
        elif kind == "receive":
            stock += v["k"] * v["b"]
        elif kind == "split":
            if v["k"] == 0:
                return None
            stock //= v["k"]
        elif kind == "double":
            stock *= 2
        elif kind == "triple":
            stock *= 3
        elif kind in ("theft", "ruined"):
            stock -= v["n"]
        elif kind == "moved":
            stock -= stock // 4
        elif kind == "sell_half":
            stock -= stock // 2
        elif kind == "add_pct":
            stock += stock * v["p"] // 100
        elif kind == "lose_pct":
            stock -= stock * v["p"] // 100
        assert stock >= 0
    return stock


def test_math_expected_reverified(sets):
    seen = 0
    for diff in ("medium", "hard"):
        for d, meta in _tasks(sets / diff):
            if meta["task_type"] != "math":
                continue
            got = _solve_math(meta["instruction"])
            assert (None if got is None else str(got)) == meta["expected"], meta["task_id"]
            seen += 1
    assert seen == 40


# ---- data: recompute with the csv module from the question text

def _solve_data(ws, text):
    fname = re.search(r"The file (\S+\.csv) has columns", text).group(1)
    q = text.split(" rows). ", 1)[1].split(" Use bash")[0]
    if not (ws / fname).is_file():
        return None
    rows = list(csv.DictReader((ws / fname).open(encoding="utf-8")))
    if "discount" in q:
        return None
    sel = rows
    if m := re.search(r"product '(\w+)'", q):
        if m.group(1) not in {r["product"] for r in rows}:
            return None  # unknown entity: no answer
        sel = [r for r in sel if r["product"] == m.group(1)]
    if m := re.search(r"region '(\w+)'", q):
        if m.group(1) not in {r["region"] for r in rows}:
            return None
        sel = [r for r in sel if r["region"] == m.group(1)]
    if m := re.search(r"months (\d+) through (\d+) inclusive", q):
        sel = [r for r in sel if int(m.group(1)) <= int(r["month"]) <= int(m.group(2))]
    elif m := re.search(r"in month (\d+)", q):
        sel = [r for r in sel if int(r["month"]) == int(m.group(1))]
    u = [int(r["units"]) for r in sel]
    p = [int(r["price"]) for r in sel]
    if q.startswith(("How many units", "What is the total number of units")):
        return str(sum(u))
    if "total revenue" in q:
        return str(sum(a * b for a, b in zip(u, p)))
    if q.startswith("How many orders"):
        return str(len(sel))
    if not sel:
        return None
    from decimal import ROUND_HALF_UP, Decimal
    if "average price" in q:
        return str((Decimal(sum(p)) / len(p)).quantize(Decimal("0.01"), ROUND_HALF_UP))
    if "average number of units" in q:
        return str((Decimal(sum(u)) / len(u)).quantize(Decimal("0.01"), ROUND_HALF_UP))
    if "largest" in q:
        return str(max(u))
    if "smallest" in q:
        return str(min(u))
    raise AssertionError(q)


def test_data_expected_reverified(sets):
    seen = 0
    for diff in ("medium", "hard"):
        for d, meta in _tasks(sets / diff):
            if meta["task_type"] != "data_analysis":
                continue
            ws = d / "workspace"
            exp = _solve_data(ws, meta["instruction"])
            if meta["expected"] is None:
                assert exp is None, meta["task_id"]
            else:
                assert exp == meta["expected"], meta["task_id"]
            rows = int(re.search(r"\((\d+) rows\)", meta["instruction"]).group(1))
            assert 20 <= rows <= 2000
            seen += 1
    assert seen == 40


# ---- search: walk the kb files

def _kb(ws):
    kb = {}
    for f in (ws / "kb").glob("*.md"):
        lines = f.read_text(encoding="utf-8").splitlines()
        rec = {"name": lines[0][2:]}
        for ln in lines[2:]:
            k, _, v = ln.partition(": ")
            rec[k] = v
        kb[(f.name.split("_")[0], rec["name"])] = rec
    return kb


def _solve_search(ws, text):
    q = text.split("files. ", 1)[1].split(" Use the file")[0]
    kb = _kb(ws)
    people = {n: r for (t, n), r in kb.items() if t == "person"}
    comps = {n: r for (t, n), r in kb.items() if t == "company"}
    depth = q.count("the mentor of") - (1 if depth_attr(q, "mentor") else 0)
    # the person is the tail after the last "of " / "was " / " work" etc: find a known or unknown name
    cands = [n for n in people if n in q]
    if not cands:
        return None  # name not in the kb at all
    name = max(cands, key=len)
    cur = people[name]
    for _ in range(depth):
        if "Mentor" not in cur:
            return None
        if cur["Mentor"] not in people:
            return None
        cur = people[cur["Mentor"]]
    if "born" in q or "birth year" in q:
        return cur["Born"]
    if "mentor" in q and depth_attr(q, "mentor"):
        return cur.get("Mentor")
    emp = cur["Employer"]
    if "headquarters" in q:
        return comps[emp]["Headquarters"] if emp in comps else None
    if "founded" in q:
        return comps[emp]["Founded"] if emp in comps else None
    return emp


def depth_attr(q, word):
    # final attribute is "mentor" only for "Who is the mentor of ..." / "Name the mentor of ..."
    return q.startswith(("Who is the mentor of", "Name the mentor of"))


def test_search_expected_reverified(sets):
    seen = 0
    for diff in ("medium", "hard"):
        for d, meta in _tasks(sets / diff):
            if meta["task_type"] != "search":
                continue
            exp = _solve_search(d / "workspace", meta["instruction"])
            assert exp == meta["expected"], (meta["task_id"], meta["instruction"])
            n = len(list((d / "workspace" / "kb").glob("person_*")))
            c = len(list((d / "workspace" / "kb").glob("company_*")))
            assert 20 <= n <= 150 and 9 <= c <= 40
            seen += 1
    assert seen == 40


def test_search_near_duplicates_and_odd_names(sets):
    firsts, lasts, odd = {}, {}, 0
    for d, meta in _tasks(sets / "hard"):
        if meta["task_type"] != "search":
            continue
        for (t, n) in _kb(d / "workspace"):
            pass
        for (t, n), r in _kb(d / "workspace").items():
            if t == "person":
                parts = n.split()
                firsts[parts[0]] = firsts.get(parts[0], 0) + 1
                lasts[parts[-1]] = lasts.get(parts[-1], 0) + 1
                odd += bool(re.search(r"\b[A-Z]\.|-", n))
    assert max(firsts.values()) > 3 and max(lasts.values()) > 3 and odd > 0


# ---- coding: correct library passes hidden tests, bugged one fails them

def _run(workdir, test):
    r = subprocess.run([sys.executable, "-B", test], cwd=workdir, capture_output=True, timeout=120)
    return r.returncode == 0


def _original_library(mylib_text):
    names = re.findall(r"^(?:def|class) (\w+)", mylib_text, re.M)
    srcs = []
    for n in names:
        srcs.append(gen_tasks.LIB2[n]["src"].rstrip("\n") + "\n")
    return "\n\n".join(srcs)


def test_coding_hidden_tests_discriminate(sets, tmp_path):
    seen = 0
    for diff in ("medium", "hard"):
        for d, meta in _tasks(sets / diff):
            if meta["task_type"] != "coding":
                continue
            ws = d / "workspace"
            hid = sets / diff / "hidden" / meta["task_id"] / "test_hidden.py"
            if meta["expected"] is None:
                assert not hid.exists()
                continue
            w = tmp_path / diff / meta["task_id"]
            w.mkdir(parents=True)
            (w / "test_hidden.py").write_text(hid.read_text(encoding="utf-8"), encoding="utf-8")
            text = (ws / "mylib.py").read_text(encoding="utf-8")
            asserts = hid.read_text(encoding="utf-8").count("self.assertEqual")
            tested = len(re.findall(r"def test_", hid.read_text(encoding="utf-8")))
            assert asserts >= 4 * tested
            vis = (ws / "test_mylib.py").read_text(encoding="utf-8")
            assert vis.count("self.assertEqual") == len(re.findall(r"def test_", vis))  # 1 assert per function
            (w / "mylib.py").write_text(_original_library(text), encoding="utf-8")
            assert _run(w, "test_hidden.py"), meta["task_id"]  # original library passes
            (w / "mylib.py").write_text(text, encoding="utf-8")
            assert not _run(w, "test_hidden.py"), meta["task_id"]  # the shipped (bugged) one fails
            (w / "test_mylib.py").write_text(vis, encoding="utf-8")
            assert not _run(w, "test_mylib.py"), meta["task_id"]  # and visibly so
            seen += 1
    assert seen >= 20


def test_coding_infeasible_variants(sets):
    kinds = set()
    for diff in ("medium", "hard"):
        for d, meta in _tasks(sets / diff):
            if meta["task_type"] == "coding" and meta["expected"] is None:
                ws = d / "workspace"
                kinds.add("missing_file" if not (ws / "mylib.py").exists() else "contradiction")
    assert kinds == {"missing_file", "contradiction"}


def test_library_has_at_least_40_functions():
    assert sum(1 for v in gen_tasks.LIB2.values() if v["src"].startswith("def ")) >= 40
    assert sum(1 for v in gen_tasks.LIB2.values() if v["src"].startswith("class ")) >= 3
    assert any(v["deps"] for v in gen_tasks.LIB2.values())
    assert all(len(v["cases"]) >= 5 for v in gen_tasks.LIB2.values())


def test_infeasible_ratio_and_types(sets):
    for diff in ("medium", "hard"):
        metas = [m for _, m in _tasks(sets / diff)]
        assert sum(m["expected"] is None for m in metas) == 20


# ---- bench smoke

def test_bench_smoke(tmp_path):
    r = bench.bench(tmp_path / "b", 40, 1, "mixed", workers=4)
    o = r["summary"]["overall"]
    assert o["total"] == 40
    assert o["correct"] + o["wrong"] + o["unparsed"] + o["infeasible_ok"] + o["infeasible_answered"] == 40
    j = json.loads((tmp_path / "b" / "bench.json").read_text(encoding="utf-8"))
    assert {"summary", "failure_groups", "failures", "meta"} <= set(j)
    assert set(j["summary"]["by_type"]) == {"math", "search", "data_analysis", "coding"}
    assert set(j["summary"]["by_difficulty"]) <= {"easy", "medium", "hard"}
    md = (tmp_path / "b" / "bench.md").read_text(encoding="utf-8")
    assert "## By type" in md and "## Failures" in md and "p95" in md
    # every wrong/unparsed task is listed with id, expected, got and the solver note
    for f in j["failures"]:
        assert {"task_id", "instruction", "expected", "got", "note"} <= set(f)
    assert sum(g["count"] for g in j["failure_groups"]) == len(j["failures"])
    # easy-only bench: the generator's own easy templates are solved by phase 1
    r = bench.bench(tmp_path / "e", 24, 1, "easy", workers=4, novel=0.0)
    assert r["summary"]["overall"]["accuracy_feasible"] >= 0.9


@pytest.mark.parametrize("diff,seed", [("medium", 77), ("hard", 77), ("hard", 5), ("medium", 6), ("mixed", 8)])
def test_every_feasible_coding_task_is_really_buggy(tmp_path, diff, seed):
    gen_tasks.main(["--out", str(tmp_path / "s"), "--n", "120", "--seed", str(seed), "--difficulty", diff,
                    "--infeasible", "0.1"])
    n = 0
    for d, meta in _tasks(tmp_path / "s"):
        if meta["task_type"] != "coding" or meta["expected"] is None:
            continue
        hid = tmp_path / "s" / "hidden" / meta["task_id"] / "test_hidden.py"
        w = tmp_path / "w" / meta["task_id"]
        w.mkdir(parents=True)
        (w / "test_hidden.py").write_text(hid.read_text(encoding="utf-8"), encoding="utf-8")
        (w / "mylib.py").write_text((d / "workspace" / "mylib.py").read_text(encoding="utf-8"), encoding="utf-8")
        assert not _run(w, "test_hidden.py"), meta["task_id"]
        (w / "test_mylib.py").write_text((d / "workspace" / "test_mylib.py").read_text(encoding="utf-8"),
                                         encoding="utf-8")
        assert not _run(w, "test_mylib.py"), meta["task_id"]
        n += 1
    assert n >= 20

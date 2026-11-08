"""Adversarial gate for phase 1 (review of task 29): each probe must end `unparsed` / `infeasible` or with the true
answer, never a confident wrong value. Probes: tests/fixtures/probes_r6/probes.json (`allowed` = true answers)."""
import json
from pathlib import Path

import pytest

from harness.solvers import data_solver, math_solver, search_solver
from harness.tasks import Task

M_SUF = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, "
         "write ONLY the final answer to the file answer.txt in the current directory.")
D_SUF = (" Use bash with python3 to compute it. Work only inside the current directory. When you are done, "
         "write ONLY the final answer to the file answer.txt in the current directory.")
S_SUF = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. "
         "When you are done, write ONLY the final answer to the file answer.txt in the current directory.")

_FIX = Path(__file__).parent / "fixtures" / "probes_r6"
_PROBES = json.loads((_FIX / "probes.json").read_text(encoding="utf-8"))
_SUFFIX = {"data": D_SUF, "math": M_SUF, "search": S_SUF}
_MOD = {"data": (data_solver, "data_analysis"), "math": (math_solver, "math"), "search": (search_solver, "search")}
_WS = {"data": _FIX / "data", "north": _FIX / "north", "search": _FIX / "search"}


@pytest.mark.parametrize("probe", _PROBES, ids=[p["id"] for p in _PROBES])
def test_probe_never_confidently_wrong(probe):
    mod, typ = _MOD[probe["dom"]]
    ws = _WS[probe["ws"]]
    r = mod.solve(Task(probe["id"], typ, probe["text"] + _SUFFIX[probe["dom"]], 1.0, "exact_str", ws, ws))
    if r.status == "solved":
        assert r.answer in probe["allowed"], (probe["note"], r.answer)


def test_employer_of_possessive_mentor_still_answers():
    ws = _WS["search"]
    for q in ("Who is the employer of Alice Khan's mentor?", "Who is Alice Khan's mentor's employer?"):
        r = search_solver.solve(Task("t", "search", q + S_SUF, 1.0, "exact_str", ws, ws))
        assert r.status == "solved" and r.answer == "Juniper Logistics", q


def test_mentor_of_possessive_employer_is_not_answered():
    ws = _WS["search"]
    r = search_solver.solve(Task("t", "search", "Who is the mentor of Alice Khan's employer?" + S_SUF,
                                 1.0, "exact_str", ws, ws))
    assert r.status in ("unparsed", "infeasible")

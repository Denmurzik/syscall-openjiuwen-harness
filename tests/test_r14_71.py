"""Round 14, task 71 (expert 7, idea a): the end-of-run audit of coding functions that stand green in rival forms
across the task set (harness.audit, scheduler._audit_stage, HARNESS_AUDIT_FORMS)."""
import asyncio
import json
import time

import pytest

from harness import audit, scheduler
from harness.budget import TokenLedger
from harness.config import ARBITRATE_MODELS, MODEL_REASON
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.result import SolveResult
from harness.tasks import Task
from harness.trajectory import record_call

_REAL_PHASE2 = scheduler._phase2

GOOD = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"
BAD = GOOD.replace("best = xs[0]", "best = 0")          # green on the visible assert, wrong on [3, 5]
GOOD2 = "def count_pos(xs):\n    return sum(1 for x in xs if x > 0)\n"
BAD2 = GOOD2.replace("x > 0", "x >= 0")                  # green on the visible assert, wrong on [0, 0, 5]
OTHER = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"   # a function of the examples
TEST = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        "    def test_minimum(self):\n        self.assertEqual(m.minimum([4, -1, 7]), -1)\n"
        "    def test_count_pos(self):\n        self.assertEqual(m.count_pos([1, -2, 3]), 2)\n"
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n")
TEST_NO_MIN = TEST.replace("    def test_minimum(self):\n        self.assertEqual(m.minimum([4, -1, 7]), -1)\n", "")


class Fake:
    """Replies per model, in order; every call recorded. `charge` is added to the ledger per call. A digit reply of
    MODEL_ALT names the option in MODEL_FAST's order (the audit shows ALT the options reversed); `raw` turns that off."""

    def __init__(self, replies, charge=0.0, budget=100000, raw=False):
        self.replies = {m: list(replies) for m in ARBITRATE_MODELS} if isinstance(replies, list) else \
            {m: list(v) for m, v in replies.items()}
        self.calls = []
        self.charge = charge
        self.raw = raw
        self.ledger = TokenLedger(budget, 0)

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append((model, messages, max_tokens))
        self.ledger.spent += self.charge
        record_call(model, 10, 1, 2)
        r = self.replies[model].pop(0)
        if isinstance(r, BaseException):
            raise r
        if isinstance(r, ChatResult):
            return r
        regex = (extra.get("extra_body") or {}).get("guided_regex")
        if model == ARBITRATE_MODELS[1] and not self.raw and regex and r.isdigit():
            n = int(regex[3:-1])
            r = str(n + 1 - int(r))
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=1, cost=2)


def _write(d, lib, test):
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(test, encoding="utf-8")


def _tasks(tmp_path, libs, tests=None, libs2=None):
    out = []
    for i, lib in enumerate(libs):
        d = tmp_path / f"coding_{i}"
        # `_TASK`: every workspace differs (identical ones count once)
        _write(d, lib + "\n\n" + ((libs2 or {}).get(i, GOOD2)) + "\n\n" + OTHER + f"\n_TASK = {i}\n",
               (tests or {}).get(i, TEST))
        out.append(Task(f"coding_{i}", "coding", "fix", 2.0, "unittest", d, d / "workspace"))
    return out


def _lib(task):
    return (task.workspace / "mylib.py").read_text(encoding="utf-8")


def _decide(client, rivals, holders, name="minimum", may_reason=None):
    return asyncio.run(audit.decide(client, name, rivals[name], holders, may_reason))


# ---------------------------------------------------------------- census

def test_no_rival_forms_no_call(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, GOOD, GOOD]))
    assert len(holders) == 3 and rivals == {}


def test_rival_forms_are_found_with_votes(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, BAD, GOOD]))
    assert list(rivals) == ["minimum"]   # count_pos is unique in the set; total is an example function
    assert [f["votes"] for f in rivals["minimum"]] == [2, 1]
    shown, rows = audit.options("minimum", rivals["minimum"], holders)
    assert [o["votes"] for o in shown] == [2, 1]
    assert rows and all(c.startswith("minimum(") and len(res) == 2 and res[0] != res[1] for c, res in rows)


def test_identical_workspaces_vote_once_and_still_take_the_winner(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD, BAD])
    (tasks[3].workspace / "mylib.py").write_text(_lib(tasks[1]), encoding="utf-8")   # coding_3 = coding_1
    holders, rivals = audit.census(tasks)
    assert len(holders) == 4 and [f["votes"] for f in rivals["minimum"]] == [2, 1]
    winner, _ = _decide(Fake(["1"]), rivals, holders)
    assert sorted(audit.apply("minimum", winner, holders)) == ["coding_1", "coding_3"]


def test_a_holder_passes_every_test_file(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD])
    (tasks[1].workspace / "test_extra.py").write_text(
        TEST.replace("m.minimum([4, -1, 7]), -1", "m.minimum([3, 5]), 3"), encoding="utf-8")
    holders, rivals = audit.census(tasks)
    assert [h.task.task_id for h in holders] == ["coding_0", "coding_2"] and rivals == {}


def test_a_replacement_must_keep_every_test_file_green(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD])
    (tasks[1].workspace / "test_extra.py").write_text(
        TEST.replace("m.minimum([4, -1, 7]), -1", "m.minimum([4, 1, 7]), 0"), encoding="utf-8")   # only BAD passes
    holders, rivals = audit.census(tasks)
    winner, _ = _decide(Fake(["1"]), rivals, holders)
    assert audit.apply("minimum", winner, holders) == [] and "best = 0" in _lib(tasks[1])


def test_example_functions_are_never_audited(tmp_path):
    other_total = "def total(xs):\n    return sum(xs) if xs else 1\n"   # green on [1, 2, 3], wrong on []
    tasks = _tasks(tmp_path, [GOOD, GOOD, GOOD])
    for t in tasks[:2]:
        (t.workspace / "mylib.py").write_text(_lib(t).replace(OTHER, other_total), encoding="utf-8")
    holders, rivals = audit.census(tasks)
    assert len(holders) == 3 and rivals == {}


def test_red_task_is_not_a_holder(tmp_path):
    red = GOOD.replace("x < best", "x > best")
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, red, GOOD]))
    assert len(holders) == 2 and rivals == {}


def test_different_signatures_are_not_rivals(tmp_path):
    other = "def minimum(xs, default=None):\n    return min(xs) if xs else default\n"
    assert audit.census(_tasks(tmp_path, [GOOD, other, GOOD]))[1] == {}


def test_untested_forms_alone_are_not_rivals(tmp_path):
    """One file tests the function, the other does not: fewer than MIN_HOLDERS green votes, nothing to judge."""
    tasks = _tasks(tmp_path, [GOOD, BAD], tests={1: TEST_NO_MIN})
    assert audit.census(tasks)[1] == {}


def test_census_out_of_time_gives_nothing(tmp_path):
    assert audit.census(_tasks(tmp_path, [GOOD, BAD, GOOD]), budget_s=0.0) == ([], {})


# ---------------------------------------------------------------- decision

def test_both_models_agree_the_loser_is_replaced(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD])
    holders, rivals = audit.census(tasks)
    client = Fake(["1"])
    winner, note = _decide(client, rivals, holders)
    assert winner["text"] == GOOD.strip() and "both models" in note
    assert len(client.calls) == 2 and all(c[2] == audit.MAX_TOKENS for c in client.calls)
    assert sorted(c[0] for c in client.calls) == sorted(ARBITRATE_MODELS)
    assert audit.apply("minimum", winner, holders) == ["coding_1"]
    assert "best = xs[0]" in _lib(tasks[1]) and "def total" in _lib(tasks[1]) and "def count_pos" in _lib(tasks[1])


def test_models_disagree_nothing_changes(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD])
    holders, rivals = audit.census(tasks)
    client = Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: ["2"]})
    winner, note = _decide(client, rivals, holders)
    assert winner is None and "no agreement" in note
    assert "best = 0" in _lib(tasks[1])


@pytest.mark.parametrize("second", [LLMUnavailable("down"), "x", "7"])
def test_a_model_unavailable_or_unusable_nothing_changes(tmp_path, second):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, BAD, GOOD]))
    client = Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: [second]})
    assert _decide(client, rivals, holders)[0] is None


def test_budget_exhausted_propagates(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, BAD, GOOD]))
    client = Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: [BudgetExhausted("spent")]})
    with pytest.raises(BudgetExhausted):
        _decide(client, rivals, holders)


def test_the_minority_wins_only_with_the_reasoning_model_too(tmp_path):
    """Two files hold the bug, one the right text: the set votes for the bug, so a third opinion is asked."""
    tasks = _tasks(tmp_path, [BAD, BAD, GOOD])
    holders, rivals = audit.census(tasks)
    assert "best = 0" in rivals["minimum"][0]["text"]
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: ["Option 2 is right. 2"]})
    winner, note = _decide(client, rivals, holders)
    assert winner["text"] == GOOD.strip() and "all three" in note and len(client.calls) == 3
    assert client.calls[-1][0] == MODEL_REASON and client.calls[-1][2] == audit.REASON_TOKENS
    assert sorted(audit.apply("minimum", winner, holders)) == ["coding_0", "coding_1"]
    assert all("best = xs[0]" in _lib(t) for t in tasks)


def test_three_against_two_still_needs_the_third_vote(tmp_path):
    """Any pick with fewer votes than the top option is a minority pick, not only one with half of them."""
    holders, rivals = audit.census(_tasks(tmp_path, [BAD, BAD, BAD, GOOD, GOOD]))
    assert [f["votes"] for f in rivals["minimum"]] == [3, 2]
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: ["1"]})
    winner, note = _decide(client, rivals, holders)
    assert winner is None and "not confirmed" in note and len(client.calls) == 3


@pytest.mark.parametrize("third", ["1", LLMUnavailable("down"), "no idea",
                                   ChatResult(text="2", prompt_tokens=1, completion_tokens=1, cost=1, truncated=True)])
def test_the_minority_without_the_third_vote_changes_nothing(tmp_path, third):
    tasks = _tasks(tmp_path, [BAD, BAD, GOOD])
    holders, rivals = audit.census(tasks)
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: [third]})
    winner, note = _decide(client, rivals, holders)
    assert winner is None and "not confirmed" in note
    assert "best = 0" in _lib(tasks[0])


def test_the_third_vote_is_not_asked_when_the_gate_refuses(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [BAD, BAD, GOOD]))
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: ["2"]})
    asked = []
    winner, note = _decide(client, rivals, holders, may_reason=lambda est: asked.append(est) or False)
    assert winner is None and "no room" in note and len(client.calls) == 2
    assert asked and asked[0] >= audit.REASON_TOKENS


def test_the_majority_pick_needs_no_third_vote(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, GOOD, GOOD, BAD]))
    client = Fake(["1"])
    assert _decide(client, rivals, holders)[0]["text"] == GOOD.strip()
    assert len(client.calls) == 2


def test_a_tie_needs_no_third_vote(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [BAD, GOOD]))
    good = 2 if "best = 0" in rivals["minimum"][0]["text"] else 1
    client = Fake([str(good)])
    winner, _ = _decide(client, rivals, holders)
    assert winner["text"] == GOOD.strip() and len(client.calls) == 2


def test_forms_with_the_same_behaviour_are_one_option_without_adding_votes(tmp_path):
    """A rewrite of the right text and the right text are one option (shown once, its files left as they are), but
    its votes are those of one text: 3 files hold the bug, 2 + 2 the right behaviour, and the bug is still the
    majority, so the right option needs the third opinion."""
    same = GOOD.replace("if x < best:", "if best > x:")
    tasks = _tasks(tmp_path, [BAD, BAD, BAD, GOOD, GOOD, same, same])
    holders, rivals = audit.census(tasks)
    assert [f["votes"] for f in rivals["minimum"]] == [3, 2, 2]
    shown, rows = audit.options("minimum", rivals["minimum"], holders)
    assert [o["votes"] for o in shown] == [3, 2] and len(shown[1]["dumps"]) == 2
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: ["2"]})
    winner, note = _decide(client, rivals, holders)
    assert len(client.calls) == 3 and "minority" in note
    assert sorted(audit.apply("minimum", winner, holders)) == ["coding_0", "coding_1", "coding_2"]
    assert "if best > x:" in _lib(tasks[5]) and "best = xs[0]" in _lib(tasks[0])


def test_forms_with_the_same_behaviour_only_are_not_asked(tmp_path):
    same = GOOD.replace("if x < best:", "if best > x:")
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, same, GOOD]))
    client = Fake(["2"])
    winner, note = _decide(client, rivals, holders)
    assert winner is None and client.calls == [] and "do not differ" in note


def test_replacement_that_turns_a_visible_test_red_is_not_written(tmp_path):
    strict = TEST.replace("m.minimum([4, -1, 7]), -1", "m.minimum([4, 1, 7]), 0")   # only the `best = 0` text passes
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD], tests={1: strict})
    holders, rivals = audit.census(tasks)
    winner, _ = _decide(Fake(["1"]), rivals, holders)
    assert audit.apply("minimum", winner, holders) == []
    assert "best = 0" in _lib(tasks[1])


def test_apply_stops_at_its_deadline(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, BAD, GOOD])
    holders, rivals = audit.census(tasks)
    winner, _ = _decide(Fake(["1"]), rivals, holders)
    assert audit.apply("minimum", winner, holders, stop=time.monotonic() - 1) == []
    assert "best = 0" in _lib(tasks[1])


def test_an_untested_form_is_judged_by_the_models_not_replaced_for_free(tmp_path):
    tasks = _tasks(tmp_path, [GOOD, GOOD, GOOD, BAD], tests={3: TEST_NO_MIN})
    holders, rivals = audit.census(tasks)
    assert [(f["votes"], f["untested"]) for f in rivals["minimum"]] == [(3, 0), (0, 1)]
    assert _decide(Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: ["2"]}), rivals, holders)[0] is None
    client = Fake(["1"])
    winner, _ = _decide(client, rivals, holders)
    assert winner["text"] == GOOD.strip() and len(client.calls) == 2
    assert audit.apply("minimum", winner, holders) == ["coding_3"]
    assert "best = xs[0]" in _lib(tasks[3])


def test_untested_form_picked_against_the_tested_one_needs_the_third_vote(tmp_path):
    tasks = _tasks(tmp_path, [BAD, BAD, GOOD], tests={2: TEST_NO_MIN})
    holders, rivals = audit.census(tasks)
    client = Fake({ARBITRATE_MODELS[0]: ["2"], ARBITRATE_MODELS[1]: ["2"], MODEL_REASON: ["1"]})
    assert _decide(client, rivals, holders)[0] is None and len(client.calls) == 3


def test_an_untested_form_not_shown_is_never_rewritten(tmp_path):
    third = GOOD.replace("best = xs[0]", "best = 100")
    tasks = _tasks(tmp_path, [GOOD, GOOD, BAD, third], tests={3: TEST_NO_MIN})
    holders, rivals = audit.census(tasks)
    shown, _ = audit.options("minimum", rivals["minimum"], holders)
    assert [o["votes"] for o in shown] == [2, 1]
    client = Fake(["1"])
    winner, _ = _decide(client, rivals, holders)
    assert all("best = 100" not in c[1][1]["content"] for c in client.calls)   # the untested option is not shown
    assert audit.apply("minimum", winner, holders) == ["coding_2"]
    assert "best = 100" in _lib(tasks[3])


def test_two_wrong_tested_forms_never_overwrite_the_right_untested_one(tmp_path):
    """The reviewer's case: two wrong forms under visible tests (1:1, the models must name one) and the right form
    in a file that does not test the function. That file keeps its form."""
    bad3 = "def minimum(xs):\n    return min(xs[1:])\n"   # green on [4, -1, 7], wrong on [3, 5]
    tasks = _tasks(tmp_path, [BAD, bad3, GOOD], tests={2: TEST_NO_MIN})
    holders, rivals = audit.census(tasks)
    shown, _ = audit.options("minimum", rivals["minimum"], holders)
    assert [o["votes"] for o in shown] == [1, 1] and not any("best = xs[0]" in o["text"] for o in shown)
    winner, note = _decide(Fake(["1"]), rivals, holders)
    assert winner is not None and "both models" in note
    assert "coding_2" not in audit.apply("minimum", winner, holders)
    assert "best = xs[0]" in _lib(tasks[2])


def test_alt_sees_the_options_in_reverse_order(tmp_path):
    holders, rivals = audit.census(_tasks(tmp_path, [GOOD, GOOD, BAD]))
    client = Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: ["1"]}, raw=True)
    winner, note = _decide(client, rivals, holders)
    assert winner is None and "no agreement (1, 2)" in note   # both said "1": the first place, two different options
    fast = next(c[1][1]["content"] for c in client.calls if c[0] == ARBITRATE_MODELS[0])
    alt = next(c[1][1]["content"] for c in client.calls if c[0] == ARBITRATE_MODELS[1])
    assert fast.index("best = xs[0]") < fast.index("best = 0") and alt.index("best = 0") < alt.index("best = xs[0]")
    row = lambda text: next(ln for ln in text.splitlines() if ln.startswith("minimum([3, 5])"))  # noqa: E731
    assert row(fast).endswith("3 | 0") and row(alt).endswith("0 | 3")
    client = Fake({ARBITRATE_MODELS[0]: ["1"], ARBITRATE_MODELS[1]: ["2"]}, raw=True)
    assert _decide(client, rivals, holders)[0]["text"] == GOOD.strip()


@pytest.mark.parametrize("text, pick", [
    ("2", 1), ("Option 2", 1), ("**2**", 1), ("Option 1 has a bug.\n2", 1), ("<think>option 1</think>Option 2.", 1),
    ("Option 1 is wrong, option 2 is right.", 1), ("the answer is option 2 (n=1)", 1),
    ("I would pick the second one", None), ("Answer: 9", None), ("", None), ("option 3", None)])
def test_reason_pick(text, pick):
    assert audit.reason_pick(text, 2) == pick


# ---------------------------------------------------------------- the stage inside a run

def solved_target(queue, task, solver_name=None, known_texts=None):
    """Phase 1 stand-in: a coding task is green as it stands; anything else stays open."""
    queue.put((task.task_id, SolveResult("solved", "DONE") if task.task_type == "coding"
               else SolveResult("unparsed", note="forced")))


def _run_dir(tmp_path, libs, libs2=None, search=0):
    root = tmp_path / "tasks"
    for i, lib in enumerate(libs):
        d = root / f"coding_{i}"
        _write(d, lib + "\n\n" + ((libs2 or {}).get(i, GOOD2)) + "\n\n" + OTHER + f"\n_TASK = {i}\n", TEST)
        (d / "task.json").write_text(json.dumps({"task_id": f"coding_{i}", "task_type": "coding", "instruction": "fix",
                                                 "reward": 2, "check": "unittest"}), encoding="utf-8")
    for i in range(search):
        d = root / f"search_{i}"
        (d / "workspace").mkdir(parents=True)
        (d / "task.json").write_text(json.dumps({"task_id": f"search_{i}", "task_type": "search",
                                                 "instruction": f"q {i}?", "reward": 1, "check": "exact_str"}),
                                     encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("HARNESS_MINE_TEXTS", "0")
    monkeypatch.delenv("HARNESS_AUDIT_FORMS", raising=False)
    monkeypatch.setattr(scheduler, "_phase1_target", solved_target)


def _mylib(root, i):
    return (root / f"coding_{i}" / "workspace" / "mylib.py").read_text(encoding="utf-8")


def test_run_audits_green_tasks_even_when_phase_1_settled_everything(tmp_path):
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD])
    client = Fake(["1"])
    counts = scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    assert counts["solved"] == 4
    assert len(client.calls) == 2                      # one function with rival forms: one call per model
    assert "best = xs[0]" in _mylib(root, 3)
    assert all((root / f"coding_{i}" / "answer.txt").read_text().strip() == "DONE" for i in range(4))
    lines = [json.loads(x) for x in (tmp_path / "log" / "trajectory.jsonl").read_text().splitlines()]
    unit = [x for x in lines if x.get("task_id") == "audit:minimum"]
    assert len(unit) == 1 and unit[0]["cost"] == 4 and "coding_3" in unit[0]["note"]


def test_run_without_rival_forms_makes_no_call(tmp_path):
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD])
    client = Fake([])
    assert scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)["solved"] == 3
    assert client.calls == []


def test_audit_switched_off_is_round_13(tmp_path, monkeypatch):
    """Off: phase 2 is not even called when phase 1 left nothing open, and the census never runs."""
    monkeypatch.setenv("HARNESS_AUDIT_FORMS", "0")
    monkeypatch.setattr(audit, "census", lambda *a, **k: pytest.fail("census with the audit off"))
    called = []
    monkeypatch.setattr(scheduler, "_phase2", lambda *a, **k: called.append(k))
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD])
    client = Fake(["1"])
    scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    assert called == [] and client.calls == [] and "best = 0" in _mylib(root, 3)


def _route_stubs(monkeypatch, client):
    async def grp(ttype, template, members, cl):
        await cl.chat("route-2a", [], 10)
        return {t.task_id: SolveResult("unparsed") for t, _ in members}

    async def prog(task, cl):
        await cl.chat("route-2b", [], 10)
        return SolveResult("unparsed", note="program failed")

    async def reason(task, cl):
        await cl.chat("route-reason", [], 10)
        return SolveResult("unparsed", note="no answer")

    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "solve_with_reason", reason)
    monkeypatch.setattr(scheduler, "json_route_batchable", lambda *a: False)
    for m in ("route-2a", "route-2b", "route-reason"):
        client.replies[m] = ["x"] * 20


def _sequence(tmp_path, monkeypatch, sub, audit_env):
    if audit_env is None:
        monkeypatch.delenv("HARNESS_AUDIT_FORMS", raising=False)
    else:
        monkeypatch.setenv("HARNESS_AUDIT_FORMS", audit_env)
    root = _run_dir(tmp_path / sub, [GOOD, GOOD, GOOD, BAD], search=2)
    client = Fake(["1"])
    _route_stubs(monkeypatch, client)
    seen = []
    monkeypatch.setattr(scheduler, "_phase2", lambda *a, **k: (seen.append(sorted(k)), _REAL_PHASE2(*a, **k)))
    scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    return [c[0] for c in client.calls], seen, root


def test_call_sequence_off_is_round_13_and_on_only_appends_the_audit(tmp_path, monkeypatch):
    off, seen_off, root_off = _sequence(tmp_path, monkeypatch, "off", "0")
    assert seen_off == [["soft_infeasible"]]           # phase 2 called with round 13's arguments
    assert off and all(m.startswith("route-") for m in off) and "best = 0" in _mylib(root_off, 3)
    on, seen_on, root_on = _sequence(tmp_path, monkeypatch, "on", None)
    assert seen_on == [["audit_tasks", "soft_infeasible"]]
    assert on[:len(off)] == off and sorted(on[len(off):]) == sorted(ARBITRATE_MODELS)   # the audit comes last
    assert "best = xs[0]" in _mylib(root_on, 3)


def test_no_audit_once_half_the_budget_is_spent(tmp_path):
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD])
    client = Fake(["1"])
    client.ledger.spent = 60000
    scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    assert client.calls == [] and "best = 0" in _mylib(root, 3)


def test_the_spend_gate_holds_for_every_unit(tmp_path, monkeypatch):
    """Two functions with rival forms, one worker: the first unit's calls cross the gate, the second unit asks
    nothing."""
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD], libs2={0: GOOD2, 1: GOOD2, 2: BAD2, 3: GOOD2})
    client = Fake(["1", "1"], charge=26000)
    scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    assert len(client.calls) == 2


def test_no_audit_too_close_to_the_deadline(tmp_path):
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD])
    client = Fake(["1"])
    scheduler.run(root, 100000, client, deadline_s=scheduler.AUDIT_MIN_LEFT_S - 1, phase1_workers=2)
    assert client.calls == []


def test_third_opinions_are_capped_per_run(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "AUDIT_MAX_REASON", 1)
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    root = _run_dir(tmp_path, [BAD, BAD, GOOD], libs2={0: BAD2, 1: BAD2, 2: GOOD2})
    client = Fake({ARBITRATE_MODELS[0]: ["2", "2"], ARBITRATE_MODELS[1]: ["2", "2"],
                   MODEL_REASON: ["Option 2.", "Option 2."]})
    scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)
    assert [c[0] for c in client.calls].count(MODEL_REASON) == 1
    fixed = sum("best = xs[0]" in _mylib(root, i) for i in range(3)) + sum("x > 0" in _mylib(root, i)
                                                                           for i in range(3))
    assert fixed == 4   # one function overturned in the two files holding the bug (3 good), the other left (1)


def test_a_failing_stage_keeps_the_files(tmp_path, monkeypatch):
    root = _run_dir(tmp_path, [GOOD, GOOD, GOOD, BAD])
    monkeypatch.setattr(audit, "census", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    client = Fake(["1"])
    assert scheduler.run(root, 100000, client, deadline_s=600, phase1_workers=2)["solved"] == 4
    assert client.calls == [] and "best = 0" in _mylib(root, 3)

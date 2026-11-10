"""Confirmation of 2a answers by a majority of independent program votes (round 8, task 40)."""
import asyncio
import json

import pytest

from harness import config, confirm, scheduler
from harness.result import SolveResult
from harness.tasks import Task
from harness.trajectory import CALLS, record_call


def _task(tmp_path, tid="math_1", check="exact_int", ttype="math", reward=2.0, instruction="How many?"):
    d = tmp_path / tid
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    return Task(tid, ttype, instruction, reward, check, d, d / "workspace")


def _program(answers, seen=None):
    """A stand-in for solve_with_program: answers[model] is an answer string, a SolveResult or an exception.
    Each vote records one model call of its own model, as the real client does."""
    async def prog(task, client, model=config.MODEL_FAST, shape_model=None):
        if seen is not None:
            seen.append((task.task_id, model, shape_model))
        a = answers[model] if not callable(answers) else answers(task, model)
        record_call(model, 100, 50, 60)
        if isinstance(a, BaseException):
            raise a
        if isinstance(a, SolveResult):
            return a
        return SolveResult("solved", answer=a, note="model program")
    return prog


def _confirm(task, answer, prog, afford=lambda: True):
    async def go():
        CALLS.set([])
        return await confirm.confirm(task, answer, object(), prog, afford)
    return asyncio.run(go())


# ---------------------------------------------------------------- the voting rule

def test_fast_agrees_keeps_2a_without_a_third_vote(tmp_path):
    seen = []
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "42"}, seen))
    assert v.replacement is None and v.fast_agrees
    assert [m for _, m, _ in seen] == [config.MODEL_FAST]


def test_votes_are_single_model_programs(tmp_path):
    seen = []
    _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "8"}, seen))
    assert [(m, s) for _, m, s in seen] == [(config.MODEL_FAST, config.MODEL_FAST), (config.MODEL_ALT, config.MODEL_ALT)]


def test_alt_siding_with_2a_keeps_2a(tmp_path):
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "42"}))
    assert v.replacement is None and not v.fast_agrees and "alt" in v.note


def test_fast_and_alt_agreeing_on_another_answer_replace_2a(tmp_path):
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}))
    assert v.replacement == "7" and "42" in v.note


def test_three_different_answers_keep_2a(tmp_path):
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "8"}))
    assert v.replacement is None


@pytest.mark.parametrize("bad", [
    SolveResult("infeasible", note="program printed IMPOSSIBLE"),
    SolveResult("unparsed", note="program timed out"),
    SolveResult("unparsed", note="llm unavailable: HTTP 500"),
    SolveResult("unparsed", note="answer shape: x", extra={"best_guess": "x", "shape_mismatch": True}),
    SolveResult("solved", answer="IMPOSSIBLE"),
    SolveResult("solved", answer="seven"),   # wrong shape for exact_int
])
def test_a_fast_vote_that_does_not_count_keeps_2a_and_buys_no_third(tmp_path, bad):
    seen = []
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: bad, config.MODEL_ALT: "7"}, seen))
    assert v.replacement is None and not v.fast_agrees and len(seen) == 1


@pytest.mark.parametrize("bad", [
    SolveResult("infeasible", note="program printed IMPOSSIBLE"),
    SolveResult("unparsed", note="program exit code 1"),
    SolveResult("solved", answer="seven"),
])
def test_an_alt_vote_that_does_not_count_keeps_2a(tmp_path, bad):
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: bad}))
    assert v.replacement is None


def test_wrong_shape_search_answer_never_replaces(tmp_path):
    t = _task(tmp_path, tid="search_1", ttype="search", check="exact_str",
              instruction="In which year was the mentor of Ann Lee born?")
    v = _confirm(t, "1970", _program({config.MODEL_FAST: "Bob Ray", config.MODEL_ALT: "Bob Ray"}))
    assert v.replacement is None


def test_a_crashing_vote_does_not_count(tmp_path):
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: RuntimeError("boom")}))
    assert v.replacement is None


def test_budget_exhausted_is_not_swallowed(tmp_path):
    from harness.llm import BudgetExhausted
    with pytest.raises(BudgetExhausted):
        _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: BudgetExhausted("x")}))


def test_no_third_vote_when_the_ledger_is_short(tmp_path):
    seen = []
    v = _confirm(_task(tmp_path), "42", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen),
                 afford=lambda: False)
    assert v.replacement is None and len(seen) == 1


def test_votes_answered_by_the_same_model_are_not_independent(tmp_path):
    async def prog(task, client, model=config.MODEL_FAST, shape_model=None):
        record_call(config.MODEL_ALT, 100, 50, 60)   # the FAST vote fell back to ALT (FAST failing)
        return SolveResult("solved", answer="7")
    v = _confirm(_task(tmp_path), "42", prog)
    assert v.replacement is None and "independent" in v.note


@pytest.mark.parametrize("a, b, same", [
    ("12.50", "12.5", True), ("1000", "1000.0", True), ("-3", "-3", True), ("3", "-3", False),
    ("Acme Corp", "acme  corp", True), ("Acme", "Acme Corp", False), ("1970", "1971", False),
    ("NaN", "NaN", True), ("", "0", False),
])
def test_same_answer(a, b, same):
    assert confirm.same_answer(a, b) is same


def test_numerically_equal_answer_agrees_and_keeps_2a_spelling(tmp_path):
    t = _task(tmp_path, check="exact_str")
    v = _confirm(t, "12.50", _program({config.MODEL_FAST: "12.5"}))
    assert v.replacement is None and v.fast_agrees


# ---------------------------------------------------------------- the program route takes a model

class _Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        from harness.llm import ChatResult
        self.calls.append({"model": model, "messages": messages})
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=10, cost=11)


def test_program_route_runs_on_the_given_model_and_shape_retry_model(tmp_path):
    from harness.llm_routes import solve_with_program
    t = _task(tmp_path, tid="search_1", ttype="search", check="exact_str",
              instruction="In which year was the mentor of Ann Lee born?")
    c = _Client(["print('Bob' + ' Ray')", "print(1900 + 70)"])
    r = asyncio.run(solve_with_program(t, c, model=config.MODEL_ALT, shape_model=config.MODEL_ALT))
    assert r.answer == "1970" and [x["model"] for x in c.calls] == [config.MODEL_ALT, config.MODEL_ALT]


def test_program_route_default_is_unchanged(tmp_path):
    from harness.llm_routes import solve_with_program
    t = _task(tmp_path, tid="search_1", ttype="search", check="exact_str",
              instruction="In which year was the mentor of Ann Lee born?")
    c = _Client(["print('Bob' + ' Ray')", "print(1900 + 70)"])
    asyncio.run(solve_with_program(t, c))
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT]


# ---------------------------------------------------------------- the knob

def test_knob(monkeypatch):
    monkeypatch.delenv("HARNESS_CONFIRM_2A", raising=False)
    assert config.confirm_2a() is config.CONFIRM_2A
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "0")
    assert config.confirm_2a() is False
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "1")
    assert config.confirm_2a() is True
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "maybe")
    assert config.confirm_2a() is config.CONFIRM_2A


# ---------------------------------------------------------------- the scheduler stage

def unparsed_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


def solved_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("solved", answer="1", note="phase 1")))


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "1")
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")


def _make(dest, tid, ttype="math", instruction=None, reward=1.0, check="exact_int"):
    d = dest / tid
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({
        "task_id": tid, "task_type": ttype, "instruction": instruction or f"Riddle {tid}?",
        "reward": reward, "check": check}), encoding="utf-8")
    return d


def _setup(tmp_path, monkeypatch, answers_2a, rewards=None, groups=None, budget=100000, spent=0.0):
    """answers_2a: {tid: 2a answer}; groups: [[tid, ...], ...] (default: one group per task).
    The mechanism tests run on math tasks, so math is switched on here; the shipped default is data only
    (see test_default_confirms_data_only)."""
    monkeypatch.setattr(config, "CONFIRM_TYPES", ("math", "data_analysis"))
    tasks = tmp_path / "tasks"
    rewards = rewards or {}
    for tid in answers_2a:
        _make(tasks, tid, reward=rewards.get(tid, 1.0))
    groups = groups or [[tid] for tid in answers_2a]
    gid = {tid: k for k, g in enumerate(groups) for tid in g}

    def fake_group(tasks_):
        out = {}
        for t in tasks_:
            out.setdefault(("math", f"template {gid[t.task_id]}"), []).append((t, {}))
        return out

    async def grp(ttype, template, members, client):
        return {t.task_id: SolveResult("solved", answer=answers_2a[t.task_id], note="json route")
                for t, _ in members}

    async def prog_2b(task, client):
        raise AssertionError("2b must not run for 2a answers")

    from harness import templates
    monkeypatch.setattr(templates, "group", fake_group)
    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog_2b)
    from harness.budget import TokenLedger

    class Client:
        ledger = TokenLedger(budget, 0)
    Client.ledger.spent = spent
    return tasks, Client()


def _answers(tasks):
    return {d.name: (d / "answer.txt").read_text(encoding="utf-8").strip()
            for d in sorted(tasks.iterdir()) if (d / "answer.txt").exists()}


def _traj(tmp_path):
    return [json.loads(x) for x in (tmp_path / "logs" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]


def test_stage_replaces_only_on_a_majority_and_records_costs(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42", "math_b": "10", "math_c": "5"})
    votes = {("math_a", config.MODEL_FAST): "7", ("math_a", config.MODEL_ALT): "7",     # majority against 2a
             ("math_b", config.MODEL_FAST): "11", ("math_b", config.MODEL_ALT): "10",   # alt sides with 2a
             ("math_c", config.MODEL_FAST): "5"}                                        # fast agrees
    monkeypatch.setattr(scheduler, "confirm_program", _program(lambda t, m: votes[(t.task_id, m)]))
    counts = scheduler.run(tasks, 100000, client)
    assert _answers(tasks) == {"math_a": "7", "math_b": "10", "math_c": "5"}
    assert counts["solved"] == 3
    rows = [r for r in _traj(tmp_path) if (r.get("unit") or "").startswith("confirm:")]
    cost = {r["task_id"]: r["cost"] for r in rows}
    assert cost == {"math_a": 120, "math_b": 120, "math_c": 60}
    assert all(r["status"] == "solved" for r in rows)


def test_knob_off_buys_no_votes(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "0")
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"})
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}))
    scheduler.run(tasks, 100000, client)
    assert _answers(tasks) == {"math_a": "42"}


def test_phase1_answers_are_not_confirmed(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"})
    monkeypatch.setattr(scheduler, "_phase1_target", solved_target)
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen))
    scheduler.run(tasks, 100000, client)
    assert seen == [] and _answers(tasks) == {"math_a": "1"}


def test_confirmation_goes_in_reward_order(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "1", "math_b": "2", "math_c": "3"},
                           rewards={"math_a": 1, "math_b": 5, "math_c": 3})
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program(lambda t, m: _answers(tasks)[t.task_id], seen))
    scheduler.run(tasks, 100000, client)
    assert [tid for tid, _, _ in seen] == ["math_b", "math_c", "math_a"]


def test_big_group_accepted_on_two_agreeing_samples(tmp_path, monkeypatch):
    ids = [f"math_{i}" for i in range(5)]
    tasks, client = _setup(tmp_path, monkeypatch, {t: "9" for t in ids}, groups=[ids],
                           rewards={"math_3": 5, "math_1": 4})
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "9"}, seen))
    scheduler.run(tasks, 100000, client)
    assert [tid for tid, _, _ in seen] == ["math_3", "math_1"]


def test_big_group_checks_every_member_after_a_disagreement(tmp_path, monkeypatch):
    ids = [f"math_{i}" for i in range(4)]
    tasks, client = _setup(tmp_path, monkeypatch, {t: "9" for t in ids}, groups=[ids],
                           rewards={"math_0": 5})
    seen = []
    votes = lambda t, m: "8" if t.task_id == "math_0" else "9"   # noqa: E731
    monkeypatch.setattr(scheduler, "confirm_program", _program(votes, seen))
    scheduler.run(tasks, 100000, client)
    assert sorted({tid for tid, _, _ in seen}) == ids
    assert _answers(tasks)["math_0"] == "8" and _answers(tasks)["math_1"] == "9"


def test_skipped_when_the_ledger_is_short(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"}, budget=1000, spent=900)  # 2a (~80) fits, a vote (~120) does not
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen))
    scheduler.run(tasks, 1000, client)
    assert seen == [] and _answers(tasks) == {"math_a": "42"}


def test_skipped_near_the_deadline(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"})
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen))
    monkeypatch.setattr(scheduler, "_confirm_time_left", lambda *a: False)
    scheduler.run(tasks, 100000, client)
    assert seen == [] and _answers(tasks) == {"math_a": "42"}


def test_stage_runs_after_the_reason_pass_and_before_2d(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"})
    _make(tasks, "search_x", ttype="search", instruction="q?", reward=9, check="exact_str")
    order = []

    async def prog_2b(task, client):
        order.append("2b")
        return SolveResult("unparsed", note="program failed")

    async def reason(task, client):
        order.append("reason")
        return SolveResult("unparsed", note="no")

    async def votes(task, client, model=config.MODEL_FAST, shape_model=None):
        order.append("confirm")
        return SolveResult("solved", answer="42")

    from harness import templates
    real_group = templates.group

    def group2(ts):
        return real_group([t for t in ts if t.task_id != "search_x"]) | {("search", "s"): [(t, {}) for t in ts if t.task_id == "search_x"]}
    monkeypatch.setattr(templates, "group", group2)

    async def grp(ttype, template, members, client):
        return {t.task_id: (SolveResult("solved", answer="42") if t.task_id == "math_a" else SolveResult("unparsed"))
                for t, _ in members}
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog_2b)
    monkeypatch.setattr(scheduler, "solve_with_reason", reason)
    monkeypatch.setattr(scheduler, "confirm_program", votes)
    scheduler.run(tasks, 100000, client)
    assert order == ["2b", "reason", "confirm"]


def test_2d_keeps_its_budget(tmp_path, monkeypatch):
    """With a ledger that covers the 2d check but not the 2d check plus a vote, the vote is not bought."""
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"}, budget=100000)
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen))
    monkeypatch.setattr(scheduler, "_reserve_2d", lambda units: 100000 - 50)
    scheduler.run(tasks, 100000, client)
    assert seen == []


def test_votes_never_see_the_2a_answer(tmp_path, monkeypatch):
    """The real program route on a fake client: no message carries the 2a answer."""
    tasks, client_ = _setup(tmp_path, monkeypatch, {"math_a": "987654"})
    from harness import llm_routes
    from harness.llm import ChatResult
    sent = []

    class Client:
        ledger = client_.ledger

        async def chat(self, model, messages, max_tokens, **extra):
            sent.append(json.dumps(messages))
            return ChatResult(text="print(3 + 4)", prompt_tokens=10, completion_tokens=10, cost=11)
    monkeypatch.setattr(scheduler, "confirm_program", llm_routes.solve_with_program)
    scheduler.run(tasks, 100000, Client())
    assert len(sent) == 2 and not any("987654" in s for s in sent)
    assert _answers(tasks) == {"math_a": "7"}


def verify_target(queue, task, solver_name=None):
    if task.task_type == "coding":
        queue.put((task.task_id, SolveResult("solved", answer="DONE", note="fixed", tentative=True,
                                             extra={"verify": [{"func": "f"}]})))
    else:
        queue.put((task.task_id, SolveResult("unparsed", note="forced")))


def test_2d_runs_after_the_stage_and_its_estimate_is_held_back(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"})
    d = _make(tasks, "coding_x", ttype="coding", instruction="fix", check="unittest")
    (d / "workspace" / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    order, reserves = [], []

    async def votes(task, client, model=config.MODEL_FAST, shape_model=None):
        order.append("confirm")
        return SolveResult("solved", answer="42")

    async def verify(task, extra, client):
        order.append("2d")
        return SolveResult("solved", answer="DONE", note="checked")
    real = scheduler._reserve_2d
    monkeypatch.setattr(scheduler, "_reserve_2d", lambda units: reserves.append(real(units)) or reserves[-1])
    monkeypatch.setattr(scheduler, "confirm_program", votes)
    monkeypatch.setattr(scheduler, "verify_fix", verify)
    scheduler.run(tasks, 100000, client)
    assert order == ["confirm", "2d"]
    assert reserves == [pytest.approx(len(str({"verify": [{"func": "f"}]})) // 30 + config.VERIFY_MAX_TOKENS)]


def test_search_answers_are_not_confirmed(tmp_path, monkeypatch):
    """Search is left out: its votes cost ~10x math/data in live runs and never changed an answer."""
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42", "search_b": "Bob Ray"})
    tj = tasks / "search_b" / "task.json"
    tj.write_text(tj.read_text(encoding="utf-8").replace('"math"', '"search"').replace('"exact_int"', '"exact_str"'),
                  encoding="utf-8")
    seen = []
    votes = lambda t, m: "Ann Lee" if t.task_id == "search_b" else "42"   # noqa: E731
    monkeypatch.setattr(scheduler, "confirm_program", _program(votes, seen))
    scheduler.run(tasks, 100000, client)
    assert {tid for tid, _, _ in seen} == {"math_a"} and "search" not in config.CONFIRM_TYPES
    assert _answers(tasks) == {"math_a": "42", "search_b": "Bob Ray"}


def test_no_vote_once_half_the_budget_is_spent(tmp_path, monkeypatch):
    tasks, client = _setup(tmp_path, monkeypatch, {"math_a": "42"}, spent=50000)
    seen = []
    monkeypatch.setattr(scheduler, "confirm_program", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}, seen))
    scheduler.run(tasks, 100000, client)
    assert seen == [] and _answers(tasks) == {"math_a": "42"}


# ---------------------------------------------------------------- fix round 1: real program votes on a fake client

class _CodeClient:
    """Every call answers with `codes[model]` (one program per model); `answered` maps a requested model to the
    model that really answers (a fallback), `failing` models raise ModelFailing."""

    def __init__(self, codes, failing=()):
        self.codes, self.failing, self.models = codes, set(failing), []

    async def chat(self, model, messages, max_tokens, **extra):
        from harness.llm import ChatResult, ModelFailing
        if model in self.failing:
            record_call(model, 0, 0, 0.0, error="failing")
            raise ModelFailing("down")
        self.models.append(model)
        record_call(model, 10, 10, 11)
        return ChatResult(text=self.codes[model], prompt_tokens=10, completion_tokens=10, cost=11)


def _vote_real(task, answer_2a, client):
    from harness import llm_routes

    async def go():
        CALLS.set([])
        return await confirm.confirm(task, answer_2a, client, llm_routes.solve_with_program, lambda: True)
    return asyncio.run(go())


SALES = "tests/fixtures/confirm_v1live_sales.csv"
_HEAD = "The file sales.csv has columns order_id,region,product,month,units,price (300 rows). "


def _sales_task(tmp_path, tid, question, check):
    from pathlib import Path
    t = _task(tmp_path, tid=tid, ttype="data_analysis", check=check, instruction=_HEAD + question)
    (t.workspace / "sales.csv").write_bytes((Path(__file__).parent.parent / SALES).read_bytes())
    return t


def test_p2_a_fractional_print_is_no_vote_for_exact_int(tmp_path):
    t = _task(tmp_path, instruction="63 apples are shared equally among 5 baskets. "
                                    "How many whole apples go into each basket?")
    code = "apples = 63\nbaskets = 5\nprint(apples / baskets)"
    v = _vote_real(t, "12", _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code}))
    assert v.replacement is None and "did not count" in v.note


def test_an_integral_float_print_is_a_vote(tmp_path):
    t = _task(tmp_path, instruction="63 apples are shared equally among 5 baskets. How many go into each basket?")
    code = "print(float(63 // 5) + 1)"   # 13.0
    v = _vote_real(t, "12", _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code}))
    assert v.replacement == "13"


def test_p1_float_noise_is_no_disagreement(tmp_path):
    csv = "order_id,region,product,month,units,price\n1,east,gizmo,1,1,0.1\n2,east,gizmo,1,1,0.2\n"
    t = _task(tmp_path, tid="data_1", ttype="data_analysis", check="exact_str",
              instruction="The file sales.csv has columns order_id,region,product,month,units,price (2 rows). "
                          "What is the total revenue?")
    (t.workspace / "sales.csv").write_text(csv, encoding="utf-8")
    code = ("import csv\nrows=list(csv.DictReader(open('sales.csv')))\n"
            "print(sum(int(r['units'])*float(r['price']) for r in rows))")
    c = _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code})
    v = _vote_real(t, "0.3", c)
    assert v.replacement is None and v.fast_agrees and c.models == [config.MODEL_FAST]


def test_a_real_difference_at_2a_precision_still_shows(tmp_path):
    t = _task(tmp_path, tid="data_1", ttype="data_analysis", check="exact_str",
              instruction="What is the total revenue?")
    code = "print(142.5649999)"   # 142.56 at 2a's two places
    v = _vote_real(t, "142.55", _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code}))
    assert v.replacement == "142.56"


def test_replacement_is_written_at_2a_precision(tmp_path):
    t = _task(tmp_path, tid="data_1", ttype="data_analysis", check="exact_str",
              instruction="What is the total revenue?")
    code = "print(1.1 + 2.2)"   # 3.3000000000000003
    v = _vote_real(t, "2.5", _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code}))
    assert v.replacement == "3.3"


def test_p4_a_vote_that_fell_back_to_the_fast_weights_does_not_count(tmp_path):
    """ALT and FAST failing during the ALT vote: chat_or_alt reaches MODEL_REASON, the FAST weights with reasoning."""
    from harness import llm_routes
    t = _task(tmp_path)

    class Client(_CodeClient):
        async def chat(self, model, messages, max_tokens, **extra):
            if self.models:   # from the second call on, the ALT vote: ALT and FAST are failing
                self.failing = {config.MODEL_ALT, config.MODEL_FAST}
            return await super().chat(model, messages, max_tokens, **extra)
    code = "print(3 + 4)"
    c = Client({config.MODEL_FAST: code, config.MODEL_ALT: code, config.MODEL_REASON: code})

    async def go():
        CALLS.set([])
        return await confirm.confirm(t, "42", c, llm_routes.solve_with_program, lambda: True)
    v = asyncio.run(go())
    assert c.models == [config.MODEL_FAST, config.MODEL_REASON]
    assert v.replacement is None and "independent" in v.note


def test_model_families():
    assert confirm.family(config.MODEL_FAST) == confirm.family(config.MODEL_REASON)
    assert confirm.family(config.MODEL_ALT) != confirm.family(config.MODEL_FAST)


def test_gate_fix_9020_round_down_still_replaced(tmp_path):
    """v1live data_analysis_9020: 2a rounded 24.99 half-up to 25; both votes floored to 24 (the expected answer)."""
    t = _sales_task(tmp_path, "data_analysis_9020",
                    "What is the average number of units per order for product 'gadget'? Round down to an integer.",
                    "exact_int")
    fast = ("import csv\nrows = [r for r in csv.DictReader(open('sales.csv')) if r['product'] == 'gadget']\n"
            "units = [int(r['units']) for r in rows]\nprint(sum(units) // len(units))")
    alt = ("import csv, math\nwith open('sales.csv') as f:\n    u = [int(r['units']) for r in csv.DictReader(f) "
           "if r['product'] == 'gadget']\nprint(math.floor(sum(u) / len(u)))")
    v = _vote_real(t, "25", _CodeClient({config.MODEL_FAST: fast, config.MODEL_ALT: alt}))
    assert v.replacement == "24"


def test_gate_fix_9007_lowest_region_still_replaced(tmp_path):
    """v1live data_analysis_9007: 2a answered the highest-revenue region; both votes found the lowest (west)."""
    t = _sales_task(tmp_path, "data_analysis_9007", "Which region had the lowest total revenue?", "exact_str")
    fast = ("import csv\nfrom collections import defaultdict\nrev = defaultdict(float)\n"
            "for r in csv.DictReader(open('sales.csv')):\n    rev[r['region']] += int(r['units']) * float(r['price'])\n"
            "print(min(rev, key=rev.get))")
    alt = ("import pandas as pd\ndf = pd.read_csv('sales.csv')\ndf['rev'] = df['units'] * df['price']\n"
           "print(df.groupby('region')['rev'].sum().idxmin())")
    v = _vote_real(t, "east", _CodeClient({config.MODEL_FAST: fast, config.MODEL_ALT: alt}))
    assert v.replacement == "west"


# ---------------------------------------------------------------- fix round 2: reviewer's probes (p_r1.py)

_Q = "The file sales.csv has columns order_id,region,product,month,units,price (2 rows). "
_CSV = "order_id,region,product,month,units,price\n1,east,gizmo,1,1000,1.2345\n2,east,gizmo,1,234,1.0\n"
_KB = {"kb/ann.md": "# Ann Lee\n- Employer: Acme Corp\n"}
_SQ = "The directory kb/ contains a small knowledge base of markdown files. Who is the employer of Ann Lee?"


def _probe(tmp_path, ttype, check, instruction, answer_2a, code, files=None):
    t = _task(tmp_path, tid=f"{ttype}_p", ttype=ttype, check=check, instruction=instruction)
    for name, text in (files or {}).items():
        p = t.workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return _vote_real(t, answer_2a, _CodeClient({config.MODEL_FAST: code, config.MODEL_ALT: code}))


@pytest.mark.parametrize("label, ttype, check, instruction, answer_2a, code, files", [
    ("N1a thousands", "data_analysis", "exact_str", _Q + "What is the total revenue, rounded to 2 decimal places?",
     "1468.5", "x = 1000*1.2345 + 234*1.0\nprint(f'{x:,.2f}')", {"sales.csv": _CSV}),
    ("N1b currency", "data_analysis", "exact_str", _Q + "What is the total revenue, rounded to 2 decimal places?",
     "1468.5", "x = 1000*1.2345 + 234*1.0\nprint(f'${x:.2f}')", {"sales.csv": _CSV}),
    ("N2 nearest thousandth", "data_analysis", "exact_str", _Q + "What is the average price, to the nearest thousandth?",
     "1.117", "x = (1.2345 + 1.0) / 2\nprint(round(x, 3))", {"sales.csv": _CSV}),
    ("N2b 3 dp in parens", "data_analysis", "exact_str", _Q + "What is the average price (3 d.p.)?",
     "1.117", "x = (1.2345 + 1.0) / 2\nprint(round(x, 3))", {"sales.csv": _CSV}),
    ("N3 percent", "data_analysis", "exact_str", _Q + "What share of units is in month 1, in percent?",
     "100", "print(str(100.0) + '%')", {"sales.csv": _CSV}),
    ("N4 neg zero str", "data_analysis", "exact_str", _Q + "What is the change in price, rounded to 2 decimal places?",
     "0.00", "print('%.2f' % (-0.001))", {"sales.csv": _CSV}),
    ("N4b neg zero int", "math", "exact_int", "A has 5 and loses 5. How many are left?", "0", "a = 5\nprint(-(a - 5) * 1.0)", None),
    ("N5a case+space", "search", "exact_str", _SQ, "Acme Corp", "s = 'ACME  corp'\nprint(s)", _KB),
    ("N5c markdown bold", "search", "exact_str", _SQ, "Acme Corp", "x = 'Acme Corp'\nprint('**' + x + '**')", _KB),
    ("N6a debug after (int)", "math", "exact_int", "A store has 500 apples and sells 80. How many remain?",
     "420", "a = 500 - 80\nprint(a)\nprint('checked', 2, 'steps')", None),
    ("N6b debug after (str)", "data_analysis", "exact_str", _Q + "Which region has most units?",
     "east", "r = 'east'\nprint(r)\nprint('done')", {"sales.csv": _CSV}),
    ("N6c answer prefix", "data_analysis", "exact_str", _Q + "Which region has most units?",
     "east", "r = 'east'\nprint('Region: ' + r)", {"sales.csv": _CSV}),
    ("N7 exponent int", "math", "exact_int", "A has 20 boxes of 1000000000000000000 nails. How many nails?",
     "20000000000000000000", "print(20 * 1e18)", None),
    ("N8 tie half", "data_analysis", "exact_str", _Q + "What is the average price, rounded to 2 decimal places?",
     "1.12", "print(round((1.2345 + 1.0) / 2, 4))", {"sales.csv": _CSV}),
])
def test_probe_keeps_a_correct_2a_answer(tmp_path, label, ttype, check, instruction, answer_2a, code, files):
    v = _probe(tmp_path, ttype, check, instruction, answer_2a, code, files)
    assert v.replacement is None, label


def test_a_single_bare_line_still_replaces(tmp_path):
    v = _probe(tmp_path, "data_analysis", "exact_str", _Q + "Which region has most units?", "west",
               "print('east')", {"sales.csv": _CSV})
    assert v.replacement == "east"


def test_a_name_vote_passes_the_narrow_name_check(tmp_path):
    v = _probe(tmp_path, "search", "exact_str", _SQ, "Bob Ray", "print('Acme Corp')", _KB)
    assert v.replacement == "Acme Corp"


def test_three_place_2a_answer_compares_at_three_places(tmp_path):
    """2a shows three places: a vote that differs in the third place is a real difference, written at three places."""
    v = _probe(tmp_path, "data_analysis", "exact_str", _Q + "What is the average price, to the nearest thousandth?",
               "1.117", "print(1.1183)", {"sales.csv": _CSV})
    assert v.replacement == "1.118"


def test_exact_int_accepts_a_trailing_zero_fraction(tmp_path):
    v = _probe(tmp_path, "math", "exact_int", "How many?", "12", "print(13.0)")
    assert v.replacement == "13"


def test_default_confirms_data_only():
    """math votes outvoted a correct spec-based answer (runs/r5/llm7 math_0008) and never fixed one; search votes cost
    ~10x and never changed an answer; data votes made the only measured fixes."""
    assert config.CONFIRM_TYPES == ("data_analysis",)


def test_zero_votes_never_replace_non_zero_2a(tmp_path):
    for check, zero in (("exact_str", "0"), ("exact_str", "0.00"), ("exact_int", "0")):
        seen = []
        v = _confirm(_task(tmp_path, check=check), "64092",
                     _program({config.MODEL_FAST: zero, config.MODEL_ALT: zero}, seen))
        assert v.replacement is None and not v.fast_agrees
        assert "both votes" in v.note and "kept 2a" in v.note
        assert [m for _, m, _ in seen] == [config.MODEL_FAST, config.MODEL_ALT]   # the alt vote was really requested


def test_guard_does_not_block_replacement_of_a_zero_2a(tmp_path):
    v = _confirm(_task(tmp_path), "0", _program({config.MODEL_FAST: "5", config.MODEL_ALT: "5"}))
    assert v.replacement == "5"


def test_zero_2a_with_zero_votes_is_unchanged(tmp_path):
    for a in ("0", "-0", "0.0"):
        v = _confirm(_task(tmp_path, check="exact_str"), a, _program({config.MODEL_FAST: "0", config.MODEL_ALT: "0"}))
        assert v.replacement is None and v.fast_agrees


def test_non_zero_agreement_still_replaces(tmp_path):
    v = _confirm(_task(tmp_path), "64092", _program({config.MODEL_FAST: "7", config.MODEL_ALT: "7"}))
    assert v.replacement == "7"


def test_non_numeric_2a_with_zero_votes_keeps_previous_behaviour(tmp_path):
    v = _confirm(_task(tmp_path, check="exact_str"), "Gizmo", _program({config.MODEL_FAST: "0", config.MODEL_ALT: "0"}))
    assert v.replacement == "0" and "both votes" not in v.note

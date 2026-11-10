"""Round 10, task 56: math on the model path. MODEL_REASON answers math in 2b (the FAST program is the fallback) and
is the vote on 2a math answers; 2a changes only when a FAST program sides with REASON. HARNESS_REASON_MATH=0 is
round 9."""
import asyncio
import json
from pathlib import Path

import pytest

from harness import config, scheduler
from harness.result import SolveResult
from harness.tasks import Task


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HARNESS_CONFIRM_2A", "1")
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")
    monkeypatch.delenv("HARNESS_REASON_MATH", raising=False)
    monkeypatch.delenv("HARNESS_CONFIRM_MATH", raising=False)


def unparsed_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


def _make(dest, tid, ttype, reward=1.0, check="exact_int", instruction=None):
    d = dest / tid
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({
        "task_id": tid, "task_type": ttype, "instruction": instruction or f"Riddle {tid}?",
        "reward": reward, "check": check}), encoding="utf-8")
    return d


def _answers(tasks):
    return {d.name: (d / "answer.txt").read_text(encoding="utf-8").strip()
            for d in sorted(tasks.iterdir()) if (d / "answer.txt").exists()}


# ---------------------------------------------------------------- one scenario through the whole of phase 2

# 2a answers (None: 2a gives nothing, the task goes to 2b); groups of the template grouping
_2A = {"math_a": "42", "math_b": "10", "math_g0": "9", "math_g1": "9", "math_g2": "9",
       "math_u": None, "math_v": None, "data_d": "3.5", "search_s": None}
_GROUPS = [["math_a"], ["math_b"], ["math_g0", "math_g1", "math_g2"], ["math_u"], ["math_v"], ["data_d"], ["search_s"]]
_REWARDS = {"math_a": 3, "math_b": 2, "math_g0": 2, "math_g1": 1, "math_g2": 1, "math_u": 3, "math_v": 3,
            "data_d": 2, "search_s": 3}
_TYPE = {"math": "math", "data": "data_analysis", "search": "search"}


def _scenario(tmp_path, monkeypatch, *, reason_answers=None, budget=100000, spent=0.0):
    tasks = tmp_path / "tasks"
    for tid in _2A:
        ttype = _TYPE[tid.split("_")[0]]
        _make(tasks, tid, ttype, reward=_REWARDS[tid], check="exact_int" if ttype == "math" else "exact_str")
    gid = {tid: k for k, g in enumerate(_GROUPS) for tid in g}
    seq = []

    def fake_group(tasks_):
        out = {}
        for t in tasks_:
            out.setdefault((t.task_type, f"template {gid[t.task_id]}"), []).append((t, {}))
        return out

    async def grp(ttype, template, members, client):
        seq.append(("2a", tuple(t.task_id for t, _ in members)))
        return {t.task_id: (SolveResult("solved", answer=_2A[t.task_id], note="json route") if _2A[t.task_id]
                            else SolveResult("unparsed", note="spec declined"))
                for t, _ in members}

    prog_answers = {"math_u": SolveResult("solved", "5"), "math_v": SolveResult("infeasible", note="program printed IMPOSSIBLE"),
                    "search_s": SolveResult("unparsed", note="program failed")}

    async def prog(task, client):
        seq.append(("2b", task.task_id))
        return prog_answers[task.task_id]

    reason_answers = reason_answers or {}

    async def reason(task, client):
        seq.append(("reason", task.task_id))
        return reason_answers.get(task.task_id, SolveResult("unparsed", note="reason program gave no usable answer"))

    votes = {("math_a", config.MODEL_FAST): "7", ("math_a", config.MODEL_ALT): "7",
             ("math_b", config.MODEL_FAST): "10", ("data_d", config.MODEL_FAST): "4.5",
             ("data_d", config.MODEL_ALT): "4.5"}

    async def vote(task, client, model=config.MODEL_FAST, shape_model=None):
        seq.append(("vote", task.task_id, model))
        a = votes.get((task.task_id, model), "9")
        return SolveResult("solved", answer=a, extra={"stdout": a + "\n"})

    from harness import templates
    from harness.budget import TokenLedger
    monkeypatch.setattr(templates, "group", fake_group)
    monkeypatch.setattr(scheduler, "LLM_WORKERS", 1)
    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)
    monkeypatch.setattr(scheduler, "solve_group", grp)
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "solve_with_reason", reason)
    monkeypatch.setattr(scheduler, "confirm_program", vote)

    class Client:
        ledger = TokenLedger(budget, 0)
    Client.ledger.spent = spent
    return tasks, Client(), seq


# recorded on e65cbb2 (round 9) with this scenario
_ROUND9_SEQ = [
    ("2a", ("math_g0", "math_g1", "math_g2")), ("2a", ("math_a",)), ("2a", ("math_u",)), ("2a", ("math_v",)),
    ("2a", ("search_s",)), ("2a", ("data_d",)), ("2a", ("math_b",)),
    ("2b", "math_u"), ("2b", "math_v"), ("2b", "search_s"),
    ("reason", "math_v"), ("reason", "search_s"),
    ("vote", "data_d", config.MODEL_FAST), ("vote", "data_d", config.MODEL_ALT),
]
_ROUND9_ANSWERS = {"math_a": "42", "math_b": "10", "math_g0": "9", "math_g1": "9", "math_g2": "9", "math_u": "5",
                   "data_d": "4.5"}


def test_flag_off_is_round_9_call_for_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REASON_MATH", "0")
    tasks, client, seq = _scenario(tmp_path, monkeypatch)
    scheduler.run(tasks, 100000, client)
    assert seq == _ROUND9_SEQ
    assert _answers(tasks) == _ROUND9_ANSWERS


_REASON_ANSWERS = {"math_u": SolveResult("solved", "6"),
                   "math_v": SolveResult("infeasible", note="reason program printed IMPOSSIBLE"),
                   "math_a": SolveResult("solved", "7"),      # the FAST and ALT programs side with it: replaced
                   "math_b": SolveResult("solved", "11"),     # the FAST program says 2a's 10: kept
                   "math_g0": SolveResult("solved", "9"), "math_g1": SolveResult("solved", "9")}
# 2b and the final pass with HARNESS_REASON_MATH on (the default)
_REASON_2B_SEQ = _ROUND9_SEQ[:7] + [
    ("reason", "math_u"),                               # answered: no program
    ("reason", "math_v"), ("2b", "math_v"),             # IMPOSSIBLE: the program is asked too
    ("2b", "search_s"),
    ("reason", "search_s"),                             # the final pass skips the math tasks REASON answered
    ("vote", "data_d", config.MODEL_FAST), ("vote", "data_d", config.MODEL_ALT),   # data first, as in round 9
]


def test_by_default_reason_answers_math_in_2b_and_no_math_answer_is_voted_on(tmp_path, monkeypatch):
    tasks, client, seq = _scenario(tmp_path, monkeypatch, reason_answers=_REASON_ANSWERS)
    scheduler.run(tasks, 100000, client)
    assert seq == _REASON_2B_SEQ
    assert _answers(tasks) == {**_ROUND9_ANSWERS, "math_u": "6"}


def test_math_votes_on_reason_answers_math_in_2b_and_votes_on_2a_math(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CONFIRM_MATH", "1")
    tasks, client, seq = _scenario(tmp_path, monkeypatch, reason_answers=_REASON_ANSWERS)
    scheduler.run(tasks, 100000, client)
    assert seq == _REASON_2B_SEQ + [
        ("reason", "math_g0"), ("reason", "math_g1"),       # the group's two best-paid agree: math_g2 is not asked
        ("reason", "math_a"), ("vote", "math_a", config.MODEL_FAST), ("vote", "math_a", config.MODEL_ALT),
        ("reason", "math_b"), ("vote", "math_b", config.MODEL_FAST),
    ]
    assert _answers(tasks) == {**_ROUND9_ANSWERS, "math_u": "6", "math_a": "7"}


def test_math_votes_on_with_reason_math_off_is_still_round_9(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CONFIRM_MATH", "1")
    monkeypatch.setenv("HARNESS_REASON_MATH", "0")
    tasks, client, seq = _scenario(tmp_path, monkeypatch, reason_answers=_REASON_ANSWERS)
    scheduler.run(tasks, 100000, client)
    assert seq == _ROUND9_SEQ and _answers(tasks) == _ROUND9_ANSWERS


def test_no_reason_call_that_the_ledger_cannot_hold_whole(tmp_path, monkeypatch):
    # 2a and the program votes fit, a whole REASON call (3000 out) next to the task's prompt does not: round 9 for
    # math (the fakes charge nothing, so the final pass's own gate, an estimate of 3000, still lets it run as before)
    monkeypatch.setenv("HARNESS_CONFIRM_MATH", "1")
    reason_answers = {t: SolveResult("solved", "7") for t in ("math_u", "math_a", "math_b", "math_g0", "math_g1")}
    tasks, client, seq = _scenario(tmp_path, monkeypatch, budget=3000, reason_answers=reason_answers)
    scheduler.run(tasks, 3000, client)
    assert seq == _ROUND9_SEQ


def test_no_math_vote_once_half_the_budget_is_spent(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CONFIRM_MATH", "1")
    tasks, client, seq = _scenario(tmp_path, monkeypatch, spent=50000,
                                   reason_answers={"math_a": SolveResult("solved", "7")})
    scheduler.run(tasks, 100000, client)
    assert not [s for s in seq if s[0] in ("reason", "vote") and s[1] in ("math_a", "math_b", "math_g0")]
    assert _answers(tasks)["math_a"] == "42"


def test_a_crashing_reason_route_falls_back_to_the_program(tmp_path, monkeypatch):
    tasks, client, seq = _scenario(tmp_path, monkeypatch)

    async def boom(task, client):
        seq.append(("reason", task.task_id))
        raise RuntimeError("no code block")
    monkeypatch.setattr(scheduler, "solve_with_reason", boom)
    scheduler.run(tasks, 100000, client)
    assert seq.index(("reason", "math_u")) + 1 == seq.index(("2b", "math_u"))
    assert _answers(tasks)["math_u"] == "5" and _answers(tasks)["math_a"] == "42"


def test_reason_call_tokens_is_what_solve_with_reason_asks_for(tmp_path):
    from harness.llm import ChatResult
    from harness.llm_routes import solve_with_reason
    seen = []

    class Client:
        async def chat(self, model, messages, max_tokens, **extra):
            seen.append((model, max_tokens, extra))
            return ChatResult(text="print(6)", prompt_tokens=10, completion_tokens=10, cost=11)
    (tmp_path / "workspace").mkdir()
    t = Task("math_x", "math", "How many?", 1.0, "exact_int", tmp_path, tmp_path / "workspace")
    res = asyncio.run(solve_with_reason(t, Client()))
    assert res.answer == "6"
    assert seen == [(config.MODEL_REASON, scheduler.REASON_CALL_TOKENS, {"reasoning_effort": "low"})]
    assert scheduler.REASON_CALL_TOKENS > config.MIN_REASONING_CALL_TOKENS


def test_knob(monkeypatch):
    monkeypatch.delenv("HARNESS_REASON_MATH", raising=False)
    assert config.reason_math() is True
    monkeypatch.setenv("HARNESS_REASON_MATH", "0")
    assert config.reason_math() is False
    monkeypatch.delenv("HARNESS_CONFIRM_MATH", raising=False)
    assert config.confirm_math() is False
    monkeypatch.setenv("HARNESS_CONFIRM_MATH", "1")
    assert config.confirm_math() is True
    assert config.CONFIRM_TYPES == ("data_analysis",) and config.CONFIRM_REASON_TYPES == ("math",)


# ---------------------------------------------------------------- confirm_reason

def _math_task(tmp_path):
    return Task("math_x", "math", "A warehouse starts with 10 boxes. How many boxes does it have at the end?", 3.0,
                "exact_int", tmp_path, tmp_path / "workspace")


def _confirm_reason(tmp_path, answer_2a, reason_answer, program_answer, afford=True, program_model=None):
    from harness.confirm import confirm_reason
    from harness.trajectory import CALLS
    calls = []

    async def reason(task, client):
        calls.append("reason")
        if isinstance(reason_answer, Exception):
            raise reason_answer
        return SolveResult("solved", reason_answer) if reason_answer else SolveResult("infeasible", note="IMPOSSIBLE")

    async def program(task, client, model=None, shape_model=None):
        calls.append(("program", model, shape_model))
        log = CALLS.get()
        served = program_model.get(model, model) if isinstance(program_model, dict) else (program_model or model)
        if log is not None:
            log.append({"model": served, "cost": 1})
        a = program_answer.get(model, "5") if isinstance(program_answer, dict) else program_answer
        return SolveResult("solved", a, extra={"stdout": a + "\n"})

    async def go():
        CALLS.set([])
        return await confirm_reason(_math_task(tmp_path), answer_2a, object(), reason, program, lambda: afford)
    return asyncio.run(go()), calls


def test_reason_agrees_and_no_third_vote_is_bought(tmp_path):
    v, calls = _confirm_reason(tmp_path, "1204", "1204", "0")
    assert v.replacement is None and v.fast_agrees and calls == ["reason"]


def test_reason_fast_and_alt_together_replace_2a(tmp_path):
    v, calls = _confirm_reason(tmp_path, "1184", "1204", "1204")
    assert v.replacement == "1204" and not v.fast_agrees
    assert calls == ["reason", ("program", config.MODEL_FAST, config.MODEL_FAST),
                     ("program", config.MODEL_ALT, config.MODEL_ALT)]


def test_reason_and_fast_alone_never_replace_2a(tmp_path):
    # llm8 math_0076 live: REASON and the FAST program (one set of weights) both read "2 deliveries of 467" as
    # outgoing, 7978 against 2a's right 9846; the ALT program did not side with them
    v, calls = _confirm_reason(tmp_path, "9846", "7978", {config.MODEL_FAST: "7978", config.MODEL_ALT: "9846"})
    assert v.replacement is None and "alt '9846' does not side" in v.note and len(calls) == 3
    v, _ = _confirm_reason(tmp_path, "9846", "7978", {config.MODEL_FAST: "7978", config.MODEL_ALT: "8879"})
    assert v.replacement is None
    # ALT failing fell through to FAST's weights: no other reading
    v, _ = _confirm_reason(tmp_path, "9846", "7978", "7978", program_model={config.MODEL_ALT: config.MODEL_FAST})
    assert v.replacement is None and "not independent" in v.note


def test_reason_alone_never_replaces_2a(tmp_path):
    v, _ = _confirm_reason(tmp_path, "82920", "82908", "82920")   # math_0008 of llm7: the program sides with 2a
    assert v.replacement is None and not v.fast_agrees
    v, _ = _confirm_reason(tmp_path, "82920", "82908", "5")       # three answers
    assert v.replacement is None
    v, calls = _confirm_reason(tmp_path, "82920", "82908", "82908", afford=False)
    assert v.replacement is None and calls == ["reason"]


def test_reason_without_an_answer_keeps_2a(tmp_path):
    v, calls = _confirm_reason(tmp_path, "1184", None, "1204")
    assert v.replacement is None and not v.fast_agrees and calls == ["reason"]
    v, calls = _confirm_reason(tmp_path, "1184", RuntimeError("boom"), "1204")
    assert v.replacement is None and calls == ["reason"]
    v, calls = _confirm_reason(tmp_path, "1184", "12.5", "12.5")   # wrong shape for exact_int
    assert v.replacement is None and calls == ["reason"]


def test_budget_exhausted_in_the_reason_vote_is_not_swallowed(tmp_path):
    from harness.llm import BudgetExhausted
    with pytest.raises(BudgetExhausted):
        _confirm_reason(tmp_path, "1184", BudgetExhausted("HTTP 402"), "1204")


def test_a_third_vote_answered_by_reason_itself_does_not_count(tmp_path):
    # FAST was failing and the program fell through to MODEL_REASON: the same model twice is one vote
    v, _ = _confirm_reason(tmp_path, "1184", "1204", "1204", program_model=config.MODEL_REASON)
    assert v.replacement is None and "third vote answered by" in v.note
    v, _ = _confirm_reason(tmp_path, "1184", "1204", "1204", program_model=config.MODEL_ALT)
    assert v.replacement is None and "not independent" in v.note   # FAST fell through to ALT: ALT voted twice

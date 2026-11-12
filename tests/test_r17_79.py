"""Round 17, task 79: a prose answer needs two agreeing attempts; a multi-hop question never takes a printed literal."""
import asyncio
import json

import pytest

from harness import config, llm_routes as L, scheduler
from harness.budget import TokenLedger
from harness.llm import BudgetExhausted, ChatResult, ModelFailing
from harness.result import SolveResult
from harness.tasks import Task
from harness.trajectory import CALLS, record_call

BIO = {"kb/ann_lee.md": "# Ann Lee\n\nAnn Lee was born in Oslo in 1970. After graduating from Tartu College, Ann moved "
                        "to Dune Robotics. Her mentor is Bo Park.\n",
       "kb/bo_park.md": "# Bo Park\n\nBo Park was born in Lyon in 1984. Bo works at Dune Robotics as an accountant.\n",
       "kb/dune_robotics.md": "# Dune Robotics\n\nDune Robotics was set up in 1999 and has its head office in Graz.\n"}
FIELDS = {"kb/person_ann_lee.md": "# Ann Lee\n\nBorn: 1970\nEmployer: Dune Robotics\n",
          "kb/person_bo_park.md": "# Bo Park\n\nBorn: 1984\nEmployer: Dune Robotics\n",
          "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Graz\nFounded: 1999\n"}
LEAD = "The directory kb/ contains a small knowledge base of markdown files written as short biographies. "
Q = LEAD + "Which college did Ann Lee graduate from?"
HOP = LEAD + "In which year was the employer of the mentor of Ann Lee founded?"

TARTU = "print('Tartu College')"
LYON = "print('Lyon')"
GRAZ = "print('Graz')"
CRASH = "print(pages['Ann Lee'].split('nothing like this')[1])"
FOUNDED = ("import re\nm = re.search(r'mentor is ([A-Z]\\w+ [A-Z]\\w+)', pages['Ann Lee']).group(1)\n"
           "print(re.search(r'set up in (\\d{4})', pages['Dune Robotics']).group(1))")


class Fake:
    """replies[model]: the replies of that model in order; a ModelFailing reply makes chat_or_alt go on to the
    next model. Every answered call is recorded as the real client does."""

    def __init__(self, replies):
        self.replies = {m: list(r) for m, r in replies.items()}
        self.models = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.models.append(model)
        reply = self.replies[model].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        record_call(model, 10, 5, 6)
        return ChatResult(text=reply, prompt_tokens=10, completion_tokens=5, cost=6)


def _task(tmp_path, files=BIO, q=Q, check="exact_str", tid="search_1", reward=2.0):
    ws = tmp_path / tid / "workspace"
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return Task(tid, "search", q, reward, check, tmp_path / tid, ws)


def _agree(task, replies, may_call=lambda reason: True):
    client = Fake(replies)

    async def go():
        CALLS.set([])
        first = await L.solve_with_program(task, client)
        return await L.prose_agreement(task, client, first, may_call)
    return asyncio.run(go()), client.models


F, A, R = config.MODEL_FAST, config.MODEL_ALT, config.MODEL_REASON


# ---------------------------------------------------------------- the agreement rule

def test_agreeing_attempts_take_two_calls(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [TARTU], A: [TARTU]})
    assert res.status == "solved" and res.answer == "Tartu College"
    assert models == [F, A] and "fast and alt" in res.note and not (res.extra or {}).get("reasoned")


def test_disagreement_asks_reason_and_the_majority_with_fast_wins(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [TARTU], A: [LYON], R: [TARTU]})
    assert res.answer == "Tartu College" and models == [F, A, R]
    assert "fast and reason agree" in res.note and res.extra["reasoned"]


def test_disagreement_the_majority_with_alt_wins(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [LYON], A: [TARTU], R: [TARTU]})
    assert res.answer == "Tartu College" and models == [F, A, R] and "alt and reason agree" in res.note


def test_three_different_answers_take_reasons(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [LYON], A: [GRAZ], R: [TARTU]})
    assert res.answer == "Tartu College" and models == [F, A, R] and "no majority, reason's answer" in res.note


def test_answers_agree_as_the_soft_check_reads_them(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [TARTU], A: ["print('tartu   college')"]})
    assert res.answer == "Tartu College" and models == [F, A]


def test_a_failed_program_is_no_vote(tmp_path):
    # FAST's program crashes twice (exit code 1): ALT and REASON answer, they agree
    res, models = _agree(_task(tmp_path), {F: [CRASH, CRASH], A: [GRAZ], R: [GRAZ]})
    assert res.status == "solved" and res.answer == "Graz" and models == [F, F, A, R]


def test_a_failed_program_and_two_different_answers_take_reasons(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [CRASH, CRASH], A: [GRAZ], R: [TARTU]})
    assert res.answer == "Tartu College" and models == [F, F, A, R]


def test_reason_failing_keeps_the_first_answer(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [TARTU], A: [LYON], R: [CRASH]})
    assert res.answer == "Tartu College" and models == [F, A, R] and "first kept" in res.note


def test_every_attempt_failing_gives_no_answer(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [CRASH, CRASH], A: [CRASH, CRASH], R: [CRASH]})
    assert res.status != "solved" and models == [F, F, A, A, R]


def test_alt_answered_by_fasts_weights_is_no_vote(tmp_path):
    # ALT keeps failing: chat_or_alt hands its call to FAST, whose answer cannot confirm FAST
    res, models = _agree(_task(tmp_path), {F: [TARTU, TARTU], A: [ModelFailing("down")], R: [LYON]})
    assert models == [F, A, F, R] and "no vote" in res.note
    assert res.answer == "Lyon" and "no majority, reason's answer" in res.note


def test_impossible_is_returned_as_it_is(tmp_path):
    res, models = _agree(_task(tmp_path), {F: ["print('IMPOSSIBLE')"]})
    assert res.status == "infeasible" and models == [F]


# ---------------------------------------------------------------- the budget gate

def test_no_budget_for_a_second_attempt_keeps_the_first(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [TARTU]}, may_call=lambda reason: False)
    assert res.answer == "Tartu College" and models == [F] and "no budget for a second attempt" in res.note


def test_no_budget_for_reason_keeps_the_first(tmp_path):
    asked = []

    def gate(reason):
        asked.append(reason)
        return not reason
    res, models = _agree(_task(tmp_path), {F: [TARTU], A: [LYON]}, may_call=gate)
    assert res.answer == "Tartu College" and models == [F, A] and asked == [False, True] and "first kept" in res.note


def test_no_budget_for_reason_and_no_first_answer_takes_alts(tmp_path):
    res, models = _agree(_task(tmp_path), {F: [CRASH, CRASH], A: [GRAZ]}, may_call=lambda reason: not reason)
    assert res.answer == "Graz" and models == [F, F, A]


def test_budget_exhausted_is_not_swallowed(tmp_path):
    with pytest.raises(BudgetExhausted):
        _agree(_task(tmp_path), {F: [TARTU], A: [BudgetExhausted()]})


def test_prose_more_gate(tmp_path):
    import time
    task = _task(tmp_path)
    ledger = TokenLedger(100000, 0)
    t0 = time.monotonic()
    assert scheduler._prose_more(ledger, task, t0, 3600, False) and scheduler._prose_more(ledger, task, t0, 3600, True)
    ledger.spent = scheduler.REASON_PASS_SPEND_LIMIT * ledger.budget
    assert not scheduler._prose_more(ledger, task, t0, 3600, False)
    ledger.spent = 0
    assert not scheduler._prose_more(ledger, task, t0 - 3600, 3600, False)   # new units have stopped
    small = TokenLedger(2000, 0)
    assert scheduler._prose_more(small, task, t0, 3600, False) and not scheduler._prose_more(small, task, t0, 3600, True)


# ---------------------------------------------------------------- the multi-hop literal rule

@pytest.mark.parametrize("q, hop", [
    ("In which year was the employer of the mentor of Dmitri Moreau founded?", True),
    ("Where is the head office of the employer of Ann Lee's mentor?", True),
    ("In which year was Ann Lee's mentor's company founded?", True),
    ("In which year was the mentor of Wim Kowalski born?", False),
    ("In which city is the headquarters of the company where Rosa Kowalski works?", False),
    ("What is the name of the spouse of Quinn Olsen?", False),
    ("In which year was Nikolai Haddad born?", False),
])
def test_asks_multihop(q, hop):
    assert L._asks_multihop(q) is hop


def test_a_multihop_literal_is_never_taken(tmp_path):
    # 1999 stands on a page, but the chain was followed in the head: one more try for a program over pages
    task = _task(tmp_path, q=HOP)
    client = Fake({F: ["print('1999')", FOUNDED]})
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.status == "solved" and res.answer == "1999" and client.models == [F, F]


def test_a_one_hop_literal_on_a_page_is_still_taken(tmp_path):
    client = Fake({F: [TARTU]})
    res = asyncio.run(L.solve_with_program(_task(tmp_path), client))
    assert res.answer == "Tartu College" and client.models == [F]


def test_the_literal_rule_stays_with_agreement_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "0")
    client = Fake({F: ["print('1999')", FOUNDED]})
    res = asyncio.run(L.solve_with_program(_task(tmp_path, q=HOP), client))
    assert res.answer == "1999" and client.models == [F, F]


def test_knob(monkeypatch):
    monkeypatch.delenv("HARNESS_PROSE_AGREE", raising=False)
    assert config.prose_agree() is config.PROSE_AGREE is True
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "0")
    assert config.prose_agree() is False
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "1")
    assert config.prose_agree() is True


# ---------------------------------------------------------------- the scheduler

def unparsed_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("unparsed", note="forced")))


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HARNESS_BATCH_2A", "0")
    monkeypatch.setattr(scheduler, "_phase1_target", unparsed_target)

    async def no_group(ttype, template, members, client):
        return {}
    monkeypatch.setattr(scheduler, "solve_group", no_group)
    from harness import templates
    monkeypatch.setattr(templates, "group", lambda tasks_: {})

    class Client:
        ledger = TokenLedger(100000, 0)
    return Client()


def _make(tasks_dir, tid, ttype, files, instruction, check="exact_str"):
    d = tasks_dir / tid
    for rel, body in files.items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"task_id": tid, "task_type": ttype, "instruction": instruction,
                                             "reward": 1.0, "check": check}), encoding="utf-8")


def _answers(tasks_dir):
    return {d.name: (d / "answer.txt").read_text(encoding="utf-8").strip()
            for d in sorted(tasks_dir.iterdir()) if (d / "answer.txt").exists()}


def test_only_prose_tasks_get_agreement(tmp_path, monkeypatch, run_env):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, Q)
    _make(tasks, "search_f", "search", FIELDS, LEAD + "Where is the headquarters of Dune Robotics?")
    _make(tasks, "math_m", "math", {}, "Riddle?", check="exact_int")

    async def prog(task, client):
        record_call(F, 10, 5, 6)
        return SolveResult("solved", answer="7" if task.task_type == "math" else "Graz", note="model program")
    seen, on_disk = [], {}

    async def agree(task, client, first, may_call):
        seen.append(task.task_id)
        on_disk.update(_answers(tasks))   # FAST's answer is written before the extra attempts
        assert may_call(False) and may_call(True)
        record_call(A, 10, 5, 6)
        return SolveResult("solved", answer="Tartu College", note="agreed", extra={"reasoned": True})
    reasons = []

    async def reason(task, client):
        reasons.append(task.task_id)
        return SolveResult("unparsed", note="no")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    monkeypatch.setattr(scheduler, "solve_with_reason", reason)
    scheduler.run(tasks, 100000, run_env)
    assert seen == ["search_p"] and on_disk.get("search_p") == "Graz"
    assert _answers(tasks) == {"search_f": "Graz", "search_p": "Tartu College", "math_m": "7"}
    assert "search_p" not in reasons
    rows = [json.loads(x) for x in (tmp_path / "logs" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    prose_rows = [r for r in rows if r.get("task_id") == "search_p" and r.get("phase") == 2]
    assert sum(r["cost"] for r in prose_rows) == 12 and prose_rows[-1]["note"] == "agreed"


def test_agreement_off_is_round_16(tmp_path, monkeypatch, run_env):
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "0")
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, Q)

    async def prog(task, client):
        return SolveResult("solved", answer="Graz", note="model program")

    async def agree(task, client, first, may_call):
        raise AssertionError("no agreement with the knob off")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    scheduler.run(tasks, 100000, run_env)
    assert _answers(tasks) == {"search_p": "Graz"}


def test_a_crashing_agreement_keeps_fasts_answer(tmp_path, monkeypatch, run_env):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, Q)

    async def prog(task, client):
        return SolveResult("solved", answer="Graz", note="model program")

    async def agree(task, client, first, may_call):
        raise RuntimeError("boom")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    counts = scheduler.run(tasks, 100000, run_env)
    assert _answers(tasks) == {"search_p": "Graz"} and counts["solved"] == 1

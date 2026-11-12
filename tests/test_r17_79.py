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


def _agree(task, replies, may_call=lambda reason: True, keep=None):
    client = Fake(replies)

    async def go():
        CALLS.set([])
        first = await L.solve_with_program(task, client)
        return await L.prose_agreement(task, client, first, may_call, keep=keep)
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


def test_impossible_twice_is_infeasible_without_reason(tmp_path):
    res, models = _agree(_task(tmp_path), {F: ["print('IMPOSSIBLE')"], A: ["print('IMPOSSIBLE')"]})
    assert res.status == "infeasible" and models == [F, A] and "fast and alt IMPOSSIBLE" in res.note


# ---------------------------------------------------------------- every way FAST ends without an answer goes on

KB_MISSING = "print(pages['Dune Robotics Inc'][:4])"   # a wrong page title: the preamble raises KbMissing


def test_r17b_search_8104_kb_missing_goes_on_to_alt_and_reason(tmp_path):
    # ea064d3 in Docker: FAST printed a literal (rejected on a chain), then a program that looked up a page that
    # does not exist -> "kb page or field not found" after two FAST calls, and no other attempt
    task = _hop(tmp_path)
    client = Fake({F: ["print(1975)", KB_MISSING]})
    first = asyncio.run(L.solve_with_program(task, client))
    assert first.status == "unparsed" and first.note == "kb page or field not found" and client.models == [F, F]
    res, models = _agree(task, {F: ["print(1975)", KB_MISSING], A: [COMP], R: [COMP]})
    assert models == [F, F, A, R] and res.status == "solved" and res.answer == "1999"
    assert "alt and reason agree" in res.note


@pytest.mark.parametrize("fast", [
    ["print('IMPOSSIBLE')"],                 # IMPOSSIBLE
    [KB_MISSING],                            # a page not found
    ["print()", "print()"],                  # empty output, twice
    [CRASH, CRASH],                          # exit code 1, twice
])
def test_every_failed_first_attempt_goes_on(tmp_path, fast):
    res, models = _agree(_hop(tmp_path), {F: fast, A: [COMP], R: [COMP]})
    assert models == [F] * len(fast) + [A, R]
    assert res.status == "solved" and res.answer == "1999"


def test_impossible_then_nothing_takes_reasons_verdict(tmp_path):
    res, models = _agree(_task(tmp_path), {F: ["print('IMPOSSIBLE')"], A: [CRASH, CRASH], R: ["print('IMPOSSIBLE')"]})
    assert models == [F, A, A, R] and res.status == "infeasible" and res.extra.get("reasoned")


def test_the_proxy_and_the_budget_cut_end_it(tmp_path):
    task = _task(tmp_path)
    for first in (SolveResult("unparsed", note="llm unavailable: down"),
                  SolveResult("unparsed", note="program cut by the budget limit")):
        res = asyncio.run(L.prose_agreement(task, Fake({}), first, lambda reason: True))
        assert res is first


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
    # less than a REASON call's timeout before new units stop: ALT may still start, REASON may not
    short = config.stop_new_before_deadline_s() + config.llm_timeout() / 2
    assert scheduler._prose_more(ledger, task, t0, short, False) and not scheduler._prose_more(ledger, task, t0, short, True)


# ---------------------------------------------------------------- fix round 1 (review of task 79)

COMP = "import re\nprint(re.search(r'set up in (\\d+)', pages['Dune Robotics']).group(1))"


def _hop(tmp_path):
    return _task(tmp_path, q=HOP, check="exact_int")


@pytest.mark.parametrize("fast, alt, reason", [
    ([COMP], [CRASH, CRASH], ["1975"]),                  # REASON answers a lone number
    ([COMP], [CRASH, CRASH], ["print(1975)"]),           # REASON prints a constant
    ([COMP, COMP], [ModelFailing("down")], ["print(1975)"]),   # ALT served by FAST's weights: no vote either
    ([CRASH, CRASH], [COMP], ["1975"]),                  # FAST failed, ALT computed it
])
def test_a_reason_constant_on_a_chain_is_no_vote(tmp_path, fast, alt, reason):
    res, _ = _agree(_hop(tmp_path), {F: fast, A: alt, R: reason})
    assert res.status == "solved" and res.answer == "1999" and "no vote" in res.note


def test_a_reason_constant_on_a_count_is_no_vote(tmp_path):
    q = LEAD + "How many people work at Dune Robotics?"
    count = "print(sum('works at Dune Robotics' in t for t in pages.values()))"
    res, _ = _agree(_task(tmp_path, q=q, check="exact_int"), {F: [count], A: [CRASH, CRASH], R: ["7"]})
    assert res.answer == "1" and "no majority, first kept" in res.note


def test_a_computed_reason_answer_on_a_chain_still_votes(tmp_path):
    res, _ = _agree(_hop(tmp_path), {F: [CRASH, CRASH], A: ["print(1975 + 1)"], R: [COMP]})
    assert res.answer == "1999" and "no majority, reason's answer" in res.note


def test_reason_marks_its_constants(tmp_path):
    def reason(reply):
        return asyncio.run(L.solve_with_reason(_task(tmp_path, q=Q), Fake({R: [reply]})))
    assert reason("print('Tartu College')").extra == {"literal": True}
    assert reason("1975").extra == {"literal": True}
    computed = reason("print(pages['Ann Lee'].split('graduating from ')[1].split(',')[0])")
    assert computed.answer == "Tartu College" and computed.extra is None


def test_alts_answer_is_kept_before_the_reason_attempt(tmp_path):
    kept, seen_at_reason = [], []

    class Watch(Fake):
        async def chat(self, model, messages, max_tokens, **extra):
            if model == R:
                seen_at_reason.append([r.answer for r in kept])
            return await super().chat(model, messages, max_tokens, **extra)

    task = _task(tmp_path)
    client = Watch({F: [CRASH, CRASH], A: [GRAZ], R: [TARTU]})

    async def go():
        CALLS.set([])
        first = await L.solve_with_program(task, client)
        return await L.prose_agreement(task, client, first, lambda reason: True, keep=kept.append)
    res = asyncio.run(go())
    assert seen_at_reason == [["Graz"]] and res.answer == "Tartu College"


def test_fasts_answer_is_not_kept_twice(tmp_path):
    kept = []
    _agree(_task(tmp_path), {F: [TARTU], A: [LYON], R: [GRAZ]}, keep=kept.append)
    assert kept == []   # the scheduler wrote FAST's answer before the agreement


@pytest.mark.parametrize("fast, alt", [
    ("print('Arktika Labs')", "print('Arktika Labs.')"),
    ("print(\"O'Brien Labs\")", "print('O\\u2019Brien Labs')"),
    ("print('Tartu College')", "print('  tartu  college ')"),
])
def test_spellings_the_soft_check_reads_alike_agree(tmp_path, fast, alt):
    res, models = _agree(_task(tmp_path, files={**BIO, "kb/x.md": "# X\n\nArktika Labs. O'Brien Labs. O’Brien Labs.\n"}),
                         {F: [fast], A: [alt]})
    assert models == [F, A] and "fast and alt" in res.note


def test_numbers_agree_as_numbers(tmp_path):
    as_float = "import re\nprint(float(re.search(r'set up in (\\d+)', pages['Dune Robotics']).group(1)))"
    res, models = _agree(_hop(tmp_path), {F: [COMP], A: [as_float]})
    assert res.answer == "1999" and models == [F, A]


def test_different_names_still_disagree():
    assert not L._same_vote("Mina Garcia", "Elena Olsen", "exact_str")
    assert not L._same_vote("18", "19", "exact_int") and L._same_vote("18", "18.0", "exact_int")
    assert not L._same_vote(".", "!", "exact_str")


# ---------------------------------------------------------------- fix round 2 (re-review 1 of task 79)

def test_a_reason_constant_left_alone_is_a_guess_not_an_answer(tmp_path):
    # FAST and ALT crash, REASON prints a constant on a chain: no vote, but a guess beats an empty slot
    res, models = _agree(_hop(tmp_path), {F: [CRASH, CRASH], A: [CRASH, CRASH], R: ["print(1999)"]})
    assert models == [F, F, A, A, R]
    assert res.status == "unparsed" and res.answer is None and res.extra == {"best_guess": "1999"}
    assert "kept as a guess" in res.note and not (res.extra or {}).get("reasoned")


def test_a_reason_constant_never_beats_an_answer(tmp_path):
    res, _ = _agree(_hop(tmp_path), {F: [CRASH, CRASH], A: [COMP], R: ["print(1975)"]})
    assert res.status == "solved" and res.answer == "1999"


@pytest.mark.parametrize("fast, alt, written", [
    ("print('Arktika Labs.')", "print('Arktika Labs')", "Arktika Labs"),
    ("print('Arktika Labs')", "print('Arktika Labs.')", "Arktika Labs"),
    ("print('O\\u2019Brien Labs')", "print(\"O'Brien Labs\")", "O'Brien Labs"),
    ("print('Arktika Labs.')", "print('Arktika Labs.')", "Arktika Labs."),   # no clean spelling: FAST's
])
def test_agreeing_spellings_write_the_clean_one(tmp_path, fast, alt, written):
    files = {**BIO, "kb/x.md": "# X\n\nArktika Labs. O'Brien Labs. O’Brien Labs.\n"}
    res, models = _agree(_task(tmp_path, files=files), {F: [fast], A: [alt]})
    assert models == [F, A] and res.answer == written


@pytest.mark.parametrize("fast, alt, written", [
    ("print('John Smith Jr.')", "print('John Smith Jr')", "John Smith Jr."),
    ("print('John Smith Jr')", "print('John Smith Jr.')", "John Smith Jr."),
    ("print('U.S. Steel Co.')", "print('U.S. Steel Co')", "U.S. Steel Co."),
    ("print('Arktika Labs.')", "print('Arktika Labs')", "Arktika Labs"),
])
def test_an_abbreviation_keeps_its_dot(tmp_path, fast, alt, written):
    files = {**BIO, "kb/x.md": "# X\n\nJohn Smith Jr. leads U.S. Steel Co. (John Smith Jr, U.S. Steel Co). Arktika Labs.\n"}
    res, models = _agree(_task(tmp_path, files=files), {F: [fast], A: [alt]})
    assert models == [F, A] and res.answer == written


def test_agreeing_numbers_write_the_plain_integer():
    # (on a year question the shape check already refuses 1999.0; other number questions can print it)
    a, b = SolveResult("solved", answer="1999.0"), SolveResult("solved", answer="1999")
    assert L._same_vote("1999.0", "1999", "exact_str") and L._cleanest(a, b) is b
    c, d = SolveResult("solved", answer="12.50"), SolveResult("solved", answer="12.5")
    assert L._cleanest(c, d) is c   # places may be the ones the task asks for: FAST's stand


def test_reason_agreeing_after_normalisation_writes_the_clean_one(tmp_path):
    files = {**BIO, "kb/x.md": "# X\n\nArktika Labs. Bluefin Systems.\n"}
    res, _ = _agree(_task(tmp_path, files=files), {F: ["print('Arktika Labs.')"], A: ["print('Bluefin Systems')"],
                                                   R: ["print('Arktika Labs')"]})
    assert res.answer == "Arktika Labs" and "fast and reason agree" in res.note


@pytest.mark.parametrize("answer, clean", [
    ("Arktika Labs", True), ("Arktika Labs.", False), ("O’Brien", False), ("O'Brien", True),
    ("Tartu  College", False), ("'Graz'", False), ("1999", True), ("1999.0", False), ("12.50", True), ("-3", True),
    ("John Smith Jr.", True), ("John Smith Jr", False), ("U.S. Steel Co.", True), ("U.S. Steel Co", False),
    ("Acme Inc.", True), ("Dune Ltd.", True), ("Baker St.", True), ("U.S.", True), ("B", True), ("B.", False),
    ("No", True), ("No.", False), ("Arktika Labs.", False), ("Jr..", False),
])
def test_is_clean(answer, clean):
    assert L._is_clean(answer) is clean


def test_the_guess_goes_to_disk_and_the_final_pass_still_asks(tmp_path, monkeypatch, run_env):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, HOP, check="exact_int")

    async def prog(task, client):
        return SolveResult("unparsed", note="program exit code 1")

    async def agree(task, client, first, may_call, keep=None):
        return SolveResult("unparsed", note="no answer; reason's constant kept as a guess", extra={"best_guess": "1975"})
    on_disk, asked = [], []

    async def reason(task, client):
        on_disk.append(_answers(tasks).get("search_p"))
        asked.append(task.task_id)
        return SolveResult("solved", answer="1999", note="model reason program")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    monkeypatch.setattr(scheduler, "solve_with_reason", reason)
    scheduler.run(tasks, 100000, run_env)
    assert on_disk == ["1975"] and asked == ["search_p"] and _answers(tasks) == {"search_p": "1999"}


# ---------------------------------------------------------------- the multi-hop literal rule

@pytest.mark.parametrize("q, hop", [
    ("In which year was the employer of the mentor of Dmitri Moreau founded?", True),
    ("Where is the head office of the employer of Ann Lee's mentor?", True),
    ("In which year was Ann Lee's mentor's company founded?", True),
    ("In which year was the mentor of Wim Kowalski born?", False),
    ("In which city is the headquarters of the company where Rosa Kowalski works?", False),
    ("What is the name of the spouse of Quinn Olsen?", False),
    ("What is the name of Quinn Olsen's spouse?", False),
    ("What is the name of Dmitri Fischer's employer?", False),
    ("What was the name of Leo Ito's first company?", False),
    ("What is the title of Leo Ito's first novel?", False),
    ("In which year was the company of Ann Lee's mentor founded?", True),
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

    async def agree(task, client, first, may_call, keep=None):
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

    async def agree(task, client, first, may_call, keep=None):
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

    async def agree(task, client, first, may_call, keep=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    counts = scheduler.run(tasks, 100000, run_env)
    assert _answers(tasks) == {"search_p": "Graz"} and counts["solved"] == 1


def test_a_raising_fast_route_is_a_failed_first_attempt(tmp_path, monkeypatch, run_env):
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, Q)

    async def prog(task, client):
        raise RuntimeError("boom")
    firsts = []

    async def agree(task, client, first, may_call, keep=None):
        firsts.append(first)
        return SolveResult("solved", answer="Tartu College", note="alt and reason agree")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    scheduler.run(tasks, 100000, run_env)
    assert [f.status for f in firsts] == ["unparsed"] and firsts[0].note.startswith("program route failed")
    assert L._program_failed(firsts[0]) and _answers(tasks) == {"search_p": "Tartu College"}


def test_a_raising_route_off_the_prose_path_is_as_before(tmp_path, monkeypatch, run_env):
    monkeypatch.setenv("HARNESS_PROSE_AGREE", "0")
    tasks = tmp_path / "tasks"
    _make(tasks, "search_p", "search", BIO, Q)

    async def prog(task, client):
        raise RuntimeError("boom")

    async def agree(task, client, first, may_call, keep=None):
        raise AssertionError("no agreement with the knob off")
    monkeypatch.setattr(scheduler, "solve_with_program", prog)
    monkeypatch.setattr(scheduler, "prose_agreement", agree)
    scheduler.run(tasks, 100000, run_env)
    rows = [json.loads(x) for x in (tmp_path / "logs" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r.get("task_id") == "search_p" and r.get("status") == "error" for r in rows)

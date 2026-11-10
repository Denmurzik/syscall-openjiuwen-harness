"""Independent model check of green single coding fixes (HARNESS_VERIFY_CODING)."""
import asyncio
import json

import pytest

from harness import config, scheduler
from harness.arbitrate import verify_fix
from harness.budget import TokenLedger
from harness.config import ARBITRATE_MODELS, MODEL_ALT, MODEL_FAST
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.result import SolveResult
from harness.solvers.coding_solver import solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task

TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
         "    def test_even_sq(self):\n        self.assertEqual(mylib.even_sq([1, 3, 4]), [16])\n\n"
         "    def test_other(self):\n        self.assertEqual(mylib.other(1), 1)\n")
BUGGY = "def even_sq(xs):\n    return [x * 2 for x in xs if x % 2 == 0]\n"
FITTED = "def even_sq(xs):\n    return [x * 4 for x in xs if x % 2 == 0]\n"   # green, wrong on hidden inputs
RIGHT = "def even_sq(xs):\n    return [x ** 2 for x in xs if x % 2 == 0]\n"
SAME = "def even_sq(xs):\n    return [4 * x for x in xs if x % 2 == 0]\n"     # other text, same behaviour
RED = "def even_sq(xs):\n    return []\n"
OTHER = "\n\ndef other(x):\n    return x\n"
INPUTS = [([1, 3, 4],), ([3, 6],), ([2, 2],)]


def _reply(src):
    return "FUNC even_sq\n" + src


def _extra():
    return {"verify": [{"func": "even_sq", "buggy": BUGGY.strip(), "inputs": INPUTS,
                        "tests": "self.assertEqual(mylib.even_sq([1, 3, 4]), [16])"}],
            "visible_asserts": ["self.assertEqual(mylib.even_sq([1, 3, 4]), [16])"]}


def _task(tmp_path, lib=FITTED + OTHER, tests=TESTS):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


class Fake:
    """`writer` is the reply of the MODEL_FAST call without a regex; `picks` {model: reply} answer arbitration."""

    def __init__(self, writer, picks=None):
        self.writer, self.picks, self.calls = writer, picks or {}, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra": extra})
        r = self.picks[model] if extra.get("extra_body") else self.writer
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=1, cost=2)


def run(c):
    return asyncio.run(c)


def _lib(task):
    return (task.workspace / "mylib.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- the knob

def test_knob_default_and_values(monkeypatch):
    monkeypatch.delenv("HARNESS_VERIFY_CODING", raising=False)
    assert config.verify_coding() is config.VERIFY_CODING
    for v, want in (("1", True), ("on", True), ("0", False), ("off", False)):
        monkeypatch.setenv("HARNESS_VERIFY_CODING", v)
        assert config.verify_coding() is want
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "maybe")
    assert config.verify_coding() is config.VERIFY_CODING


# ---------------------------------------------------------------- the hint from phase 1

SINGLE_LIB = "def has(xs, v):\n    return v not in xs\n"
SINGLE_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
                "    def test_has(self):\n        self.assertEqual(mylib.has([1, 2], 2), True)\n\n"
                "    def test_has_more(self):\n        self.assertEqual(mylib.has([3], 3), True)\n\n"
                "    def test_unrelated(self):\n        self.assertEqual(1, 1)\n")


def test_green_single_fix_is_left_for_the_check_when_the_knob_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    task = _task(tmp_path, SINGLE_LIB, SINGLE_TESTS)
    res = solve(task)
    assert res.status == "solved" and res.answer == "DONE" and res.tentative
    assert "v in xs" in _lib(task)    # the fix is on disk whatever happens next
    it = res.extra["verify"][0]
    assert it["func"] == "has" and it["buggy"] == SINGLE_LIB.strip()
    assert it["inputs"][0] == ([1, 2], 2) and len(it["inputs"]) > 1
    assert "mylib.has([1, 2], 2)" in it["tests"] and "mylib.has([3], 3)" in it["tests"]
    assert "test_unrelated" not in it["tests"]
    assert res.extra["visible_asserts"] == ["self.assertEqual(mylib.has([1, 2], 2), True)"]
    assert "v in xs" not in json.dumps(res.extra, default=str)   # our fix is not in what the model is shown


def test_knob_off_changes_nothing_in_phase_1(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "0")
    res = solve(_task(tmp_path, SINGLE_LIB, SINGLE_TESTS))
    assert res.status == "solved" and not res.tentative and res.extra is None


def test_ambiguous_fix_goes_to_arbitration_not_to_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = "def count_evens(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_count(self):\n        self.assertEqual(mylib.count_evens([1, 2, 4, 6, 7]), 3)\n")
    res = solve(_task(tmp_path, lib, tests))
    assert res.extra["ambiguous"] and "verify" not in res.extra


def test_partial_repair_is_not_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = "def has(xs, v):\n    return v not in xs\n\n\ndef weird(x):\n    return x\n"
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_has(self):\n        self.assertEqual(mylib.has([1, 2], 2), True)\n\n"
             "    def test_weird(self):\n        self.assertEqual(mylib.weird(1), 77123)\n")
    res = solve(_task(tmp_path, lib, tests))
    assert res.tentative and "partial" in res.note and not res.extra


def test_contradictory_tests_are_not_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = "def f(x):\n    return x + 1\n"
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_a(self):\n        self.assertEqual(mylib.f(1), 5)\n\n"
             "    def test_b(self):\n        self.assertEqual(mylib.f(1), 6)\n")
    res = solve(_task(tmp_path, lib, tests))
    assert res.status == "infeasible" and not res.extra


def test_fix_without_a_literal_visible_call_is_final(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    tests = ("import unittest\nimport mylib\n\nXS = [1, 2]\n\n\nclass T(unittest.TestCase):\n"
             "    def test_has(self):\n        self.assertEqual(mylib.has(XS, 2), True)\n")
    res = solve(_task(tmp_path, SINGLE_LIB, tests))
    assert res.status == "solved" and not res.tentative and res.extra is None


def test_method_fix_is_final(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")
    lib = "class C:\n    def has(self, xs, v):\n        return v not in xs\n"
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_has(self):\n        self.assertEqual(mylib.C().has([1, 2], 2), True)\n")
    res = solve(_task(tmp_path, lib, tests))
    assert res.status == "solved" and not res.tentative and res.extra is None


# ---------------------------------------------------------------- the check

def test_same_behaviour_keeps_the_fix_with_one_call(tmp_path):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    c = Fake(_reply(SAME))
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and res.answer == "DONE" and not res.tentative
    assert len(c.calls) == 1 and c.calls[0]["model"] == MODEL_FAST
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_model_sees_buggy_function_and_tests_but_not_our_fix(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(FITTED))
    run(verify_fix(task, _extra(), c))
    text = str(c.calls[0]["messages"])
    assert "x * 2" in text and "even_sq([1, 3, 4])" in text and "x * 4" not in text


def test_identical_text_needs_no_probe(tmp_path, monkeypatch):
    task = _task(tmp_path)

    def boom(*a, **k):
        raise AssertionError("probed")
    monkeypatch.setattr("harness.arbitrate.behaviour", boom)
    res = run(verify_fix(task, _extra(), Fake(_reply(FITTED))))
    assert res.status == "solved"


def test_different_and_both_models_agree_replaces_the_fix(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), {MODEL_FAST: "2", MODEL_ALT: "2"})
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and not res.tentative
    assert "x ** 2" in _lib(task) and "def other(x)" in _lib(task)
    assert run_tests(task.workspace).ok
    assert sorted(x["model"] for x in c.calls) == sorted([MODEL_FAST, MODEL_FAST, MODEL_ALT])


@pytest.mark.parametrize("fast,alt", [("1", "2"), ("2", "1"), ("1", "1")])
def test_anything_but_agreement_on_the_other_keeps_the_fix(tmp_path, fast, alt):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    res = run(verify_fix(task, _extra(), Fake(_reply(RIGHT), {MODEL_FAST: fast, MODEL_ALT: alt})))
    assert res.status == "solved" and "kept" in res.note
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_arbitration_shows_both_behaviours(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), {MODEL_FAST: "1", MODEL_ALT: "1"})
    run(verify_fix(task, _extra(), c))
    text = next(x for x in c.calls if x["extra"].get("extra_body"))["messages"][1]["content"]
    assert "even_sq([3, 6]) -> [24] | [36]" in text and "Option 2:" in text


def test_candidate_red_on_visible_tests_is_not_considered(tmp_path):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    c = Fake(_reply(RED))
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and len(c.calls) == 1
    assert (task.workspace / "mylib.py").read_bytes() == before


@pytest.mark.parametrize("reply", ["no idea", "FUNC even_sq\ndef broken(:\n", "FUNC ghost\ndef ghost():\n    return 1\n", ""])
def test_unusable_reply_keeps_the_fix(tmp_path, reply):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    c = Fake(reply)
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and len(c.calls) == 1
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_reply_cut_by_the_budget_keeps_the_fix(tmp_path):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()

    class Cut(Fake):
        async def chat(self, model, messages, max_tokens, **extra):
            self.calls.append(model)
            return ChatResult(text=_reply(RIGHT), prompt_tokens=10, completion_tokens=1, cost=2, truncated=True)
    c = Cut("")
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and c.calls == [MODEL_FAST]
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_reply_changing_another_function_touches_only_the_checked_one(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT) + "\nFUNC other\ndef other(x):\n    return x\n\n",
             {MODEL_FAST: "2", MODEL_ALT: "2"})
    run(verify_fix(task, _extra(), c))
    assert "x ** 2" in _lib(task) and "return x\n" in _lib(task)


def test_writer_unavailable_is_reported_and_keeps_the_file(tmp_path):
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    res = run(verify_fix(task, _extra(), Fake(LLMUnavailable("HTTP 503"))))
    assert res.status == "unparsed" and res.note.startswith("llm unavailable")
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_arbitration_unavailable_resumes_without_buying_the_candidate_again(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply(RIGHT), {m: LLMUnavailable("HTTP 503") for m in ARBITRATE_MODELS})
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "unparsed" and res.note.startswith("llm unavailable")
    assert res.extra["ambiguous"][0]["candidates"][1].strip() == RIGHT.strip()
    c2 = Fake("must not be asked", {MODEL_FAST: "2", MODEL_ALT: "2"})
    res2 = run(verify_fix(task, res.extra, c2))
    assert res2.status == "solved" and "x ** 2" in _lib(task)
    assert all(x["extra"].get("extra_body") for x in c2.calls)


@pytest.mark.parametrize("at", ["writer", "arbiter"])
def test_budget_exhausted_propagates(tmp_path, at):
    task = _task(tmp_path)
    c = (Fake(BudgetExhausted("x")) if at == "writer"
         else Fake(_reply(RIGHT), {MODEL_FAST: BudgetExhausted("x"), MODEL_ALT: "2"}))
    with pytest.raises(BudgetExhausted):
        run(verify_fix(task, _extra(), c))


def test_two_functions_are_checked_with_one_writer_call(tmp_path):
    lib = FITTED + "\n\ndef neg(x):\n    return -x\n"
    tests = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_a(self):\n        self.assertEqual(mylib.even_sq([1, 3, 4]), [16])\n\n"
             "    def test_b(self):\n        self.assertEqual(mylib.neg(0), 0)\n")
    task = _task(tmp_path, lib, tests)
    extra = _extra()
    extra["verify"].append({"func": "neg", "buggy": "def neg(x):\n    return x", "inputs": [(0,), (3,)],
                            "tests": "self.assertEqual(mylib.neg(0), 0)"})
    c = Fake(_reply(RIGHT) + "\nFUNC neg\ndef neg(x):\n    return 0 - x\n", {MODEL_FAST: "2", MODEL_ALT: "2"})
    res = run(verify_fix(task, extra, c))
    assert res.status == "solved" and "x ** 2" in _lib(task) and "return -x" in _lib(task)
    assert len([x for x in c.calls if not x["extra"]]) == 1


def test_file_gone_keeps_the_answer(tmp_path):
    task = _task(tmp_path)
    (task.workspace / "mylib.py").unlink()
    c = Fake(_reply(RIGHT))
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and not c.calls


# ---------------------------------------------------------------- the scheduler

def verify_target(queue, task, solver_name=None):
    queue.put((task.task_id, SolveResult("solved", "DONE", "unchecked", tentative=True,
                                         extra={"verify": [{"func": "f", "buggy": "def f(): pass", "inputs": [],
                                                            "tests": "t"}], "visible_asserts": []})))


def mixed_verify_target(queue, task, solver_name=None):
    if task.task_id == "c_verify":
        verify_target(queue, task)
    else:
        queue.put((task.task_id, SolveResult("unparsed", note="no")))


def _coding(tasks, tid="c1", reward=1.0):
    d = tasks / tid
    (d / "workspace").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": tid, "task_type": "coding", "instruction": "fix it",
                                             "reward": reward, "check": "unittest"}), encoding="utf-8")
    (d / "workspace" / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text("import mylib\n", encoding="utf-8")
    return d


def test_scheduler_checks_after_2b_and_does_not_send_it_to_2b(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks, "c_verify", 9.0)
    _coding(tasks, "c_open", 1.0)
    order = []

    async def fake_verify(task, extra, client):
        order.append(("verify", task.task_id, extra["verify"][0]["func"]))
        return SolveResult("solved", "DONE", "same behaviour")

    async def fake_fix(task, client):
        order.append(("fix", task.task_id))
        return SolveResult("solved", "DONE", "patched")
    monkeypatch.setattr(scheduler, "_phase1_target", mixed_verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    monkeypatch.setattr(scheduler, "fix_coding", fake_fix)
    counts = scheduler.run(tasks, 100000, object())
    assert order == [("fix", "c_open"), ("verify", "c_verify", "f")]
    assert counts["solved"] == 2


def test_scheduler_without_a_client_keeps_the_written_fix(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    d = _coding(tasks)
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    counts = scheduler.run(tasks, 100000, None)
    assert counts["solved"] == 1 and (d / "answer.txt").read_text() == "DONE\n"


def test_scheduler_unavailable_keeps_the_answer_and_asks_again_with_the_resume(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    d = _coding(tasks)
    seen = []

    async def fake_verify(task, extra, client):
        seen.append(sorted(extra))
        assert (d / "answer.txt").is_file()
        if len(seen) == 1:
            return SolveResult("unparsed", note="llm unavailable: 503", extra={"ambiguous": ["resume"]})
        return SolveResult("solved", "DONE", "kept")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    counts = scheduler.run(tasks, 100000, object())
    assert counts["solved"] == 1
    assert seen == [["verify", "visible_asserts"], ["ambiguous"]]


def test_scheduler_unavailable_twice_stays_solved(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks)
    runs = []

    async def fake_verify(task, extra, client):
        runs.append(1)
        return SolveResult("unparsed", note="llm unavailable: 503")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    counts = scheduler.run(tasks, 100000, object())
    assert counts["solved"] == 1 and len(runs) == 2


def test_scheduler_skips_the_check_it_cannot_afford(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    d = _coding(tasks)
    runs = []

    async def fake_verify(task, extra, client):
        runs.append(1)
        return SolveResult("solved", "DONE", "x")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)

    class C:
        ledger = TokenLedger(100, 1)   # below one writer call's output limit
    counts = scheduler.run(tasks, 100, C())
    assert not runs and counts["solved"] == 1 and (d / "answer.txt").is_file()


def test_scheduler_budget_exhausted_in_the_check_stops_and_keeps_the_answer(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    d = _coding(tasks)

    async def fake_verify(task, extra, client):
        raise BudgetExhausted("x")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    counts = scheduler.run(tasks, 100000, object())
    assert counts["solved"] == 1 and (d / "answer.txt").is_file()


def test_scheduler_records_the_cost_of_the_check_under_its_unit(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks)
    logs = tmp_path / "logs"
    monkeypatch.setenv("HARNESS_LOG_DIR", str(logs))

    async def fake_verify(task, extra, client):
        from harness.trajectory import record_call
        record_call(MODEL_FAST, 10, 5, 6.0)
        return SolveResult("solved", "DONE", "same behaviour")
    monkeypatch.setattr(scheduler, "_phase1_target", verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    scheduler.run(tasks, 100000, object())
    rows = [json.loads(x) for f in logs.rglob("*.jsonl") for x in f.read_text(encoding="utf-8").splitlines()]
    row = next(r for r in rows if r.get("unit") == "verify:c1")
    assert row["cost"] == 6.0 and row["task_id"] == "c1" and row["calls"][0]["model"] == MODEL_FAST


# ---------------------------------------------------------------- fix round 1

P1_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_f(self):\n        self.assertEqual(mylib.f(5), 10)\n")


def test_arbiter_sees_the_only_differing_probe_input(tmp_path):
    """Reviewer's P1: the candidate differs from ours only on the 6th probe input."""
    from harness.solvers.probe import inputs_from_call
    inputs = inputs_from_call((5,))
    last = inputs[5][0]
    task = _task(tmp_path, "def f(x):\n    return x * 2\n", P1_TESTS)
    extra = {"verify": [{"func": "f", "buggy": "def f(x):\n    return x + 2", "inputs": inputs, "tests": "t"}],
             "visible_asserts": ["self.assertEqual(mylib.f(5), 10)"]}
    cand = f"def f(x):\n    if x == {last}:\n        return -999\n    return x * 2\n"
    c = Fake("FUNC f\n" + cand, {MODEL_FAST: "1", MODEL_ALT: "1"})
    run(verify_fix(task, extra, c))
    text = next(x for x in c.calls if x["extra"].get("extra_body"))["messages"][1]["content"]
    rows = text.split("sample calls:\n")[1].split("\n\n")[0].splitlines()
    assert rows[0] == f"f({last}) -> {last * 2} | -999"
    assert len(rows) == 5


def test_arbiter_rows_differing_first_then_the_rest():
    from harness.arbitrate import _prompt
    item = {"func": "g", "buggy": "def g(x): pass", "candidates": ["a", "b"], "chosen": 0,
            "calls": [f"g({i})" for i in range(7)],
            "outputs": [["1", "1", "1", "1", "1", "1", "1"], ["1", "2", "1", "2", "1", "1", "2"]]}
    rows = _prompt([item], []).split("sample calls:\n")[1].splitlines()
    assert rows[:3] == ["g(1) -> 1 | 2", "g(3) -> 1 | 2", "g(6) -> 1 | 2"]
    assert rows[3:] == ["g(0) -> 1 | 1", "g(2) -> 1 | 1"]


def test_arbiter_sees_every_differing_row_even_past_the_limit():
    from harness.arbitrate import _prompt
    item = {"func": "g", "buggy": "def g(x): pass", "candidates": ["a", "b"], "chosen": 0,
            "calls": [f"g({i})" for i in range(7)],
            "outputs": [[str(i) for i in range(7)], [str(-i - 1) for i in range(7)]]}
    rows = _prompt([item], []).split("sample calls:\n")[1].splitlines()
    assert len(rows) == 7


def test_reply_at_the_output_limit_is_not_a_candidate(tmp_path):
    """Reviewer's P2: a reply that hit VERIFY_MAX_TOKENS may be cut and still parse."""
    task = _task(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()

    class AtLimit(Fake):
        async def chat(self, model, messages, max_tokens, **extra):
            self.calls.append(model)
            return ChatResult(text=_reply(RIGHT), prompt_tokens=10, completion_tokens=max_tokens, cost=2)
    c = AtLimit("")
    res = run(verify_fix(task, _extra(), c))
    assert res.status == "solved" and c.calls == [MODEL_FAST] and "limit" in res.note
    assert (task.workspace / "mylib.py").read_bytes() == before


@pytest.mark.parametrize("fast,alt", [("2", "1"), ("2", "2")])
def test_note_carries_each_models_vote(tmp_path, fast, alt):
    task = _task(tmp_path)
    res = run(verify_fix(task, _extra(), Fake(_reply(RIGHT), {MODEL_FAST: fast, MODEL_ALT: alt})))
    assert f"{MODEL_FAST} {fast}" in res.note and f"{MODEL_ALT} {alt}" in res.note
    assert len(res.note) <= 200


def test_note_votes_name_an_unavailable_model(tmp_path):
    task = _task(tmp_path)
    res = run(verify_fix(task, _extra(), Fake(_reply(RIGHT), {MODEL_FAST: "2", MODEL_ALT: LLMUnavailable("404")})))
    assert res.status == "solved" and f"{MODEL_ALT} unavailable" in res.note and f"{MODEL_FAST} 2" in res.note


def test_hint_failure_keeps_a_final_answer(tmp_path, monkeypatch):
    from harness.solvers import coding_solver
    monkeypatch.setenv("HARNESS_VERIFY_CODING", "1")

    def boom(*a):
        raise ValueError("x")
    monkeypatch.setattr(coding_solver, "_verify_hint", boom)
    res = solve(_task(tmp_path, SINGLE_LIB, SINGLE_TESTS))
    assert res.status == "solved" and not res.tentative and res.extra is None


def test_default_is_on(monkeypatch):
    monkeypatch.delenv("HARNESS_VERIFY_CODING", raising=False)
    assert config.verify_coding() is True


def open_then_verify_target(queue, task, solver_name=None):
    if task.task_id == "c_verify":
        verify_target(queue, task)
    else:
        queue.put((task.task_id, SolveResult("unparsed", note="no")))


def test_check_runs_after_the_retry_pass_of_open_tasks(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks, "c_verify", 9.0)
    _coding(tasks, "c_open", 1.0)
    order = []

    async def fake_verify(task, extra, client):
        order.append("verify")
        return SolveResult("solved", "DONE", "kept")

    async def fake_fix(task, client, resume=None):
        order.append("fix")
        if len([o for o in order if o == "fix"]) == 1:
            return SolveResult("unparsed", note="llm unavailable: 503", extra={"ladder": {"answered": []}})
        return SolveResult("solved", "DONE", "patched")
    monkeypatch.setattr(scheduler, "_phase1_target", open_then_verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    monkeypatch.setattr(scheduler, "fix_coding", fake_fix)
    counts = scheduler.run(tasks, 100000, object())
    assert order == ["fix", "fix", "verify"] and counts["solved"] == 2


def test_budget_exhausted_in_the_check_cancels_nothing_before_it(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks, "c_verify", 9.0)
    _coding(tasks, "c_open", 1.0)
    order = []

    async def fake_verify(task, extra, client):
        order.append("verify")
        raise BudgetExhausted("x")

    async def fake_fix(task, client, resume=None):
        order.append("fix")
        if len([o for o in order if o == "fix"]) == 1:
            return SolveResult("unparsed", note="llm unavailable: 503", extra={"ladder": {"answered": []}})
        return SolveResult("solved", "DONE", "patched")
    monkeypatch.setattr(scheduler, "_phase1_target", open_then_verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    monkeypatch.setattr(scheduler, "fix_coding", fake_fix)
    counts = scheduler.run(tasks, 100000, object())
    assert order == ["fix", "fix", "verify"] and counts["solved"] == 2


def test_check_runs_after_the_final_reason_pass(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    _coding(tasks, "c_verify", 9.0)
    d = tasks / "m_open"
    (d / "workspace").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": "m_open", "task_type": "math", "instruction": "x?",
                                             "reward": 9.0, "check": "exact_int"}), encoding="utf-8")
    order = []

    async def fake_verify(task, extra, client):
        order.append("verify")
        return SolveResult("solved", "DONE", "kept")

    async def fake_program(task, client):
        return SolveResult("unparsed", note="no")

    async def fake_reason(task, client):
        order.append("reason")
        return SolveResult("solved", "7", "reason")

    async def no_groups(*a, **k):
        return {}
    monkeypatch.setattr(scheduler, "_phase1_target", open_then_verify_target)
    monkeypatch.setattr(scheduler, "verify_fix", fake_verify)
    monkeypatch.setattr(scheduler, "solve_with_program", fake_program)
    monkeypatch.setattr(scheduler, "solve_with_reason", fake_reason)
    monkeypatch.setattr(scheduler, "solve_group", no_groups)
    monkeypatch.setattr(scheduler, "solve_batch", no_groups)
    scheduler.run(tasks, 100000, object())
    assert order == ["reason", "verify"]


def test_difference_past_the_cut_is_a_differing_row_and_shown():
    """Reviewer's P3: outputs equal in their first 24 characters."""
    from harness.arbitrate import _prompt
    ours, cand = repr(list(range(20))), repr(list(range(19)) + [-1])
    item = {"func": "f", "buggy": "def f(n): pass", "candidates": ["a", "b"], "chosen": 0,
            "calls": [f"f({i})" for i in range(6)] + ["f(20)"],
            "outputs": [["0"] * 6 + [ours], ["0"] * 6 + [cand]]}
    rows = _prompt([item], []).split("sample calls:\n")[1].splitlines()
    assert rows[0].startswith("f(20) -> ")
    left, right = rows[0][len("f(20) -> "):].split(" | ")
    assert left != right and left.endswith("18, 19]") and right.endswith("18, -1]")
    assert left.startswith("…") and len(left) <= 24 and len(right) <= 24


def test_long_outputs_differing_early_are_cut_as_before():
    from harness.arbitrate import _prompt
    a, b = "x" + "a" * 40, "y" + "a" * 40
    item = {"func": "f", "buggy": "def f(n): pass", "candidates": ["a", "b"], "chosen": 0,
            "calls": ["f(1)"], "outputs": [[a], [b]]}
    row = _prompt([item], []).split("sample calls:\n")[1].splitlines()[0]
    assert row == "f(1) -> " + a[:23] + "… | " + b[:23] + "…"

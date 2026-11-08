import asyncio
from pathlib import Path

from harness.arbitrate import arbitrate, parse_choice
from harness.config import ARBITRATE_MODELS, MODEL_ALT, MODEL_FAST
from harness.llm import BudgetExhausted, ChatResult, LLMUnavailable
from harness.result import SolveResult
from harness.solvers.coding_solver import solve
from harness.solvers.testrun import run_tests
from harness.tasks import Task

LIB = "def count_even(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"
TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
         "    def test_count(self):\n        self.assertEqual(mylib.count_even([1, 2, 4, 6, 7]), 3)\n")


class Fake:
    """Replies per model: a list is shared verbatim by every model (each gets its own copy),
    a dict maps model -> list. Both arbitration models must agree for a change."""

    def __init__(self, replies):
        self.replies = ({m: list(v) for m, v in replies.items()} if isinstance(replies, dict)
                        else {m: list(replies) for m in ARBITRATE_MODELS})
        self.calls = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra": extra})
        r = self.replies[model].pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=1, cost=2)


def _solved(tmp_path):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(LIB, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(TESTS, encoding="utf-8")
    task = Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")
    res = solve(task)
    assert res.tentative and res.extra["ambiguous"]
    it = res.extra["ambiguous"][0]
    assert "x % 2 == 0" in it["candidates"][it["chosen"]]   # phase 1 itself picks the meaning of the name
    return task, res.extra


def run(c):
    return asyncio.run(c)


def _target(extra):
    """Option number (1-based) of a candidate other than the phase-1 choice."""
    it = extra["ambiguous"][0]
    return next(i for i in range(len(it["candidates"])) if i != it["chosen"]) + 1


def _written(task, extra, opt):
    """True if mylib's function is now candidate `opt` (1-based)."""
    import ast
    want = ast.unparse(ast.parse(extra["ambiguous"][0]["candidates"][opt - 1]))
    return ast.unparse(ast.parse((task.workspace / "mylib.py").read_text(encoding="utf-8"))) == want


def test_picks_other_option_and_rewrites_mylib(tmp_path):
    task, extra = _solved(tmp_path)
    want = _target(extra)
    c = Fake([str(want)])
    res = run(arbitrate(task, extra, c))
    assert res.status == "solved" and res.answer == "DONE" and not res.tentative
    assert len(c.calls) == 2 and all(x["max_tokens"] <= 8 for x in c.calls)
    assert sorted(x["model"] for x in c.calls) == sorted([MODEL_FAST, MODEL_ALT])
    n = len(extra["ambiguous"][0]["candidates"])
    assert all(x["extra"] == {"extra_body": {"guided_regex": "[1-%d]" % n}} for x in c.calls)
    assert _written(task, extra, want)
    assert run_tests(task.workspace).ok


def test_picks_current_choice_leaves_file_untouched(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    res = run(arbitrate(task, extra, Fake([str(extra["ambiguous"][0]["chosen"] + 1)])))
    assert res.status == "solved" and not res.tentative
    assert res.note == "arbitration kept the choice (models confirmed)"
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_garbage_under_regex_retries_without_it(tmp_path):
    task, extra = _solved(tmp_path)
    want = _target(extra)
    c = Fake(["zzz", f"Answer: {want}"])
    res = run(arbitrate(task, extra, c))
    for m in ARBITRATE_MODELS:
        mine = [x for x in c.calls if x["model"] == m]
        assert "extra_body" in mine[0]["extra"] and mine[1]["extra"] == {}
    assert _written(task, extra, want)
    assert res.status == "solved"


def test_client_without_structured_support_skips_regex(tmp_path):
    task, extra = _solved(tmp_path)
    c = Fake([str(extra["ambiguous"][0]["chosen"] + 1)])
    c.structured_ok = False
    run(arbitrate(task, extra, c))
    assert len(c.calls) == 2 and all(x["extra"] == {} for x in c.calls)


def test_prompt_shows_changed_lines_and_probe_table(tmp_path):
    task, extra = _solved(tmp_path)
    c = Fake(["1"])
    run(arbitrate(task, extra, c))
    text = c.calls[0]["messages"][1]["content"]
    assert "count_even([1, 2, 4, 6, 7]) ->" in text and " | " in text
    assert "Option 1: line" in text and "`" in text
    assert "Option 1:\n" not in text   # no full sources per option


MULTI_LIB = "def a(x):\n    return x + 1\n\n\ndef b(x):\n    return x * 2\n"
MULTI_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
               "    def test_ab(self):\n        self.assertEqual(mylib.b(mylib.a(1)), 4)\n")


def test_multi_function_combination_failing_falls_back_to_single_change(tmp_path):
    d = tmp_path / "m"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(MULTI_LIB, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(MULTI_TESTS, encoding="utf-8")
    task = Task("m", "coding", "fix", 2.0, "unittest", d, d / "workspace")
    # both changed: b(a(1)) = (1*2)+2 = 4 would be green, so make the pair red on purpose:
    # a -> x * 3 gives 3, b -> x + 2 gives 5 (red together); b alone -> x + 2 on a(1)=2 gives 4 (green)
    extra = {"ambiguous": [
        {"func": "a", "candidates": ["def a(x):\n    return x + 1", "def a(x):\n    return x * 3"],
         "chosen": 0},
        {"func": "b", "candidates": ["def b(x):\n    return x * 2", "def b(x):\n    return x + 2"],
         "chosen": 0}]}
    res = run(arbitrate(task, extra, Fake(["2,2"])))
    text = (task.workspace / "mylib.py").read_text(encoding="utf-8")
    assert res.status == "solved"
    assert "return x + 1" in text and "return x + 2" in text   # only b changed
    assert run_tests(task.workspace).ok


def test_garbage_then_garbage_changes_nothing(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    c = Fake(["???", "no idea"])
    res = run(arbitrate(task, extra, c))
    assert res.status == "solved" and len(c.calls) == 4
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_out_of_range_number_changes_nothing(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    run(arbitrate(task, extra, Fake(["9", "9"])))
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_option_failing_tests_is_not_written(tmp_path):
    task, extra = _solved(tmp_path)
    extra["ambiguous"][0]["candidates"].append("def count_even(xs):\n    return 0")
    before = (task.workspace / "mylib.py").read_bytes()
    n = len(extra["ambiguous"][0]["candidates"])
    res = run(arbitrate(task, extra, Fake([str(n)])))
    assert res.status == "solved"
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_unavailable_is_unparsed_and_budget_propagates(tmp_path):
    task, extra = _solved(tmp_path)
    assert run(arbitrate(task, extra, Fake([LLMUnavailable("HTTP 503")]))).status == "unparsed"
    try:
        run(arbitrate(task, extra, Fake({MODEL_FAST: ["1"], MODEL_ALT: [BudgetExhausted("x")]})))
        assert False
    except BudgetExhausted:
        pass


def test_parse_choice():
    items = [{"candidates": ["a", "b"]}, {"candidates": ["a", "b", "c"]}]
    assert parse_choice("2,3", items) == [1, 2]
    assert parse_choice("1", items) is None
    assert parse_choice("1,4", items) is None


# ---------------------------------------------------------------- two-model agreement

def test_models_disagree_keeps_phase1_choice(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    want = _target(extra)
    keep = extra["ambiguous"][0]["chosen"] + 1
    for fast, alt in ((want, keep), (keep, want)):
        res = run(arbitrate(task, extra, Fake({MODEL_FAST: [str(fast)], MODEL_ALT: [str(alt)]})))
        assert res.status == "solved" and res.note == "arbitration kept the choice (models disagree)"
        assert (task.workspace / "mylib.py").read_bytes() == before


def test_models_naming_two_different_new_options_keep_phase1_choice(tmp_path):
    task, extra = _solved(tmp_path)
    it = extra["ambiguous"][0]
    others = [i + 1 for i in range(len(it["candidates"])) if i != it["chosen"]]
    if len(others) < 2:
        it["candidates"].append(it["candidates"][others[0] - 1])
        others.append(len(it["candidates"]))
    before = (task.workspace / "mylib.py").read_bytes()
    res = run(arbitrate(task, extra, Fake({MODEL_FAST: [str(others[0])], MODEL_ALT: [str(others[1])]})))
    assert res.status == "solved" and res.note == "arbitration kept the choice (models disagree)"
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_one_model_unavailable_keeps_phase1_choice(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    want = _target(extra)
    for replies in ({MODEL_FAST: [str(want)], MODEL_ALT: [LLMUnavailable("HTTP 404")]},
                    {MODEL_FAST: [LLMUnavailable("HTTP 404")], MODEL_ALT: [str(want)]}):
        res = run(arbitrate(task, extra, Fake(replies)))
        assert res.status == "solved" and res.answer == "DONE"
        assert res.note == "arbitration kept the choice (one model unavailable)"
        assert (task.workspace / "mylib.py").read_bytes() == before


def test_alt_unusable_reply_keeps_phase1_choice(tmp_path):
    task, extra = _solved(tmp_path)
    before = (task.workspace / "mylib.py").read_bytes()
    want = _target(extra)
    res = run(arbitrate(task, extra, Fake({MODEL_FAST: [str(want)], MODEL_ALT: ["??", "no"]})))
    assert res.status == "solved" and res.note == "arbitration kept the choice (unusable reply)"
    assert (task.workspace / "mylib.py").read_bytes() == before


def test_budget_exhausted_from_fast_propagates(tmp_path):
    task, extra = _solved(tmp_path)
    try:
        run(arbitrate(task, extra, Fake({MODEL_FAST: [BudgetExhausted("x")], MODEL_ALT: ["1"]})))
        assert False
    except BudgetExhausted:
        pass


def test_two_models_are_asked_concurrently(tmp_path):
    task, extra = _solved(tmp_path)
    want = _target(extra)

    class Slow(Fake):
        async def chat(self, model, messages, max_tokens, **kw):
            self.started.add(model)
            if len(self.started) == 1:
                await asyncio.wait_for(self.both.wait(), 2)   # times out if asked one after another
            else:
                self.both.set()
            return await super().chat(model, messages, max_tokens, **kw)

    async def go():
        c = Slow([str(want)])
        c.started, c.both = set(), asyncio.Event()
        return await arbitrate(task, extra, c)

    res = run(go())
    assert "chose" in res.note
    assert _written(task, extra, want)


def test_agreement_is_per_function(tmp_path):
    d = tmp_path / "m"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(MULTI_LIB, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(MULTI_TESTS, encoding="utf-8")
    task = Task("m", "coding", "fix", 2.0, "unittest", d, d / "workspace")
    extra = {"ambiguous": [
        {"func": "a", "candidates": ["def a(x):\n    return x + 1", "def a(x):\n    return x * 3"],
         "chosen": 0},
        {"func": "b", "candidates": ["def b(x):\n    return x * 2", "def b(x):\n    return x + 2"],
         "chosen": 0}]}
    # a: FAST 2 / ALT 1 -> keep; b: both 2 -> change
    res = run(arbitrate(task, extra, Fake({MODEL_FAST: ["2,2"], MODEL_ALT: ["1,2"]})))
    text = (task.workspace / "mylib.py").read_text(encoding="utf-8")
    assert res.status == "solved" and "return x + 1" in text and "return x + 2" in text
    assert run_tests(task.workspace).ok

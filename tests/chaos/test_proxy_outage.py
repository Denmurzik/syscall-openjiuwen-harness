"""Short proxy outages must not cost the run; a proxy gone for good must still end it."""
import json
import re
import time
from pathlib import Path

import pytest

from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from tests.chaos import fakeproxy
from tests.chaos.conftest import answers_on_disk, check_common, run_harness

pytestmark = pytest.mark.chaos

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
WORDS = "apples pears plums figs dates limes kiwis mangos peaches cherries grapes melons".split()
CODING = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
          "edit the tests) until all tests pass. Work only inside the current directory. When you are done, write "
          "ONLY the word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > "
          "answer.txt).")
LIB = "def f(x):\n    return x + 1\n\n\ndef g(xs):\n    return sum(xs)\n"
TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n"
         "        self.assertEqual(m.f(2), 4)\n\n    def test_b(self):\n        self.assertEqual(m.f(2), 5)\n\n"
         "    def test_g(self):\n        self.assertEqual(m.g([1, 2]), 3)\n\n\n"
         "if __name__ == '__main__':\n    unittest.main()\n")


def _math_tasks(root: Path, n: int) -> Path:
    """Task texts the deterministic solvers cannot read: only the model can answer them."""
    for i in range(n):
        d = root / f"math_{i:04d}"
        (d / "workspace").mkdir(parents=True)
        w = WORDS[i % len(WORDS)]
        (d / "task.json").write_text(json.dumps({
            "task_id": d.name, "task_type": "math", "reward": 5, "check": "exact_int",
            "instruction": f"Basket number {i} is full of {w} and a friend brings {i + 3} more {w} on day {i + 1}; "
                           f"how many {w} were brought?" + TAIL}), encoding="utf-8")
    return root


def _coding_tasks(root: Path, n: int) -> Path:
    for i in range(n):
        d = root / f"coding_{i:04d}"
        (d / "workspace").mkdir(parents=True)
        (d / "workspace" / "mylib.py").write_text(LIB, newline="\n", encoding="utf-8")
        (d / "workspace" / "test_mylib.py").write_text(TESTS, newline="\n", encoding="utf-8")
        (d / "task.json").write_text(json.dumps({"task_id": d.name, "task_type": "coding", "reward": 2,
                                                 "check": "unittest", "instruction": CODING}), encoding="utf-8")
    return root


class Blip:
    """`action` for the first `seconds` after the first request, then healthy."""
    def __init__(self, seconds, action="500"):
        self.seconds, self.action, self.t0 = seconds, action, None

    def __call__(self, n, body):
        self.t0 = self.t0 or time.monotonic()
        return self.action if time.monotonic() - self.t0 < self.seconds else "normal"


def _spent(run) -> float:
    return float(re.search(r"spent=([\d.]+)", run.stdout).group(1))


@pytest.mark.parametrize("action", ["500", "503"])
def test_blip_answers_every_task(action, tmp_path, make_proxy):
    tasks = _math_tasks(tmp_path / "tasks", 12)
    proxy = make_proxy(Blip(2.0, action))
    run = run_harness(tasks, tmp_path / "run", proxy,
                      {"BUDGET": 100000, "HARNESS_RETRY_DELAYS": "0.2,0.4,0.8,1.6", "HARNESS_PROXY_DEAD_S": 60},
                      timeout=120)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert len(answers_on_disk(tasks)) == 12, run.stdout
    assert proxy.count("normal") >= 12 and _spent(run) < 6000


def test_blip_of_broken_connections_is_recovered_by_the_retry_pass(tmp_path, make_proxy):
    """Resets mid-request are charged like timeouts and not repeated in place; the pass answers the tasks."""
    tasks = _math_tasks(tmp_path / "tasks", 6)
    proxy = make_proxy(Blip(1.0, "reset"))
    run = run_harness(tasks, tmp_path / "run", proxy,
                      {"BUDGET": 100000, "HARNESS_RETRY_DELAYS": "0.2,0.4,0.8", "HARNESS_PROXY_DEAD_S": 60},
                      timeout=120)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert len(answers_on_disk(tasks)) == 6, run.stdout


def test_single_500_loses_no_task(tmp_path, make_proxy):
    tasks = _math_tasks(tmp_path / "tasks", 3)
    proxy = make_proxy(lambda n, b: "500" if n == 1 else "normal")
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 100000, "HARNESS_PROXY_DEAD_S": 60}, timeout=120)
    assert run.rc == 0 and len(answers_on_disk(tasks)) == 3, run.stdout


@pytest.mark.parametrize("action", ["500", "reset"])
def test_proxy_gone_for_good_ends_the_run_after_dead_s(action, tasks, tmp_path, make_proxy, baseline):
    stamps = []

    def behavior(n, body):
        stamps.append(time.monotonic())
        return action
    proxy = make_proxy(behavior)
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 10 ** 7, "HARNESS_PROXY_DEAD_S": 8}, timeout=150)
    check_common(run, tasks, baseline, bound=120)
    if action == "500":  # held calls: only the dead limit can end this; a reset is not repeated in place,
        # so its tasks simply run out (twice, with the retry pass)
        assert stamps[-1] - stamps[0] >= 7, "ended before the dead limit: three failures must not end the run"
        assert _spent(run) == 0, "5xx must not be charged"


def test_deadline_ends_the_wait_on_the_shared_pause(tasks, tmp_path, make_proxy, baseline):
    proxy = make_proxy(lambda n, b: "500")
    run = run_harness(tasks, tmp_path / "run", proxy,
                      {"BUDGET": 50000, "HARNESS_PROXY_DEAD_S": 100000, "HARNESS_DEADLINE_S": 45,
                       "HARNESS_STOP_NEW_BEFORE_DEADLINE_S": 10},
                      timeout=150)
    check_common(run, tasks, baseline, bound=90)
    assert proxy.requests > 0 and _spent(run) == 0


def test_max_tokens_stays_within_budget(tmp_path, make_proxy):
    """The review's maxout: every reply as long as allowed, and a budget smaller than one ladder."""
    tasks = _coding_tasks(tmp_path / "tasks", 2)
    proxy = make_proxy(lambda n, b: "maxout")
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 3000}, timeout=150)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert proxy.requests > 0 and _spent(run) <= 3000, run.stdout


FIXABLE = ("import unittest\n\nimport mylib\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n"
           "        self.assertEqual(mylib.{fn}(4), 24)\n\n    def test_b(self):\n"
           "        self.assertEqual(mylib.{fn}(3), 6)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")


def _fixable_tasks(root: Path, n: int, reward: int = 2) -> Path:
    for i in range(n):
        d = root / f"coding_{i:04d}"
        (d / "workspace").mkdir(parents=True)
        (d / "workspace" / "mylib.py").write_text(f"def easy{i}(n):\n    return n\n", newline="\n", encoding="utf-8")
        (d / "workspace" / "test_mylib.py").write_text(FIXABLE.format(fn=f"easy{i}"), newline="\n", encoding="utf-8")
        (d / "task.json").write_text(json.dumps({"task_id": d.name, "task_type": "coding", "reward": reward,
                                                 "check": "unittest", "instruction": CODING}), encoding="utf-8")
    return root


def _patch_for(body):
    fn = re.search(r"def (\w+)\(n\)", str(body.get("messages"))).group(1)
    return f"FUNC {fn}\ndef {fn}(n):\n    return 1 if n < 2 else n * {fn}(n - 1)\n"


def _only_fast_down(seen):
    def behavior(n, body):
        seen[body.get("model")] = seen.get(body.get("model"), 0) + 1
        return "503" if body.get("model") == MODEL_FAST else "normal"
    return behavior


ONE_MODEL_ENV = {"BUDGET": 100000, "HARNESS_PROXY_DEAD_S": 20, "HARNESS_RETRY_DELAYS": "0.2,0.4,0.8",
                 "HARNESS_PHASE1_TIMEOUT_CODING": 4}


def test_one_model_down_the_ladder_solves_on_the_next(tmp_path, make_proxy, monkeypatch):
    monkeypatch.setattr(fakeproxy, "canned", _patch_for)
    tasks = _fixable_tasks(tmp_path / "tasks", 6)
    seen = {}
    run = run_harness(tasks, tmp_path / "run", make_proxy(_only_fast_down(seen)), ONE_MODEL_ENV, timeout=120)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert "solved=6" in run.stdout, (run.stdout, seen)
    assert seen[MODEL_ALT] >= 6 and seen[MODEL_FAST] >= 2


def test_one_model_down_routes_with_no_rung_reach_alt(tmp_path, make_proxy):
    tasks = _math_tasks(tmp_path / "tasks", 6)
    seen = {}
    run = run_harness(tasks, tmp_path / "run", make_proxy(_only_fast_down(seen)), ONE_MODEL_ENV, timeout=120)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert len(answers_on_disk(tasks)) == 6, (run.stdout, seen)
    assert seen[MODEL_ALT] >= 1


def test_reasoning_model_down_does_not_wait_for_the_dead_limit(tmp_path, make_proxy, monkeypatch):
    """Only the last rung fails and nothing else is left to run: the proxy answered this ladder a moment ago."""
    monkeypatch.setattr(fakeproxy, "canned", lambda body: "not a patch")
    tasks = _fixable_tasks(tmp_path / "tasks", 2, reward=5)
    t0 = time.monotonic()
    run = run_harness(tasks, tmp_path / "run",
                      make_proxy(lambda n, b: "503" if b.get("model") == MODEL_REASON else "normal"),
                      {**ONE_MODEL_ENV, "HARNESS_PROXY_DEAD_S": 100}, timeout=120)
    assert run.rc == 0 and not run.timed_out, run.stderr[-800:]
    assert time.monotonic() - t0 < 60, run.stdout

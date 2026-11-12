"""Round 19, task 84: the review checks methods of classes on scenarios built from the visible tests."""
import asyncio

import pytest

from harness import review
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from harness.llm import ChatResult
from harness.solvers.testrun import run_tests
from harness.tasks import Task

BANK = '''class BankAccount:
    """A simple account: deposits, withdrawals, a history of operations."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self.history.append(("deposit", amount))

    def transfer(self, other, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        other.balance += amount
        self.history.append(("transfer", amount))
'''
BANK_TESTS = '''import unittest
from mylib import *


class T(unittest.TestCase):
    def test_deposit(self):
        a = BankAccount("ann")
        a.deposit(50)
        self.assertEqual(a.balance, 50)

    def test_transfer(self):
        a, b = BankAccount("a", 80), BankAccount("b")
        a.transfer(b, 30)
        self.assertEqual((a.balance, b.balance), (50, 30))
        self.assertEqual(a.history, [("transfer", 30)])


if __name__ == "__main__":
    unittest.main()
'''
HEAD = ("def transfer(self, other, amount):\n    if amount <= 0:\n        raise ValueError('amount must be positive')\n"
        "    if amount > self.balance:\n        raise ValueError('insufficient funds')\n    self.balance -= amount\n")
FIX = HEAD + "    other.deposit(amount)\n    self.history.append(('transfer', amount))\n"
FIX2 = HEAD + "    other.balance += amount\n    other.history.append(('deposit', amount))\n" \
              "    self.history.append(('transfer', amount))\n"
WRONG = HEAD + "    other.balance += amount\n    other.history.append(('received', amount))\n" \
               "    self.history.append(('transfer', amount))\n"


def _reply(name, *texts):
    return "\n".join(f"FUNC {name}\n{t}" for t in texts)


def _task(tmp_path, lib=BANK, tests=BANK_TESTS, target="mylib.py", test_name="test_mylib.py"):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / target).write_text(lib, encoding="utf-8")
    (d / "workspace" / test_name).write_text(tests, encoding="utf-8")
    (d / "answer.txt").write_text("DONE", encoding="utf-8")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


class Fake:
    def __init__(self, fast, alt="NONE", reason="NONE"):
        self.replies, self.calls = {MODEL_FAST: fast, MODEL_ALT: alt, MODEL_REASON: reason}, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        return ChatResult(text=self.replies[model], prompt_tokens=10, completion_tokens=1, cost=2)


def _lib(task, target="mylib.py"):
    return (task.workspace / target).read_text(encoding="utf-8")


def _go(task, client):
    return asyncio.run(review.review(task, client))


def _bank(text):
    ns = {}
    exec(text, ns)
    a, b = ns["BankAccount"]("a", 80), ns["BankAccount"]("b")
    a.transfer(b, 80)
    return a.balance, a.history, b.balance, b.history


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv("HARNESS_REVIEW_METHODS", raising=False)


def test_two_agreeing_method_rewrites_replace_the_silent_bug(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply("BankAccount.transfer", FIX), _reply("BankAccount.transfer", FIX2))
    changed, note = _go(task, c)
    assert changed == ["BankAccount.transfer"] and c.calls == [MODEL_FAST, MODEL_ALT]
    assert _bank(_lib(task)) == (0, [("transfer", 80)], 80, [("deposit", 80)])
    assert "    def deposit(self, amount):" in _lib(task) and run_tests(task.workspace).ok


def test_switch_off_is_round_18(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_METHODS", "0")
    task = _task(tmp_path)
    c = Fake(_reply("BankAccount.transfer", FIX), _reply("BankAccount.transfer", FIX2))
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST, MODEL_ALT] and _lib(task) == BANK
    assert note == "review: no evidence against ['BankAccount.transfer'], kept"


def test_method_no_visible_test_calls_costs_no_second_call(tmp_path):
    lib = BANK + "\n    def total_deposited(self):\n        return sum(a for k, a in self.history if k == 'deposit')\n"
    task = _task(tmp_path, lib=lib)
    fix = "def total_deposited(self):\n    return sum(a for k, a in self.history if k in ('deposit', 'in'))\n"
    c = Fake(_reply("BankAccount.total_deposited", fix), _reply("BankAccount.total_deposited", fix))
    changed, note = _go(task, c)
    assert changed == [] and c.calls == [MODEL_FAST] and "no visible test calls" in note and _lib(task) == lib


def test_method_rewrites_that_disagree_go_to_the_third_opinion(tmp_path):
    task = _task(tmp_path)
    c = Fake(_reply("BankAccount.transfer", WRONG), _reply("BankAccount.transfer", FIX),
             _reply("BankAccount.transfer", FIX2))
    changed, note = _go(task, c)
    assert changed == ["BankAccount.transfer"] and c.calls == [MODEL_FAST, MODEL_ALT, MODEL_REASON]
    assert _bank(_lib(task))[3] == [("deposit", 80)]


def test_method_guard_nobody_names_is_not_stored(tmp_path):
    # both rewrites refuse a transfer to oneself, where ours carries it out: a raise nobody names, kept
    lib = BANK.replace('            raise ValueError("insufficient funds")\n', '            raise RuntimeError("no")\n')
    tests = BANK_TESTS.replace('a.transfer(b, 30)', 'a.transfer(b, 30)\n        a.transfer(a, 10)').replace(
        "(50, 30)", "(50, 30)").replace('[("transfer", 30)]', '[("transfer", 30), ("transfer", 10)]')
    guard = HEAD.replace("raise ValueError('insufficient funds')", "raise RuntimeError('no')") + \
        "    if other is self:\n        raise KeyError('self')\n    other.balance += amount\n" \
        "    self.history.append(('transfer', amount))\n"
    task = _task(tmp_path, lib=lib, tests=tests)
    assert run_tests(task.workspace).ok
    assert review._verdict(lib, tests, "BankAccount.transfer", guard, guard) == ([], False)   # the raise rule
    named = lib.replace("    def transfer(self, other, amount):\n",
                        "    def transfer(self, other, amount):\n        # KeyError for a transfer to oneself\n")
    assert review._verdict(named, tests, "BankAccount.transfer", guard, guard)[0] == [guard, guard]   # named
    changed, _ = _go(task, Fake(_reply("BankAccount.transfer", guard), _reply("BankAccount.transfer", guard)))
    assert changed == [] and _lib(task) == lib


def test_method_rewrite_red_on_visible_tests_keeps_the_file(tmp_path):
    red = HEAD + "    other.deposit(amount)\n"     # forgets our own history line: test_transfer turns red
    task = _task(tmp_path)
    changed, _ = _go(task, Fake(_reply("BankAccount.transfer", red), _reply("BankAccount.transfer", red)))
    assert changed == [] and _lib(task) == BANK


def test_scenes_of_a_target_file_under_its_own_name(tmp_path):
    tests = BANK_TESTS.replace("from mylib import *", "from bank import BankAccount")
    task = _task(tmp_path, target="bank.py", tests=tests, test_name="test_bank.py")
    changed, _ = _go(task, Fake(_reply("BankAccount.transfer", FIX), _reply("BankAccount.transfer", FIX2)))
    assert changed == ["BankAccount.transfer"] and _bank(_lib(task, "bank.py"))[3] == [("deposit", 80)]


STACK = '''class Stack:
    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if not self._items:
            raise IndexError("empty")
        return self._items.pop(0)

    def __len__(self):
        return len(self._items)
'''
STACK_TESTS = '''import pytest
from mylib import Stack


@pytest.fixture
def s():
    return Stack()


def test_one():
    st = Stack()
    st.push(4)
    assert st.pop() == 4
    with pytest.raises(IndexError):
        st.pop()


def test_with_fixture(s):
    s.push(1)
    assert s.pop() == 1
'''


def test_scenes_read_pytest_functions_and_skip_fixtures():
    head, scenes = review._scenes(STACK_TESTS, "Stack.pop")
    assert "import pytest" not in head and "fixture" not in head and "from mylib import Stack" in head
    assert scenes and all("s.push(1)" not in t for t in scenes)          # the test with a fixture is left out
    assert "st.push(4)" in scenes[0] and "_rec(st.pop(), 4)" in scenes[0]
    assert any("st.push(7)" in t for t in scenes)                         # the input varies
    assert all("_rec(st.pop(), 4)" in t for t in scenes)                  # the expected value does not
    assert review._scenes(STACK_TESTS, "Stack.__len__") == ("", [])      # a dunder other than __init__
    assert review._scenes(STACK_TESTS, "Queue.pop") == ("", [])          # a class the tests never build
    assert review._scenes("def f(:\n", "Stack.pop") == ("", [])


def test_scene_outcomes_show_values_state_and_raises():
    head, scenes = review._scenes(STACK_TESTS, "Stack.pop")
    ours = review._scene_outcomes(STACK, "mylib", head, scenes[:1], 10)
    lifo = STACK.replace("self._items.pop(0)", "self._items.pop()")
    assert ours == review._scene_outcomes(lifo, "mylib", head, scenes[:1], 10)   # one item: no difference
    assert "<raised:IndexError>" in ours[0]
    two = scenes[0].replace("_rec(st.push(4))", "_rec(st.push(4))\nst.push(5)")
    a, b = (review._scene_outcomes(t, "mylib", head, [two], 10)[0] for t in (STACK, lifo))
    assert a == '["None", "4", "4", "5", "<raised:none>", "st=Stack{}"]'      # public attributes only
    assert b == '["None", "5", "4", "4", "<raised:none>", "st=Stack{}"]'
    broken = STACK.replace("raise IndexError", "raise ZeroDivisionError")
    assert review._scene_outcomes(broken, "mylib", head, ["st = Stack()\n_rec(st.pop())"], 10) == \
        ["<exc:ZeroDivisionError>"]
    assert review._scene_outcomes("x = (", "mylib", head, scenes[:2], 10) == ["<exc:SyntaxError>"] * 2


def test_object_state_counts_where_return_values_agree():
    lib = "class Box:\n    def __init__(self):\n        self.n = 0\n        self.log = []\n\n" \
          "    def add(self, k):\n        self.n += k\n        return self.n\n"
    tests = "import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n    def setUp(self):\n" \
            "        self.b = m.Box()\n\n    def test_add(self):\n        self.assertEqual(self.b.add(2), 2)\n"
    fix = "def add(self, k):\n    self.n += k\n    self.log.append(k)\n    return self.n\n"
    fix2 = "def add(self, k):\n    self.log.append(k)\n    self.n = self.n + k\n    return self.n\n"
    texts, split = review._verdict(lib, tests, "Box.add", fix, fix2)
    assert texts == [fix, fix2] and not split

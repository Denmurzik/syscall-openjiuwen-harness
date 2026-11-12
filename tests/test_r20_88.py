"""Round 20, task 88: a green model patch of the ladder is cut back to the definitions the visible tests need."""
import asyncio

from harness.llm import ChatResult
from harness.llm_routes import MODEL_FAST, _cut_back, _restored, fix_coding
from harness.tasks import Task

BANK = '''class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self.history.append(("deposit", amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount >= self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        self.history.append(("withdraw", amount))

    def transfer(self, other, amount):
        self.withdraw(amount)
        other.deposit(amount)
        self.history.append(("transfer", amount))
'''
BANK_TESTS = '''import unittest
from mylib import BankAccount


class T(unittest.TestCase):
    def test_withdraw_all(self):
        a = BankAccount("a", 10)
        a.withdraw(10)
        self.assertEqual(a.balance, 0)

    def test_transfer(self):
        a, b = BankAccount("a", 10), BankAccount("b")
        a.transfer(b, 10)
        self.assertEqual((a.balance, b.balance), (0, 10))
'''
# the model fixes `withdraw` and also inlines the correct `transfer`: green, but the history entries are gone
BANK_PATCH = '''FUNC BankAccount.withdraw
def withdraw(self, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    if amount > self.balance:
        raise ValueError("insufficient funds")
    self.balance -= amount
    self.history.append(("withdraw", amount))

FUNC BankAccount.transfer
def transfer(self, other, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    if amount > self.balance:
        raise ValueError("insufficient funds")
    self.balance -= amount
    other.balance += amount
    self.history.append(("transfer", amount))
'''

TWO = "def lo(xs):\n    return max(xs)\n\n\ndef hi(xs):\n    return min(xs)\n"
TWO_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_lo(self):\n        self.assertEqual(mylib.lo([3, 1, 2]), 1)\n\n"
             "    def test_hi(self):\n        self.assertEqual(mylib.hi([3, 1, 2]), 3)\n")
TWO_PATCH = "FUNC lo\ndef lo(xs):\n    return min(xs)\n\nFUNC hi\ndef hi(xs):\n    return max(xs)\n"


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "user": messages[-1]["content"]})
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=10, cost=11)


def _task(root, lib, tests):
    ws = root / "coding_1" / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8", newline="\n")
    return Task("coding_1", "coding", "Fix the bugs in mylib.py", 1.0, "unittest", ws.parent, ws)


def _mylib(t):
    return (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_the_whole_definition_pass_keeps_the_fix_and_the_original_transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MINIMAL_DEFS", "1")   # off by default
    t = _task(tmp_path, BANK, BANK_TESTS)
    r = asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    assert r.status == "solved" and r.note == f"patched by {MODEL_FAST}, cut back to the original text: " \
                                              "['BankAccount.transfer']"
    assert _mylib(t) == BANK.replace("amount >= self.balance", "amount > self.balance")


def test_a_second_needed_change_is_kept(tmp_path):
    t = _task(tmp_path, TWO, TWO_TESTS)
    r = asyncio.run(fix_coding(t, FakeClient([TWO_PATCH])))
    assert r.status == "solved" and r.note == f"patched by {MODEL_FAST}"
    assert "return min(xs)" in _mylib(t) and "return max(xs)" in _mylib(t).split("def hi")[1]


def test_a_helper_the_model_added_stays(tmp_path):
    # the fix calls a new helper: restoring `lo` turns a test red, the helper stays whatever happens
    patched = ("def _least(xs):\n    return sorted(xs)[0]\n\n\ndef lo(xs):\n    return _least(xs)\n\n\n"
               "def hi(xs):\n    return max(xs)\n")
    t = _task(tmp_path, TWO, TWO_TESTS)
    fixed, cut = asyncio.run(_cut_back(t.workspace, TWO, patched))
    assert cut == [] and fixed == patched


def test_unchanged_and_added_definitions_have_no_restore():
    assert _restored(TWO, TWO, "lo") is None
    assert _restored("def new():\n    return 1\n" + TWO, TWO, "new") is None
    assert _restored(TWO.replace("max", "min", 1), TWO, "lo") == TWO


def test_switch_off_is_round_19(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MINIMAL_PATCH", "0")
    t = _task(tmp_path, BANK, BANK_TESTS)
    r = asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    assert r.status == "solved" and r.note == f"patched by {MODEL_FAST}"
    assert "other.balance += amount" in _mylib(t)


def test_out_of_time_keeps_the_green_patch(tmp_path, monkeypatch):
    from harness import config
    monkeypatch.setattr(config, "MINIMAL_PATCH_S", -1.0)
    t = _task(tmp_path, BANK, BANK_TESTS)
    r = asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    assert r.status == "solved" and r.note == f"patched by {MODEL_FAST}"
    assert "other.balance += amount" in _mylib(t)


# far8 coding_8001 as it is: the visible test wants no `withdraw` entry from `transfer`, so the model's `transfer`
# stays; only the line where it inlined `other.deposit` takes the call back (the hidden test reads b's history)
BANK_TESTS_8001 = BANK_TESTS.replace(
    "        self.assertEqual((a.balance, b.balance), (0, 10))\n",
    "        self.assertEqual((a.balance, b.balance), (0, 10))\n"
    "        self.assertEqual(a.history, [(\"transfer\", 10)])\n")


def test_the_real_coding_8001_takes_back_the_inlined_deposit_call(tmp_path):
    t = _task(tmp_path, BANK, BANK_TESTS_8001)
    r = asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    assert r.status == "solved" and "BankAccount.transfer: other.deposit(amount)" in r.note
    text = _mylib(t)
    assert "other.deposit(amount)" in text and "other.balance += amount" not in text
    assert "amount > self.balance" in text and "amount >= self.balance" not in text


CLAMP = "def take(stock, n):\n    left = stock - n\n    if left < 0:\n        return left\n    else:\n        return 0\n"
CLAMP_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
               "    def test_take(self):\n        self.assertEqual(mylib.take(5, 2), 3)\n"
               "        self.assertEqual(mylib.take(1, 4), 0)\n")
CLAMP_PATCH = "FUNC take\ndef take(stock, n):\n    left = stock - n\n    if left <= 0:\n        return 0\n" \
              "    else:\n        return left\n"


def test_a_condition_of_a_rewritten_block_is_never_taken_back(tmp_path):
    # deep19 coding_9707: `if left <= 0:` -> `if left < 0:` keeps the visible tests green and breaks the hidden ones
    t = _task(tmp_path, CLAMP, CLAMP_TESTS)
    r = asyncio.run(fix_coding(t, FakeClient([CLAMP_PATCH])))
    assert r.status == "solved" and r.note == f"patched by {MODEL_FAST}"
    assert "if left <= 0:" in _mylib(t)


def test_switch_off_keeps_the_inlined_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MINIMAL_PATCH", "0")
    t = _task(tmp_path, BANK, BANK_TESTS_8001)
    asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    assert "other.balance += amount" in _mylib(t)


# ---------------------------------------------------------------- fix round 1 (review probes P1, P2, P3b, P4)
def _cut(tmp_path, orig, fixed, tests):
    t = _task(tmp_path, fixed, tests)
    return asyncio.run(_cut_back(t.workspace, orig, fixed))


def test_p1_a_faithful_inline_gets_no_second_call(tmp_path):
    # the model wrote out all of `deposit` on `other`: giving the call back would log the deposit twice
    faithful = BANK.replace("amount >= self.balance", "amount > self.balance").replace(
        "        self.withdraw(amount)\n        other.deposit(amount)\n",
        "        if amount > self.balance:\n            raise ValueError(\"insufficient funds\")\n"
        "        self.balance -= amount\n        other.balance += amount\n"
        "        other.history.append((\"deposit\", amount))\n")
    fixed, cut = _cut(tmp_path, BANK, faithful, BANK_TESTS_8001)
    assert cut == [] and fixed == faithful


ORD = ("def fee(amount):\n    return amount * 2 // 100\n\n\ndef notify(log, order_id, who):\n"
       "    log.append((order_id, who))\n\n\ndef place(log, order_id, who, amount):\n"
       "    notify(log, who, order_id)\n    return amount + fee(amount)\n")
ORD_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n    def test_place(self):\n"
             "        self.assertEqual(mylib.place([], 1, \"x\", 100), 103)\n")


def test_p2_an_argument_fix_of_an_own_call_stays(tmp_path):
    fixed_src = ORD.replace("notify(log, who, order_id)", "notify(log, order_id, who)").replace(
        "return amount + fee(amount)", "return amount + fee(amount) + 1")
    fixed, cut = _cut(tmp_path, ORD, fixed_src, ORD_TESTS)
    assert cut == [] and "notify(log, order_id, who)" in fixed


TAGS = ("class Tags:\n    def __init__(self):\n        self.tags = set()\n        self.count = 0\n\n"
        "    def add(self, t):\n        self.tags.add(t)\n        self.count += 1\n\n"
        "    def remove(self, t):\n        self.tags.remove(t)\n        self.count += 1\n        return self.count\n")
TAGS_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n    def test_remove(self):\n"
              "        s = mylib.Tags()\n        s.add(\"a\")\n        self.assertEqual(s.remove(\"a\"), 0)\n")


def test_p3b_a_set_method_named_like_an_own_method_is_no_own_call(tmp_path):
    fixed_src = TAGS.replace("        self.tags.remove(t)\n        self.count += 1\n        return",
                             "        self.tags.discard(t)\n        self.count -= 1\n        return")
    fixed, cut = _cut(tmp_path, TAGS, fixed_src, TAGS_TESTS)
    assert cut == [] and "self.tags.discard(t)" in fixed


MEAN = "def mean(xs):\n    return sum(xs) // len(xs)\n\n\ndef spread(xs):\n    return max(xs) - max(xs)\n"
MEAN_FIX = MEAN.replace("sum(xs) // len(xs)", "sum(xs) / len(xs)").replace("max(xs) - max(xs)", "max(xs) - min(xs)")
MEAN_TESTS = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
              "    def test_mean(self):\n        self.assertEqual(mylib.mean([2, 4]), 3)\n\n"
              "    def test_spread(self):\n        self.assertEqual(mylib.spread([1, 4]), 3)\n")


def test_p4_a_silent_fix_in_another_definition_stays_by_default(tmp_path, monkeypatch):
    fixed, cut = _cut(tmp_path, MEAN, MEAN_FIX, MEAN_TESTS)
    assert cut == [] and "sum(xs) / len(xs)" in fixed
    monkeypatch.setenv("HARNESS_MINIMAL_DEFS", "1")   # the whole-definition pass would undo it
    fixed, cut = _cut(tmp_path / "on", MEAN, MEAN_FIX, MEAN_TESTS)
    assert cut == ["mean"] and "sum(xs) // len(xs)" in fixed


def test_the_green_patch_is_on_disk_before_the_cut_back_runs(tmp_path, monkeypatch):
    import harness.llm_routes as lr
    seen = []

    async def cancelled(ws, original, fixed, targs=()):
        seen.append((ws / "mylib.py").read_text(encoding="utf-8"))
        raise asyncio.CancelledError   # a deadline cancel inside a test run of the cut-back

    monkeypatch.setattr(lr, "_cut_back", cancelled)
    t = _task(tmp_path, BANK, BANK_TESTS_8001)
    try:
        asyncio.run(fix_coding(t, FakeClient([BANK_PATCH])))
    except asyncio.CancelledError:
        pass
    assert seen and "amount > self.balance" in seen[0] and _mylib(t) == seen[0]

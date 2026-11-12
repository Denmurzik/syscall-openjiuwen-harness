"""The review of a package module below the root (round 19, task 85, HARNESS_DEEP_TESTS)."""
import asyncio

from harness import review
from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import ChatResult
from harness.tasks import Task

from tests.test_r19_85 import DISC, DISC_FIX, DISC_TESTS, _ws


class ReviewFake:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        return ChatResult(text=self.reply, prompt_tokens=10, completion_tokens=1, cost=2)


def test_review_repairs_a_silent_bug_in_a_package_module(tmp_path):
    t = _ws(tmp_path, {"shop/__init__.py": "", "shop/pricing.py": DISC, "tests/unit/test_p.py": DISC_TESTS})
    (t.dir / "answer.txt").write_text("DONE", encoding="utf-8")
    c = ReviewFake("FUNC bulk_price\n" + DISC_FIX)
    changed, note = asyncio.run(review.review(t, c))
    assert changed == ["bulk_price"] and c.calls == [MODEL_FAST, MODEL_ALT], note
    assert "qty >= 10" in (t.workspace / "shop" / "pricing.py").read_text(encoding="utf-8")


# M8: the class-method scenes import a package target by its dotted name; a target no probe can load is skipped

from tests.test_r19_84 import BANK, FIX, FIX2, _bank  # noqa: E402

PKG_TESTS = '''import unittest
from bank.account import BankAccount


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
'''


class MethodFake:
    def __init__(self, fast, alt):
        self.replies, self.calls = {MODEL_FAST: fast, MODEL_ALT: alt}, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        return ChatResult(text=self.replies[model], prompt_tokens=10, completion_tokens=1, cost=2)


def test_module_name():
    assert review._module_name("mylib.py") == "mylib"
    assert review._module_name("bank/account.py") == "bank.account"
    assert review._module_name("shop/pricing/__init__.py") == "shop.pricing"
    assert review._module_name("src/wallet/ledger.py") == "wallet.ledger"


def test_method_review_of_a_package_module(tmp_path):
    t = _ws(tmp_path, {"bank/__init__.py": "", "bank/account.py": BANK, "tests/test_account.py": PKG_TESTS})
    (t.dir / "answer.txt").write_text("DONE", encoding="utf-8")
    c = MethodFake("FUNC BankAccount.transfer\n" + FIX, "FUNC BankAccount.transfer\n" + FIX2)
    changed, note = asyncio.run(review.review(t, c))
    assert changed == ["BankAccount.transfer"] and c.calls == [MODEL_FAST, MODEL_ALT], note
    assert _bank((t.workspace / "bank" / "account.py").read_text(encoding="utf-8"))[3] == [("deposit", 80)]


def test_a_package_module_no_probe_can_load_costs_no_call(tmp_path):
    lib = "from .money import cents\n\n\n" + BANK
    t = _ws(tmp_path, {"bank/__init__.py": "", "bank/money.py": "def cents(x):\n    return x\n",
                       "bank/account.py": lib, "tests/test_account.py": PKG_TESTS})
    (t.dir / "answer.txt").write_text("DONE", encoding="utf-8")
    c = MethodFake("FUNC BankAccount.transfer\n" + FIX, "FUNC BankAccount.transfer\n" + FIX2)
    changed, note = asyncio.run(review.review(t, c))
    assert changed == [] and c.calls == [] and "cannot load" in note
    assert (t.workspace / "bank" / "account.py").read_text(encoding="utf-8") == lib
    # an absolute import of the workspace's own package is the same
    assert not review._loads_alone(t.workspace, "bank/account.py", "from bank.money import cents\n")
    assert review._loads_alone(t.workspace, "bank/account.py", "import math\n")
    assert review._loads_alone(t.workspace, "mylib.py", "from .x import y\n")   # a root file: as round 18

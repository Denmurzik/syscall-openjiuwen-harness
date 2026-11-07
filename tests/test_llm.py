import asyncio
from types import SimpleNamespace

import pytest

from harness import llm as llm_mod
from harness.budget import TokenLedger
from harness.config import TIMEOUT_CHARGE
from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable


class FakeModel:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.last_messages = None

    async def invoke(self, messages, **kw):
        self.calls += 1
        self.last_messages = messages
        return await self.behavior()


def make(behavior, timeout=180, monkeypatch=None):
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger, timeout=timeout)
    fake = FakeModel(behavior)
    c._model = lambda name: fake
    return c, fake, ledger


def run(c):
    return asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16))


def test_charges_ledger_from_usage():
    async def ok():
        return SimpleNamespace(content="hi", usage_metadata=SimpleNamespace(input_tokens=500, output_tokens=40))
    c, fake, ledger = make(ok)
    r = run(c)
    assert r.text == "hi" and r.cost == 90 and ledger.spent == 90
    assert r.prompt_tokens == 500 and r.completion_tokens == 40


def test_exhausted_ledger_short_circuits():
    async def ok():
        raise AssertionError
    c, fake, ledger = make(ok)
    ledger.mark_exhausted()
    with pytest.raises(BudgetExhausted):
        run(c)
    assert fake.calls == 0


class HTTPErr(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def test_402_marks_exhausted():
    async def bad():
        try:
            raise HTTPErr(402)
        except HTTPErr as e:
            raise RuntimeError("openAI API async invoke error") from e
    c, fake, ledger = make(bad)
    with pytest.raises(BudgetExhausted):
        run(c)
    assert ledger.exhausted and fake.calls == 1


def test_timeout_charged_not_retried():
    async def hang():
        await asyncio.sleep(5)
    c, fake, ledger = make(hang, timeout=0.1)
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and ledger.spent == TIMEOUT_CHARGE


def test_429_retried_then_succeeds(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    n = {"i": 0}

    async def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise HTTPErr(429)
        return SimpleNamespace(content="ok", usage_metadata=SimpleNamespace(input_tokens=10, output_tokens=1))
    c, fake, ledger = make(flaky)
    assert run(c).text == "ok"
    assert sleeps == [2, 4] and fake.calls == 3


def test_three_consecutive_failures_exhaust():
    async def bad():
        raise HTTPErr(500)
    c, fake, ledger = make(bad)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            run(c)
    with pytest.raises(BudgetExhausted):
        run(c)
    assert ledger.exhausted

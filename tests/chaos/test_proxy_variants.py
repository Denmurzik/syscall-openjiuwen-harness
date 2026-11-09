"""A jury proxy that differs from ours: another base URL, prefixed model ids, replies without usage."""
import asyncio

import pytest

from harness.budget import TokenLedger
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from harness.llm import LLMClient, LLMUnavailable
from tests.chaos.fakeproxy import Proxy

pytestmark = pytest.mark.chaos
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def make(monkeypatch):
    made = []

    def _make(proxy, base=None):
        proxy.start()
        made.append(proxy)
        ledger = TokenLedger(100000, 1000)
        return LLMClient(base or proxy.base_url, "secret-key", ledger, timeout=20), ledger
    yield _make
    for p in made:
        p.stop()


def ask(c, model=MODEL_FAST, n=3):
    return ask_many(c, [model] * n)


def ask_many(c, models):
    """One event loop for all calls: the HTTP clients inside are bound to the loop that made them."""
    async def go():
        return [await c.chat(m, MSG, 50) for m in models]
    return asyncio.run(go())


def test_healthy_proxy_sees_exactly_the_requests_we_send(make):
    p = Proxy()
    c, _ = make(p)
    ask_many(c, [MODEL_FAST, MODEL_FAST, MODEL_REASON])
    assert p.requests == 3 and p.misses == [] and p.listings == 0
    assert [b["model"] for b in p.bodies] == [MODEL_FAST, MODEL_FAST, MODEL_REASON]


@pytest.mark.parametrize("given", ["root", "double"])
def test_wrong_base_url_is_fixed_once_and_remembered(make, given):
    p = Proxy()
    c, _ = make(p, base=(p.root_url if given == "root" else p.base_url + "/v1"))
    ask(c, MODEL_FAST, 3)
    assert p.requests == 3
    assert len(p.misses) == 1  # one failed try, then the variant is remembered
    assert c.base_url == p.base_url


def test_base_url_that_is_broken_for_good_fails_without_looping(make):
    p = Proxy(mount="/other")
    c, _ = make(p, base=p.root_url + "/v1")

    async def go():
        for _ in range(3):
            with pytest.raises(LLMUnavailable):
                await c.chat(MODEL_FAST, MSG, 50)
    asyncio.run(go())
    assert p.requests == 0
    assert len(p.misses) == 3 + 1  # three calls, plus the one other spelling tried once per run


def test_prefixed_model_ids_are_mapped_through_the_listing(make):
    ids = [f"qwen/{MODEL_FAST}", f"qwen/{MODEL_ALT}", f"qwen/{MODEL_REASON}", "other/model"]
    p = Proxy(model_ids=ids)
    c, _ = make(p)
    ask_many(c, [MODEL_FAST, MODEL_FAST, MODEL_ALT, MODEL_REASON])
    assert p.listings == 1
    assert p.requests == 4
    assert [b["model"] for b in p.bodies] == [f"qwen/{MODEL_FAST}"] * 2 + [f"qwen/{MODEL_ALT}", f"qwen/{MODEL_REASON}"]


def test_models_missing_from_the_listing_stay_unavailable_with_one_listing(make):
    p = Proxy(model_ids=["other/model"])
    c, _ = make(p)
    async def go():
        for m in (MODEL_FAST, MODEL_ALT, MODEL_FAST):
            with pytest.raises(LLMUnavailable):
                await c.chat(m, MSG, 50)
    asyncio.run(go())
    assert p.listings == 1


def test_noreason_without_usage_is_charged_by_reply_length(make):
    p = Proxy(lambda n, b: "no_usage")
    c, ledger = make(p)
    r = ask(c, MODEL_FAST, 1)[0]
    assert r.completion_tokens < 50 and r.completion_tokens >= 1


def test_reasoning_model_without_usage_is_charged_the_limit(make):
    p = Proxy(lambda n, b: "no_usage")
    c, _ = make(p)
    r = ask(c, MODEL_REASON, 1)[0]
    assert r.completion_tokens == 50

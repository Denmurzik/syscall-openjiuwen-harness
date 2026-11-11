"""Round 15, task 73: a request that reached the proxy and failed is charged its input (prompt / 10), in the ledger
and in the unit's call list; a connect error never left and stays free; a transport failure keeps its flat charge.
The fix round: the input of a failed request is estimated high (chars / 3), and a failed ping is charged too."""
import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest

from harness import llm as llm_mod
from harness.budget import TokenLedger
from harness.config import TIMEOUT_CHARGE
from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable
from harness.trajectory import CALLS

MSGS = [{"role": "user", "content": "q" * 400}]
P_OK = len(str(MSGS)) // 4  # prompt tokens of the success-path estimate (and of hold())
P = math.ceil(len(str(MSGS)) / 3)  # prompt tokens charged for a failed request: the conservative estimate
INPUT = P / 10


class HTTPErr(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def _wrapped(exc):
    try:
        raise exc
    except Exception as e:
        try:
            raise RuntimeError("openAI API async invoke error") from e
        except RuntimeError as w:
            return w


def _ok(p=10, c=1):
    return SimpleNamespace(content="ok", usage_metadata=SimpleNamespace(input_tokens=p, output_tokens=c))


class Fake:
    """Raises the scripted exceptions in turn, then answers."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.kws = []

    async def invoke(self, messages, **kw):
        self.kws.append(dict(kw))
        if self.steps:
            e = self.steps.pop(0)
            raise e if isinstance(e, httpx.HTTPError) else _wrapped(e)
        return _ok()


def _client(fake, monkeypatch, budget=100000, reserve=1000):
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.01")
    ledger = TokenLedger(budget, reserve)
    c = LLMClient("http://x/v1", "k", ledger)
    c._model = lambda name: fake
    c.probe_models = ()
    clock = {"t": 1000.0}

    def now():
        return clock["t"]

    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        clock["t"] += s
        await real_sleep(0)
    c._now = now
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return c, ledger


def _chat(c, **extra):
    calls = []

    async def go():
        tok = CALLS.set(calls)
        try:
            return await c.chat("m", MSGS, 16, **extra)
        finally:
            CALLS.reset(tok)
    try:
        r = asyncio.run(go())
    except Exception as exc:
        return exc, calls
    return r, calls


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_then_success_charges_the_failed_input(monkeypatch, status):
    fake = Fake(HTTPErr(status))
    c, ledger = _client(fake, monkeypatch)
    r, calls = _chat(c)
    assert r.text == "ok" and len(fake.kws) == 2
    assert ledger.spent == pytest.approx(INPUT + r.cost) and ledger.held == 0
    assert calls[0]["prompt_tokens"] == P and calls[0]["cost"] == pytest.approx(INPUT) and "error" in calls[0]
    assert sum(e["cost"] for e in calls) == pytest.approx(ledger.spent)


def test_retried_429s_are_each_charged(monkeypatch):
    fake = Fake(HTTPErr(429), HTTPErr(429))
    c, ledger = _client(fake, monkeypatch)
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.01,0.01")
    r, calls = _chat(c)
    assert r.text == "ok" and len(fake.kws) == 3
    assert ledger.spent == pytest.approx(2 * INPUT + r.cost) and ledger.held == 0
    assert [e["prompt_tokens"] for e in calls[:2]] == [P, P] and all("error" in e for e in calls[:2])


def test_429_out_of_retries_charges_every_try(monkeypatch):
    fake = Fake(*[HTTPErr(429)] * 10)
    c, ledger = _client(fake, monkeypatch)
    exc, calls = _chat(c)
    assert isinstance(exc, LLMUnavailable)
    n = len(fake.kws)
    assert n >= 2 and ledger.spent == pytest.approx(n * INPUT) and len(calls) == n and ledger.held == 0


@pytest.mark.parametrize("status", [400, 404, 409, 413, 422])
def test_other_4xx_charged_its_input(monkeypatch, status):
    fake = Fake(*[HTTPErr(status)] * 5)
    c, ledger = _client(fake, monkeypatch)
    exc, calls = _chat(c)
    assert isinstance(exc, LLMUnavailable)
    n = len(fake.kws)  # a 400/404 may be tried again on a fixed endpoint: that try went out too
    assert ledger.spent == pytest.approx(n * INPUT) and len(calls) == n and ledger.held == 0
    assert all(e["prompt_tokens"] == P and e["completion_tokens"] == 0 for e in calls)


def test_endpoint_fix_retries_are_charged(monkeypatch):
    fake = Fake(HTTPErr(404))
    c, ledger = _client(fake, monkeypatch)

    async def fix(model, status, msg, gen):
        return True
    c._fix_endpoint = fix
    r, calls = _chat(c)
    assert r.text == "ok" and len(fake.kws) == 2
    assert ledger.spent == pytest.approx(INPUT + r.cost) and ledger.held == 0


def test_structured_400_retry_is_charged(monkeypatch):
    fake = Fake(HTTPErr(400))
    c, ledger = _client(fake, monkeypatch)
    r, calls = _chat(c, response_format={"type": "json_object"})
    assert r.text == "ok" and "response_format" in fake.kws[0] and "response_format" not in fake.kws[1]
    assert c.structured_ok is False
    assert ledger.spent == pytest.approx(INPUT + r.cost) and len(calls) == 2 and ledger.held == 0


@pytest.mark.parametrize("status", [402, 403])
def test_402_403_charged_then_exhausted(monkeypatch, status):
    fake = Fake(HTTPErr(status))
    c, ledger = _client(fake, monkeypatch)
    exc, calls = _chat(c)
    assert isinstance(exc, BudgetExhausted) and ledger.exhausted
    assert ledger.spent == pytest.approx(INPUT) and ledger.held == 0 and len(calls) == 1


def test_connect_error_stays_free(monkeypatch):
    fake = Fake(httpx.ConnectError("refused"), httpx.ConnectError("refused"))
    c, ledger = _client(fake, monkeypatch)
    r, calls = _chat(c)
    assert r.text == "ok" and len(fake.kws) == 3
    assert ledger.spent == pytest.approx(r.cost) and ledger.held == 0
    assert [e["cost"] for e in calls[:2]] == [0.0, 0.0] and calls[0]["prompt_tokens"] == 0


def test_transport_failure_keeps_its_flat_charge_only(monkeypatch):
    fake = Fake(httpx.ReadTimeout("slow"))
    c, ledger = _client(fake, monkeypatch)
    exc, calls = _chat(c)
    assert isinstance(exc, LLMUnavailable)  # ProxyDown: no model shows the proxy alive
    flat = min(TIMEOUT_CHARGE, math.ceil(P_OK / 10 + 16))
    assert len(calls) == 1 and calls[0]["prompt_tokens"] == 0 and calls[0]["completion_tokens"] == flat
    assert ledger.spent == flat and ledger.held == 0  # not the flat amount plus the input


def test_charge_stays_inside_the_hold_on_a_tight_budget(monkeypatch):
    """The input charge comes out of what hold() set aside: a run of failures never takes the ledger past its budget
    minus the reserve, and the call that no longer fits is refused before it is sent."""
    budget, reserve = 100, 10
    fake = Fake(*[HTTPErr(500)] * 1000)
    c, ledger = _client(fake, monkeypatch, budget=budget, reserve=reserve)
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", "1000000")
    big = [{"role": "user", "content": "q" * 2000}]  # about 50 units of input a try

    async def go():
        return await c.chat("m", big, 16)
    with pytest.raises((LLMUnavailable, BudgetExhausted)):
        asyncio.run(go())
    assert 0 < ledger.spent <= budget - reserve and ledger.held == 0


# ---------------------------------------------------------------- fix round: estimate, other exceptions, pings

@pytest.mark.parametrize("chars", [0, 1, 7, 100, 4001])
def test_failed_input_estimate_is_never_below_the_success_path(chars):
    msgs = [{"role": "user", "content": "z" * chars}]
    n = len(str(msgs))
    assert llm_mod._failed_input_tokens(msgs) == math.ceil(n / 3) >= n // 4


def test_failed_input_capped_by_the_hold(monkeypatch):
    """A long prompt with a small output limit: the charge stays within what hold() set aside."""
    big = [{"role": "user", "content": "q" * 30000}]
    fake = Fake(HTTPErr(500))
    c, ledger = _client(fake, monkeypatch)
    calls = []

    async def go():
        tok = CALLS.set(calls)
        try:
            return await c.chat("m", big, 1)
        finally:
            CALLS.reset(tok)
    asyncio.run(go())
    p_ok = len(str(big)) // 4
    assert calls[0]["prompt_tokens"] == min(math.ceil(len(str(big)) / 3), p_ok + 10 * 1) == p_ok + 10
    assert ledger.held == 0


def test_failure_with_no_status_that_is_not_a_transport_error_is_charged(monkeypatch):
    fake = Fake(ValueError("bad reply"))
    c, ledger = _client(fake, monkeypatch)
    exc, calls = _chat(c)
    assert isinstance(exc, LLMUnavailable) and len(fake.kws) == 1
    assert ledger.spent == pytest.approx(INPUT) and ledger.held == 0 and len(calls) == 1


def _pinged(monkeypatch, ping_exc):
    """fast keeps failing with 503; alt answers the ping with `ping_exc` (None: a reply)."""
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger)
    fast, alt = Fake(*[HTTPErr(503)] * 50), Fake(*([ping_exc] * 50 if ping_exc is not None else []))
    c._model = lambda name: {"fast": fast, "alt": alt}[name]
    c.probe_models = ("fast", "alt")
    clock = {"t": 1000.0}
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        clock["t"] += s
        await real_sleep(0)
    c._now = lambda: clock["t"]
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", "300")
    calls = []

    async def go():
        tok = CALLS.set(calls)
        try:
            return await c.chat("fast", MSGS, 16)
        finally:
            CALLS.reset(tok)
    try:
        asyncio.run(go())
    except Exception:
        pass
    return ledger, fast, alt, calls


PING_P = math.ceil(len(str(llm_mod._PING)) / 3)


@pytest.mark.parametrize("ping_exc", [HTTPErr(503), HTTPErr(429), HTTPErr(400)])
def test_failed_ping_that_reached_the_proxy_is_charged(monkeypatch, ping_exc):
    ledger, fast, alt, calls = _pinged(monkeypatch, ping_exc)
    pings = [e for e in calls if e["model"] == "alt"]
    assert alt.kws and all(k == {"max_tokens": 1} for k in alt.kws) and len(pings) == len(alt.kws)
    assert all(e["prompt_tokens"] == PING_P and e["cost"] == pytest.approx(PING_P / 10) for e in pings)
    assert ledger.spent == pytest.approx(len(fast.kws) * INPUT + len(alt.kws) * PING_P / 10)
    assert ledger.held == 0


@pytest.mark.parametrize("ping_exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("hung")])
def test_ping_that_never_left_or_hung_stays_free(monkeypatch, ping_exc):
    ledger, fast, alt, calls = _pinged(monkeypatch, ping_exc)
    assert alt.kws and not [e for e in calls if e["model"] == "alt"]
    assert ledger.spent == pytest.approx(len(fast.kws) * INPUT)

import math
import asyncio
import random
from types import SimpleNamespace

import httpx
import openai
import pytest

from harness import llm as llm_mod
from harness import config
from harness.budget import TokenLedger
from harness.config import TIMEOUT_CHARGE

CHARGE_16 = 17  # a timed-out call of "hi" with a limit of 16: what it can cost at most, far below TIMEOUT_CHARGE
DEAD_S = 300  # the default is the run deadline: these tests want the limit within reach
# the input charge of a failed "hi" call and of a failed ping: chars / 3, the conservative estimate of round 15
HI_INPUT = math.ceil(len(str([{"role": "user", "content": "hi"}])) / 3) / 10
PING_INPUT = math.ceil(len(str(llm_mod._PING)) / 3) / 10
EMPTY_INPUT = math.ceil(len(str([{"role": "user", "content": ""}])) / 3) / 10


@pytest.fixture(autouse=True)
def _dead_limit(monkeypatch):
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", str(DEAD_S))
from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable, ModelFailing, ProxyDown, chat_or_alt


class FakeModel:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.last_messages = None
        self.limits = []  # max_tokens of each request: a ping asks for 1

    async def invoke(self, messages, **kw):
        self.calls += 1
        self.limits.append(kw.get("max_tokens"))
        self.last_messages = messages
        return await self.behavior()


def make(behavior, timeout=180, monkeypatch=None):
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger, timeout=timeout)
    fake = FakeModel(behavior)
    c._model = lambda name: fake
    c.probe_models = ()  # a proxy with this one model: no other to ask
    return c, fake, ledger


def run(c):
    return asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _fake_time(monkeypatch, c):
    """Sleeping only advances the client's clock. For sequential calls: parallel sleepers would add up."""
    clock, sleeps, real_sleep = Clock(), [], asyncio.sleep

    async def fake_sleep(s):
        sleeps.append(s)
        clock.t += s
        await real_sleep(0)
    c._now = clock
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return clock, sleeps


def _ok(p=10, c=1, **usage):
    return SimpleNamespace(content="ok", usage_metadata=SimpleNamespace(input_tokens=p, output_tokens=c, **usage))


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
    assert fake.calls == 1 and ledger.spent == CHARGE_16


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


def test_429_waits_long_enough_for_account_ceiling(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)

    async def always():
        raise HTTPErr(429)
    c, fake, ledger = make(always)
    with pytest.raises(llm_mod.LLMUnavailable):
        run(c)
    assert sleeps == [2, 4, 8, 16, 30] and sum(sleeps) >= 40


def _wrapped(exc):
    """Mimic openjiuwen: ModelError raised `from` the provider exception."""
    try:
        raise exc
    except Exception as e:
        try:
            raise RuntimeError("openAI API async invoke error") from e
        except RuntimeError as w:
            return w


def _raiser(exc_factory):
    async def bad():
        raise _wrapped(exc_factory())
    return bad


def _failed_input(*fakes):
    """What the failed requests of these fakes cost: their input, as they reached the proxy (a ping asks for 1 token)."""
    return sum(PING_INPUT if lim == 1 else HI_INPUT for f in fakes for lim in f.limits)


def _no_sleep(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return sleeps


def test_403_marks_exhausted():
    c, fake, ledger = make(_raiser(lambda: HTTPErr(403)))
    with pytest.raises(BudgetExhausted):
        run(c)
    assert ledger.exhausted and fake.calls == 1


@pytest.mark.parametrize("status", [400, 404, 422])
def test_4xx_never_exhausts(status):
    c, fake, ledger = make(_raiser(lambda: HTTPErr(status)))
    for _ in range(5):
        with pytest.raises(LLMUnavailable):
            run(c)
    # a 404 is tried once more on the other spelling of the base URL (once per run); each try is charged its input
    assert not ledger.exhausted and fake.calls == (6 if status == 404 else 5)
    assert ledger.spent == pytest.approx(HI_INPUT * fake.calls)


def _req():
    return httpx.Request("POST", "http://x/v1/chat/completions")


def test_real_httpx_connect_error_waits_for_the_proxy(monkeypatch):
    """A connection refused is an outage like a 5xx, but free: it never left. Retried until the proxy is dead."""
    async def bad():
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError as ce:
            try:
                raise openai.APIConnectionError(request=_req()) from ce
            except openai.APIConnectionError as ae:
                raise _wrapped(ae)
    c, fake, ledger = make(bad)
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        run(c)
    assert clock.t - t0 >= DEAD_S and ledger.spent == 0 and ledger.exhausted


def test_real_remote_protocol_error_not_retried_but_charged():
    async def bad():
        try:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        except httpx.RemoteProtocolError as pe:
            try:
                raise openai.APIConnectionError(request=_req()) from pe
            except openai.APIConnectionError as ae:
                raise _wrapped(ae)
    c, fake, ledger = make(bad)
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and ledger.spent == CHARGE_16


def test_real_openai_api_timeout_charged_not_retried():
    c, fake, ledger = make(_raiser(lambda: openai.APITimeoutError(request=_req())))
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and ledger.spent == CHARGE_16


def test_real_openai_status_errors():
    resp = httpx.Response(402, request=_req())
    c, fake, ledger = make(_raiser(lambda: openai.APIStatusError("pay", response=resp, body=None)))
    with pytest.raises(BudgetExhausted):
        run(c)
    assert ledger.exhausted


def test_missing_usage_charges_estimate():
    async def ok():
        return SimpleNamespace(content="x", usage_metadata=None)
    c, fake, ledger = make(ok)
    r = run(c)
    assert r.completion_tokens == 16 and ledger.spent > 0


def test_status_regex_needs_word_boundary():
    assert llm_mod._status_of(RuntimeError("error 4000 tokens")) is None
    assert llm_mod._status_of(RuntimeError("Error code: 429 - slow down")) == 429


def test_openjiuwen_logs_silent_on_stdout(capfd):
    from openjiuwen.core.common.logging import llm_logger
    async def ok():
        llm_logger.info("PROBE-INFO full prompt")
        llm_logger.error("PROBE-ERROR")
        return SimpleNamespace(content="x", usage_metadata=SimpleNamespace(input_tokens=1, output_tokens=1))
    llm_mod._quiet_openjiuwen_logs()  # what the first real _model() call does
    c, fake, ledger = make(ok)
    run(c)
    out = capfd.readouterr()
    assert "PROBE" not in out.out and "PROBE" not in out.err


def test_quiet_logs_never_raises(monkeypatch):
    import openjiuwen.core.common.logging.log_config as lc

    def boom(cfg):
        raise OSError("unwritable")
    monkeypatch.setattr(lc, "configure_log_config", boom)
    llm_mod._quiet_openjiuwen_logs()  # must not raise


def test_log_dir_from_env(monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_DIR", "/tmp/hx")
    assert llm_mod._quiet_log_config()["log_path"].replace("\\", "/") == "/tmp/hx/openjiuwen-logs"


def test_concurrency_env_override(monkeypatch):
    import importlib
    from harness import config
    for val, want in (("5", 5), ("abc", 3), ("0", 3), ("", 3)):
        monkeypatch.setenv("HARNESS_LLM_CONCURRENCY", val)
        importlib.reload(config)
        assert config.LLM_CONCURRENCY == want
    monkeypatch.delenv("HARNESS_LLM_CONCURRENCY")
    importlib.reload(config)
    assert config.LLM_CONCURRENCY == 3


def _bad_request():
    return HTTPErr(400)


def test_structured_400_retries_without_and_remembers():
    seen = []

    async def fake_invoke(messages, **kw):
        seen.append(dict(kw))
        if "response_format" in kw:
            raise _bad_request()
        return SimpleNamespace(content="ok", usage_metadata=SimpleNamespace(input_tokens=10, output_tokens=5))

    c, fake, ledger = make(None)
    fake.invoke = fake_invoke
    r = asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16,
                           response_format={"type": "json_object"}, extra_body={"x": 1}))
    assert r.text == "ok" and c.structured_ok is False
    assert "response_format" not in seen[-1] and "extra_body" not in seen[-1]
    assert ledger.spent == pytest.approx(r.cost + HI_INPUT)  # the rejected try went out: its input is charged
    asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16, response_format={"type": "json_object"}))
    assert len(seen) == 3 and "response_format" not in seen[-1]


def test_400_without_structured_is_unavailable_charged_its_input():
    async def bad():
        raise _bad_request()
    c, fake, ledger = make(bad)
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and c.structured_ok is True and ledger.spent == pytest.approx(HI_INPUT)


def test_structured_400_both_ways_keeps_flag():
    async def bad():
        raise _bad_request()
    c, fake, ledger = make(bad)
    with pytest.raises(LLMUnavailable):
        asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16,
                           response_format={"type": "json_object"}))
    assert fake.calls == 2 and c.structured_ok is True


# ---------------------------------------------------------------- proxy outages (C2)

def _script(*steps):
    """Each step is an exception to raise or a reply to return."""
    seq = iter(steps)

    async def beh():
        e = next(seq)
        if isinstance(e, BaseException):
            raise _wrapped(e)
        return e
    return beh


def test_5xx_retried_after_shared_pause_and_charged_its_input(monkeypatch):
    c, fake, ledger = make(_script(HTTPErr(500), _ok()))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    assert run(c).text == "ok"
    assert fake.calls == 2 and ledger.spent == pytest.approx(2 + HI_INPUT)  # the served call 10/10 + 1, the 500's input
    assert clock.t - t0 == config.RETRY_DELAYS[0]


def test_5xx_alone_never_kills_before_dead_s(monkeypatch):
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        run(c)
    assert clock.t - t0 >= DEAD_S and ledger.exhausted and ledger.spent == pytest.approx(HI_INPUT * fake.calls)
    assert 6 <= fake.calls <= 15  # the pause grows: not a hammering of the proxy


def test_pause_grows_up_to_a_minute(monkeypatch):
    seen = []
    c, fake, ledger = make(_raiser(lambda: HTTPErr(503)))
    clock, _ = _fake_time(monkeypatch, c)

    async def beh():
        seen.append(clock.t)
        raise _wrapped(HTTPErr(503))
    fake.behavior = beh
    with pytest.raises(BudgetExhausted):
        run(c)
    gaps = [b - a for a, b in zip(seen, seen[1:])]
    assert gaps == sorted(gaps) and gaps[0] == 2 and gaps[-1] == 60


def test_dead_s_knob(monkeypatch):
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", "10")
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        run(c)
    assert 10 <= clock.t - t0 < 60


def test_outage_of_four_minutes_is_survived(monkeypatch):
    """Failures for less than HARNESS_PROXY_DEAD_S: the held call is answered once the proxy is back."""
    c, fake, ledger = make(None)
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t

    async def beh():
        if clock.t - t0 < 240:
            raise _wrapped(HTTPErr(502))
        return _ok()
    fake.behavior = beh
    assert run(c).text == "ok"
    assert not ledger.exhausted and ledger.spent == pytest.approx(2 + HI_INPUT * (fake.calls - 1))


def test_success_ends_the_outage(monkeypatch):
    """A new outage has a fresh clock and starts from the shortest pause."""
    c, fake, ledger = make(None)
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    up = {"from": 200, "to": 10 ** 9}

    async def beh():
        if clock.t - t0 < up["from"] or clock.t - t0 >= up["to"]:
            raise _wrapped(HTTPErr(500))
        return _ok()
    fake.behavior = beh
    assert run(c).text == "ok"
    # the second outage lasts 200 s too: together 400 s, each alone below the limit
    now = clock.t - t0
    up.update({"from": now + 200, "to": 10 ** 9})
    assert run(c).text == "ok"
    assert not ledger.exhausted


def test_parallel_failures_of_one_drop_count_once(monkeypatch):
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.2,0.4,0.8,1.6")
    n = {"i": 0}
    release = None

    async def beh():
        n["i"] += 1
        if n["i"] <= 3:  # all three in flight together, all fail together
            if n["i"] == 3:
                release.set()
            await release.wait()
            raise _wrapped(HTTPErr(500))
        return _ok()
    c, fake, ledger = make(beh)

    async def go():
        nonlocal release
        release = asyncio.Event()
        t = c._now()
        rs = await asyncio.gather(*(c.chat("m", [{"role": "user", "content": "hi"}], 16) for _ in range(3)))
        return rs, c._now() - t
    rs, took = asyncio.run(go())
    assert [r.text for r in rs] == ["ok"] * 3
    assert took < 0.55  # one step of the pause, not 0.2 + 0.4 + 0.8


def test_wait_on_the_pause_follows_a_cancelled_unit():
    """The scheduler's deadline cancels the unit: a call waiting out the pause must stop at once."""
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))

    async def go():
        await asyncio.wait_for(c.chat("m", [{"role": "user", "content": "hi"}], 16), 0.2)
    t = c._now()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert c._now() - t < 1.5 and ledger.held == 0 and not ledger.exhausted


def test_call_that_others_outlive_gives_up(monkeypatch):
    """Only this request fails while the proxy serves others: not an outage, the caller moves on."""
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))
    good = FakeModel(_script(*[_ok() for _ in range(50)]))
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.01")
    c._model = lambda name: fake if name == "bad" else good

    async def others():
        for _ in range(50):
            await c.chat("good", [{"role": "user", "content": "hi"}], 16)
            await asyncio.sleep(0.005)

    async def go():
        bg = asyncio.create_task(others())
        try:
            return await c.chat("bad", [{"role": "user", "content": "hi"}], 16)
        finally:
            bg.cancel()
    with pytest.raises(LLMUnavailable):
        asyncio.run(go())
    assert not ledger.exhausted and fake.calls <= 4 and ledger.held == 0


def test_402_during_outage_exhausts_at_once(monkeypatch):
    c, fake, ledger = make(_script(HTTPErr(500), HTTPErr(402)))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        run(c)
    assert ledger.exhausted and fake.calls == 2 and clock.t - t0 < 10


def test_timeouts_are_charged_raise_at_once_and_reach_dead(monkeypatch):
    async def hang():
        await asyncio.Event().wait()
    c, fake, ledger = make(hang, timeout=0.01)
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(LLMUnavailable):
        run(c)
    assert ledger.spent == CHARGE_16 and not ledger.exhausted and fake.calls == 1
    clock.t = t0 + 299
    with pytest.raises(LLMUnavailable):
        run(c)
    assert not ledger.exhausted and ledger.spent == 2 * CHARGE_16
    with pytest.raises(BudgetExhausted):  # the pause set by the last failure brings it past the limit
        run(c)
    assert ledger.exhausted and ledger.spent == 3 * CHARGE_16


def test_next_call_after_a_timeout_waits_the_pause(monkeypatch):
    async def hang():
        await asyncio.Event().wait()
    c, fake, ledger = make(hang, timeout=0.01)
    clock, _ = _fake_time(monkeypatch, c)
    with pytest.raises(LLMUnavailable):
        run(c)
    t = clock.t
    fake.behavior = _script(_ok())
    assert run(c).text == "ok"
    assert clock.t - t == config.RETRY_DELAYS[0]


def test_late_failure_after_an_idle_stretch_is_a_new_outage(monkeypatch):
    """The clock starts at the first failure, not at the last success of long ago."""
    c, fake, ledger = make(_script(_ok(), HTTPErr(500), _ok()))
    clock, _ = _fake_time(monkeypatch, c)
    assert run(c).text == "ok"
    clock.t += 10000
    assert run(c).text == "ok"
    assert not ledger.exhausted


def test_429_is_not_an_outage(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    c, fake, ledger = make(_raiser(lambda: HTTPErr(429)))
    with pytest.raises(LLMUnavailable):
        run(c)
    assert sleeps == list(config.RETRY_DELAYS) and not ledger.exhausted
    assert c._down_since is None


# ---------------------------------------------------------------- per-call budget (I3)

def _budgeted(budget, reserve, behavior=None):
    ledger = TokenLedger(budget, reserve)
    c = LLMClient("http://x/v1", "k", ledger)
    seen = []

    async def default():
        return _ok(p=0, c=seen[-1]["max_tokens"])

    fake = FakeModel(behavior or default)
    real = fake.invoke

    async def invoke(messages, **kw):
        seen.append(kw)
        return await real(messages, **kw)
    fake.invoke = invoke
    c._model = lambda name: fake
    return c, fake, ledger, seen


def _ask(c, max_tokens, chars=0):
    return c.chat("m", [{"role": "user", "content": "x" * chars}], max_tokens)


def test_call_that_fits_keeps_its_max_tokens():
    c, fake, ledger, seen = _budgeted(10000, 100)
    asyncio.run(_ask(c, 3000))
    assert seen[0]["max_tokens"] == 3000 and ledger.held == 0


def test_max_tokens_cut_to_what_is_left():
    c, fake, ledger, seen = _budgeted(1000, 100)
    asyncio.run(_ask(c, 3000, 4000))  # about 1000 prompt tokens: 100 by the /10 rule
    assert 750 <= seen[0]["max_tokens"] <= 800  # 1000 - reserve 100 - prompt ~100
    assert ledger.spent <= ledger.budget and ledger.held == 0


def test_call_below_the_minimum_is_not_made():
    c, fake, ledger, seen = _budgeted(1000, 100)
    ledger.charge(0, 800)  # 100 left over the reserve; a prompt of 100 by the /10 rule leaves nothing
    with pytest.raises(LLMUnavailable):
        asyncio.run(_ask(c, 400, 4000))
    assert fake.calls == 0 and not ledger.exhausted and ledger.held == 0


def test_small_call_passes_when_it_fits_even_below_the_minimum():
    c, fake, ledger, seen = _budgeted(1000, 100)
    ledger.charge(0, 880)  # 20 left over the reserve; arbitration asks for 8 tokens
    asyncio.run(_ask(c, 8))
    assert fake.calls == 1 and seen[0]["max_tokens"] == 8


def test_parallel_calls_cannot_overspend_together():
    c, fake, ledger, seen = _budgeted(3000, 30)

    async def go():
        return await asyncio.gather(*(_ask(c, 3000) for _ in range(3)), return_exceptions=True)
    rs = asyncio.run(go())
    assert ledger.spent <= 3000 and ledger.held == 0
    assert any(isinstance(r, LLMUnavailable) for r in rs)


def test_call_waits_for_in_flight_holds_instead_of_cutting():
    """Another call holds the budget for a moment: its real cost is small, so this one gets the full limit."""
    async def beh():
        await asyncio.sleep(0.05)
        return _ok(p=0, c=10)
    c, fake, ledger, seen = _budgeted(3100, 30, beh)

    async def go():
        await asyncio.gather(_ask(c, 3000), _ask(c, 3000))
    asyncio.run(go())
    assert [k["max_tokens"] for k in seen] == [3000, 3000]


def test_retry_after_5xx_takes_a_fresh_hold(monkeypatch):
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.01")
    n = {"i": 0}

    async def beh():
        n["i"] += 1
        if n["i"] == 1:
            raise _wrapped(HTTPErr(500))
        return _ok(p=0, c=5)
    c, fake, ledger, seen = _budgeted(1000, 100, beh)
    asyncio.run(_ask(c, 16))
    assert ledger.held == 0 and ledger.spent == pytest.approx(5 + EMPTY_INPUT)  # the served call, the 500's input


def test_reasoning_outside_completion_is_charged():
    async def beh():
        return _ok(p=100, c=20, total_tokens=2120, reasoning_tokens=2000)
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    r = asyncio.run(_ask(c, 400))
    assert r.completion_tokens == 2020 and ledger.spent == 2030


def test_reasoning_inside_completion_is_not_charged_twice():
    async def beh():
        return _ok(p=100, c=2388, total_tokens=2488, reasoning_tokens=2000)
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    r = asyncio.run(_ask(c, 3000))
    assert r.completion_tokens == 2388 and ledger.spent == 2398


def test_reasoning_above_completion_without_total_is_outside():
    async def beh():
        return _ok(p=100, c=20, reasoning_tokens=2000)
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    r = asyncio.run(_ask(c, 400))
    assert r.completion_tokens == 2020


def test_call_that_does_not_fit_is_cut_at_once_after_overlapping_calls():
    """The review's hang: nothing in flight, yet the call polled forever on a residue in held."""
    async def invoke(messages, max_tokens=None, **kw):
        seen.append(max_tokens)
        await asyncio.sleep(0.002 * (1 + len(seen) % 3))
        return _ok(p=100, c=50)
    seen = []
    ledger = TokenLedger(10000, 100)
    c = LLMClient("http://x", "k", ledger, max_concurrency=3, timeout=5)
    c._model = lambda name: SimpleNamespace(invoke=invoke)
    rnd = random.Random(3)

    async def go():
        for _ in range(40):
            await asyncio.gather(*(c.chat("m", [{"role": "user", "content": "x" * rnd.randint(300, 9000)}],
                                          rnd.choice([250, 400, 600])) for _ in range(3)))
        assert ledger.held == 0
        ledger.spent = 9700.0
        await asyncio.wait_for(c.chat("m", [{"role": "user", "content": "hi"}], 3000), 2)
    asyncio.run(go())
    assert 150 <= seen[-1] <= 200


def test_reasoning_call_is_not_cut_below_what_it_can_use():
    c, fake, ledger, seen = _budgeted(2000, 20)
    ledger.charge(0, 1000)  # 980 left: a plain call would still go with that limit
    with pytest.raises(LLMUnavailable):
        asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 3000, reasoning_effort="low"))
    assert fake.calls == 0 and ledger.spent == 1000 and ledger.held == 0
    asyncio.run(_ask(c, 3000))
    assert 900 <= seen[0]["max_tokens"] <= 980


def _cut_reply(finish_reason):
    async def beh():
        r = _ok(p=0, c=5)
        r.finish_reason = finish_reason
        return r
    return beh


def test_reply_stopped_by_a_cut_limit_is_marked():
    c, fake, ledger, seen = _budgeted(1000, 100, _cut_reply("length"))
    assert asyncio.run(_ask(c, 3000)).truncated is True


@pytest.mark.parametrize("finish_reason, asked", [("stop", 3000), ("length", 400)])
def test_other_replies_are_not_marked_as_cut(finish_reason, asked):
    c, fake, ledger, seen = _budgeted(1000, 100, _cut_reply(finish_reason))
    assert asyncio.run(_ask(c, asked)).truncated is False


# ---------------------------------------------------------------- one model failing is not the proxy dying

def _models(monkeypatch, timeout=180, **behaviors):
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger, timeout=timeout)
    fakes = {name: FakeModel(b) for name, b in behaviors.items()}
    c._model = lambda name: fakes[name]
    c.probe_models = tuple(behaviors)
    clock, _ = _fake_time(monkeypatch, c)
    return c, fakes, ledger, clock


def _say(c, model):
    return asyncio.run(c.chat(model, [{"role": "user", "content": "hi"}], 16))


def _down():
    return _raiser(lambda: HTTPErr(503))


def _up():
    async def beh():
        return _ok()
    return beh


async def _hang():
    await asyncio.Event().wait()


def test_model_is_out_when_it_fails_after_another_answered_a_ping(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_up())
    t0 = clock.t
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    assert fakes["fast"].calls == 3  # fail, pause, fail, ping elsewhere, one more try now that the proxy is known alive
    assert fakes["alt"].calls == 1 and fakes["alt"].last_messages != [{"role": "user", "content": "hi"}]
    assert clock.t - t0 == config.RETRY_DELAYS[0] and not ledger.exhausted


def test_ping_is_charged_like_any_call_and_asks_for_one_token(monkeypatch):
    seen = {}

    async def invoke(messages, **kw):
        seen.update(kw)
        return _ok(p=10, c=1)
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=None)
    fakes["alt"].invoke = invoke
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    assert seen == {"max_tokens": 1} and ledger.spent == pytest.approx(_failed_input(fakes["fast"]) + 2)  # the 503s, the ping


def test_blip_that_ends_during_the_ping_stays_on_the_same_model(monkeypatch):
    """The ping shows only that the proxy is back: a model that answers the next try was never out."""
    c, fakes, ledger, clock = _models(monkeypatch, fast=None, alt=_up())
    n = {"i": 0}

    async def fast():
        n["i"] += 1
        if n["i"] <= 2:
            raise _wrapped(HTTPErr(503))
        return _ok()
    fakes["fast"].behavior = fast
    assert _say(c, "fast").text == "ok"
    assert fakes["fast"].calls == 3 and c._health["fast"].step == 0


def test_model_known_to_fail_is_skipped_during_its_pause_and_probed_after(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_up())
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    calls = fakes["fast"].calls
    t = clock.t
    with pytest.raises(ModelFailing):  # at once: no request, no waiting
        _say(c, "fast")
    assert fakes["fast"].calls == calls and clock.t == t
    clock.t += 100  # the pause is over: it gets a new chance
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    assert fakes["fast"].calls > calls
    clock.t += 100
    fakes["fast"].behavior = _up()
    assert _say(c, "fast").text == "ok"
    calls = fakes["fast"].calls
    assert _say(c, "fast").text == "ok" and fakes["fast"].calls == calls + 1  # healthy: no pause left behind


def test_pause_of_one_model_does_not_hold_another(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_up())
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    t = clock.t
    assert _say(c, "alt").text == "ok"
    assert clock.t == t


def test_only_one_call_at_a_time_probes_a_model_that_is_out(monkeypatch):
    """A hanging model must not take all the slots each time its pause ends."""
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_up())

    async def go():
        with pytest.raises(ModelFailing):
            await c.chat("fast", [{"role": "user", "content": "hi"}], 16)
        clock.t += 100
        fakes["fast"].behavior = _hang
        calls = fakes["fast"].calls
        probe = asyncio.create_task(c.chat("fast", [{"role": "user", "content": "hi"}], 16))
        await asyncio.sleep(0)
        for _ in range(5):
            await asyncio.sleep(0)
        with pytest.raises(ModelFailing):
            await c.chat("fast", [{"role": "user", "content": "hi"}], 16)
        assert fakes["fast"].calls == calls + 1
        probe.cancel()
    asyncio.run(go())


def test_all_models_failing_is_an_outage_the_call_waits_out(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, fast=None, alt=_down())
    t0 = clock.t

    async def fast():
        if clock.t - t0 < 100:
            raise _wrapped(HTTPErr(503))
        return _ok()
    fakes["fast"].behavior = fast
    assert _say(c, "fast").text == "ok"
    assert clock.t - t0 >= 100 and fakes["fast"].calls > 2 and fakes["alt"].calls >= 2 and not ledger.exhausted


def test_model_that_answered_nothing_yet_is_not_taken_for_alive(monkeypatch):
    """The ladder's N3: an outage must not walk a call down to a model nobody has heard from."""
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_down(), reason=_down())
    with pytest.raises(BudgetExhausted):
        _say(c, "fast")
    assert ledger.spent == pytest.approx(_failed_input(*fakes.values())) and clock.t - 1000 >= DEAD_S  # pings too


def test_success_elsewhere_ends_the_wait_on_a_pause():
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger)
    bad, good = FakeModel(_down()), FakeModel(_up())
    c._model = lambda name: bad if name == "bad" else good
    c.probe_models = ()

    async def go():
        t = c._now()
        call = asyncio.create_task(c.chat("bad", [{"role": "user", "content": "hi"}], 16))
        await asyncio.sleep(0.05)
        c._health["bad"].pause_until = c._now() + 50  # one long pause into the wait
        await c.chat("good", [{"role": "user", "content": "hi"}], 16)
        with pytest.raises(ModelFailing):
            await call
        return c._now() - t
    assert asyncio.run(go()) < 10


def test_timeout_moves_the_call_on_when_the_proxy_answers(monkeypatch):
    """N1: a hanging model is failing like one that returns 503, and the call is charged as before."""
    c, fakes, ledger, clock = _models(monkeypatch, timeout=0.01, fast=_hang, alt=_up())
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    assert fakes["fast"].calls == 1 and ledger.spent == CHARGE_16 + 2  # the timeout, the ping
    assert not ledger.exhausted
    with pytest.raises(ModelFailing):  # skipped during its pause
        _say(c, "fast")
    assert fakes["fast"].calls == 1
    clock.t += 1000
    fakes["fast"].behavior = _up()
    assert _say(c, "fast").text == "ok"  # one timeout never writes a model off


def test_timeout_while_no_model_answers_is_a_proxy_outage(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, timeout=0.01, fast=_hang, alt=_hang)
    with pytest.raises(ProxyDown):
        _say(c, "fast")
    assert ledger.spent == CHARGE_16  # the ping that hung is not charged: one token at most


def test_one_failure_of_alt_does_not_write_it_off(monkeypatch):
    """N2: FAST is down, one drop on ALT: it is asked again when its pause ends, and the run goes on."""
    n = {"i": 0}

    async def alt():
        n["i"] += 1
        if n["i"] == 1:
            raise _wrapped(HTTPErr(503))
        return _ok()
    c, fakes, ledger, clock = _models(monkeypatch, **{config.MODEL_FAST: _down(), config.MODEL_ALT: alt})
    msgs = [{"role": "user", "content": "hi"}]
    assert asyncio.run(chat_or_alt(c, config.MODEL_FAST, msgs, 16)).text == "ok"
    assert not ledger.exhausted


def test_proxy_is_alive_while_any_model_answers(monkeypatch):
    """FAST down for much longer than the dead limit: the answers of ALT keep the run going, FAST keeps being tried."""
    c, fakes, ledger, clock = _models(monkeypatch, **{config.MODEL_FAST: _down(), config.MODEL_ALT: _up()})
    t0 = clock.t
    msgs = [{"role": "user", "content": "hi"}]
    while clock.t - t0 < 3 * DEAD_S:
        assert asyncio.run(chat_or_alt(c, config.MODEL_FAST, msgs, 16)).text == "ok"
        clock.t += 10
    assert not ledger.exhausted and fakes[config.MODEL_FAST].calls >= 10


def test_proxy_is_dead_when_no_model_answers(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, **{config.MODEL_FAST: _down(), config.MODEL_ALT: _down(),
                                                      config.MODEL_REASON: _down()})
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        asyncio.run(chat_or_alt(c, config.MODEL_FAST, [{"role": "user", "content": "hi"}], 16))
    assert clock.t - t0 >= DEAD_S and ledger.exhausted and ledger.spent == pytest.approx(_failed_input(*fakes.values()))
    assert fakes[config.MODEL_FAST].calls >= 2 and fakes[config.MODEL_ALT].calls >= 2


def test_chat_or_alt_prefers_the_asked_model(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, **{config.MODEL_FAST: _up(), config.MODEL_ALT: _up()})
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(chat_or_alt(c, config.MODEL_FAST, msgs, 16))
    assert fakes[config.MODEL_FAST].calls == 1 and fakes[config.MODEL_ALT].calls == 0
    asyncio.run(chat_or_alt(c, config.MODEL_ALT, msgs, 16))
    assert fakes[config.MODEL_FAST].calls == 1 and fakes[config.MODEL_ALT].calls == 1


def _all_models(**behaviors):
    return {m: behaviors.get(m) for m in (config.MODEL_FAST, config.MODEL_ALT, config.MODEL_REASON)}


class PingsPass(FakeModel):
    """Real calls follow the behavior, a 1-token request always gets through."""
    async def invoke(self, messages, **kw):
        if kw.get("max_tokens") == 1:
            return _ok()
        return await super().invoke(messages, **kw)


def _hi(c, model=config.MODEL_FAST):
    return chat_or_alt(c, model, [{"role": "user", "content": "hi"}], 16)


def test_chat_or_alt_asking_for_alt_moves_to_fast_when_alt_fails(monkeypatch):
    """N5: no special case for ALT: it is one more model in the same order."""
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    fakes[config.MODEL_ALT].behavior = _down()
    fakes[config.MODEL_FAST].behavior = fakes[config.MODEL_REASON].behavior = _up()
    assert asyncio.run(_hi(c, config.MODEL_ALT)).text == "ok"
    assert not ledger.exhausted and fakes[config.MODEL_REASON].calls == 0


def test_chat_or_alt_goes_on_to_reason_when_fast_and_alt_fail(monkeypatch):
    """M9: the order is fixed, whichever model was asked for."""
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    fakes[config.MODEL_FAST].behavior = fakes[config.MODEL_ALT].behavior = _down()
    fakes[config.MODEL_REASON].behavior = _up()
    assert asyncio.run(_hi(c)).text == "ok"
    assert not ledger.exhausted


def test_chat_or_alt_waits_only_when_every_model_fails_then_asks_for_real(monkeypatch):
    """N4: only pings pass for a while. The call neither gives up nor takes a ping for an answer."""
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    t0 = clock.t

    async def real():
        if clock.t - t0 < 100:
            raise _wrapped(HTTPErr(503))
        return _ok()
    for m in fakes:
        fakes[m] = PingsPass(real)
    assert asyncio.run(_hi(c)).text == "ok"
    assert clock.t - t0 >= 100 and not ledger.exhausted


def test_chat_or_alt_ends_when_the_proxy_is_declared_dead(monkeypatch):
    """Even if a ping passes all along: only real answers keep the proxy alive."""
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    for m in fakes:
        fakes[m] = PingsPass(_down())
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        asyncio.run(_hi(c))
    assert clock.t - t0 >= DEAD_S and ledger.exhausted


def test_chat_or_alt_follows_a_cancelled_unit():
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger)
    c.probe_models = ()
    c._model = lambda name: FakeModel(_down())

    async def go():
        await asyncio.wait_for(_hi(c), 0.3)
    t = c._now()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert c._now() - t < 1.5 and ledger.held == 0


def test_a_swallowed_cancel_does_not_extend_the_wait():
    """M8: on 3.11 wait_for can lose a cancel when the request ends in the same iteration of the loop."""
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger)
    asks = {"n": 0}

    async def chat(model, messages, max_tokens, **extra):
        asks["n"] += 1
        if asks["n"] > 60:
            return "the wait was never ended"
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass  # what wait_for does in that case
        raise ModelFailing("503")
    c.chat = chat

    async def go():
        return await asyncio.wait_for(_hi(c), 0.1)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert asks["n"] < 10


def test_wait_for_model_always_yields_and_ignores_models_with_no_pause():
    """M7: a model with no pause (-inf) in the min made it return at once, and the ladder spun with the loop frozen."""
    c = LLMClient("http://x/v1", "k", TokenLedger(100000, 1000))
    c._health["paused"].pause_until, c._health["paused"].step = c._now() + 0.3, 1
    beats = []

    async def heart():
        while True:
            beats.append(1)
            await asyncio.sleep(0.01)

    async def go():
        h = asyncio.create_task(heart())
        await asyncio.sleep(0)
        beats.clear()
        t = c._now()
        await c.wait_for_model(["paused", "never_failed"])
        took = c._now() - t
        h.cancel()
        return took
    took = asyncio.run(go())
    assert took >= 0.2 and len(beats) >= 2


def test_wait_for_model_yields_when_there_is_nothing_to_wait_for():
    c = LLMClient("http://x/v1", "k", TokenLedger(100000, 1000))
    beats = []

    async def heart():
        while True:
            beats.append(1)
            await asyncio.sleep(0)

    async def go():
        h = asyncio.create_task(heart())
        await asyncio.sleep(0)
        beats.clear()
        await c.wait_for_model(["never_failed"])
        h.cancel()
    asyncio.run(go())
    assert beats


def test_chat_or_alt_gives_up_on_what_waiting_cannot_change(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, **{config.MODEL_FAST: _raiser(lambda: HTTPErr(404))})
    with pytest.raises(LLMUnavailable):
        asyncio.run(chat_or_alt(c, config.MODEL_FAST, [{"role": "user", "content": "hi"}], 16))
    assert fakes[config.MODEL_FAST].calls == 2  # the second one is the other spelling of the base URL


def test_ping_leaves_the_health_of_the_pinged_model_and_the_dead_clock_alone(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, fast=_down(), alt=_up())
    c._proxy_failed("alt", clock.t - 1)
    alt_before = (c._health["alt"].step, c._health["alt"].pause_until)
    clock.t = c._health["alt"].pause_until  # the pause of alt is over, the ping may go
    since = c._down_since
    with pytest.raises(ModelFailing):
        _say(c, "fast")
    assert c._health["alt"].step == alt_before[0] and c._down_since == since and c._served == 0


def test_timeout_charge_is_at_most_what_the_call_can_cost():
    async def hang():
        await asyncio.Event().wait()
    c, fake, ledger, seen = _budgeted(1000000, 100, hang)
    c.timeout = 0.01
    c.probe_models = ()
    with pytest.raises(LLMUnavailable):
        asyncio.run(_ask(c, 400, 400))  # ~100 prompt tokens: 10 by the /10 rule
    assert 400 < ledger.spent < 420
    with pytest.raises(LLMUnavailable):
        asyncio.run(_ask(c, 3000))
    assert ledger.spent - 420 < TIMEOUT_CHARGE <= ledger.spent - 400  # capped by the flat amount


def test_failure_while_building_the_model_client_is_no_answer_not_a_crash():
    """The flake of the blip test: the exception left the unit, so its tasks were lost with no retry."""
    ledger = TokenLedger(100000, 1000)
    c = LLMClient("http://x/v1", "k", ledger)
    fake = FakeModel(_up())
    left = {"n": 1}

    def model(name):
        if left["n"]:
            left["n"] -= 1
            raise PermissionError("log file is locked")
        return fake
    c._model = model
    with pytest.raises(LLMUnavailable, match="log file is locked"):
        _say(c, "m")
    assert _say(c, "m").text == "ok" and ledger.held == 0


def test_outage_longer_than_the_old_limit_is_survived_by_default(monkeypatch):
    """N6: six minutes without an answer must not switch the LLM off for the rest of the hour."""
    monkeypatch.delenv("HARNESS_PROXY_DEAD_S")
    monkeypatch.delenv("HARNESS_DEADLINE_S", raising=False)
    c, fake, ledger = make(None)
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t

    async def beh():
        if clock.t - t0 < 360:
            raise _wrapped(HTTPErr(503))
        return _ok()
    fake.behavior = beh
    assert run(c).text == "ok"
    assert not ledger.exhausted


@pytest.mark.parametrize("usage", [
    SimpleNamespace(),
    SimpleNamespace(input_tokens=0, output_tokens=0),
    SimpleNamespace(input_tokens=None, output_tokens=None),
    SimpleNamespace(total_tokens=2000),
])
def test_reply_with_empty_usage_is_charged_the_estimate(usage):
    """N7: a reply was generated, so it cost something: the same estimate as with no usage at all."""
    async def beh():
        return SimpleNamespace(content="hello", usage_metadata=usage)
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    r = asyncio.run(_ask(c, 400, 400))
    assert ledger.spent == r.cost and r.completion_tokens == 400 and r.prompt_tokens > 0
    assert 400 < ledger.spent < 420


def test_empty_reply_with_empty_usage_costs_nothing():
    async def beh():
        return SimpleNamespace(content="", usage_metadata=SimpleNamespace(input_tokens=0, output_tokens=0))
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    assert asyncio.run(_ask(c, 400)).cost == 0 and ledger.spent == 0


def test_reported_usage_is_never_replaced_by_the_estimate():
    async def beh():
        return SimpleNamespace(content="hello", usage_metadata=SimpleNamespace(input_tokens=0, output_tokens=3))
    c, fake, ledger, seen = _budgeted(100000, 1000, beh)
    assert asyncio.run(_ask(c, 400)).cost == 3


def test_fallback_to_reason_gets_a_limit_and_effort_it_can_answer_with(monkeypatch):
    """A reasoning model spends a limit of 400 on reasoning and replies with nothing."""
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    fakes[config.MODEL_FAST].behavior = fakes[config.MODEL_ALT].behavior = _down()
    seen = {}

    async def invoke(messages, **kw):
        seen.update(kw)
        return _ok()
    fakes[config.MODEL_REASON].invoke = invoke
    r = chat_or_alt(c, config.MODEL_FAST, [{"role": "user", "content": "hi"}], 400, response_format={"type": "json_object"})
    assert asyncio.run(r).text == "ok"
    assert seen["max_tokens"] == config.REASON_FALLBACK_TOKENS >= 1500 and seen["reasoning_effort"] == "low"
    assert seen["response_format"] == {"type": "json_object"}


def test_fallback_to_reason_keeps_a_larger_limit(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    fakes[config.MODEL_FAST].behavior = fakes[config.MODEL_ALT].behavior = _down()
    seen = {}

    async def invoke(messages, **kw):
        seen.update(kw)
        return _ok()
    fakes[config.MODEL_REASON].invoke = invoke
    asyncio.run(chat_or_alt(c, config.MODEL_FAST, [{"role": "user", "content": "hi"}], 4000))
    assert seen["max_tokens"] == 4000


def test_other_models_keep_the_asked_limit_and_no_effort(monkeypatch):
    c, fakes, ledger, clock = _models(monkeypatch, **_all_models())
    fakes[config.MODEL_FAST].behavior = _down()
    seen = {}

    async def invoke(messages, **kw):
        seen.update(kw)
        return _ok()
    fakes[config.MODEL_ALT].invoke = invoke
    asyncio.run(chat_or_alt(c, config.MODEL_FAST, [{"role": "user", "content": "hi"}], 400))
    assert seen.get("max_tokens") == 400 and "reasoning_effort" not in seen


def test_noreason_without_usage_is_charged_by_reply_length():
    """Expert 2.5: len(reply)/3 with a margin of 1.5, not the whole limit; the key text is not needed for it."""
    async def ok():
        return SimpleNamespace(content="a" * 300, usage_metadata=None)
    c, fake, ledger = make(ok)
    r = asyncio.run(c.chat("m-noreason", [{"role": "user", "content": "hi"}], 400))
    assert r.completion_tokens == 150


def test_noreason_estimate_never_exceeds_the_limit():
    async def ok():
        return SimpleNamespace(content="a" * 3000, usage_metadata=None)
    c, fake, ledger = make(ok)
    r = asyncio.run(c.chat("m-noreason", [{"role": "user", "content": "hi"}], 100))
    assert r.completion_tokens == 100


def test_reasoning_model_without_usage_still_charged_the_limit():
    async def ok():
        return SimpleNamespace(content="a" * 30, usage_metadata=None)
    c, fake, ledger = make(ok)
    r = asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 400))
    assert r.completion_tokens == 400


@pytest.mark.parametrize("text", ["Привет, мир! " * 40, "你好世界，这是一个测试。" * 40])
def test_noreason_estimate_counts_utf8_bytes(text):
    async def ok():
        return SimpleNamespace(content=text, usage_metadata=None)
    c, fake, ledger = make(ok)
    r = asyncio.run(c.chat("m-noreason", [{"role": "user", "content": "hi"}], 100000))
    assert r.completion_tokens == math.ceil(len(text.encode()) / 3 * 1.5)
    assert r.completion_tokens >= len(text) * 0.8  # about a token per Cyrillic or Chinese char


@pytest.mark.parametrize("msg", [
    "FrameworkError: [181001] model call failed, reason: HTTPStatusError: 404 Not Found",
    "400 model max context length exceeded, invalid request",
    "400 max_tokens too large for model, unsupported parameter",
    "400 model is overloaded, not available right now",
])
def test_only_an_explicit_model_not_found_counts(msg):
    assert not llm_mod._MODEL_MISSING_RE.search(llm_mod._clean_msg(msg))


@pytest.mark.parametrize("msg", [
    "404 The model `x` does not exist",
    "400 model_not_found",
    "404 No endpoints found for model x",
    "400 Unknown model: x",
    "404 model 'x' not found",
])
def test_explicit_model_not_found_matches(msg):
    assert llm_mod._MODEL_MISSING_RE.search(llm_mod._clean_msg(msg))

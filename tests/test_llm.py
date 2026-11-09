import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from harness import llm as llm_mod
from harness import config
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
    assert not ledger.exhausted and ledger.spent == 0 and fake.calls == 5


def _req():
    return httpx.Request("POST", "http://x/v1/chat/completions")


def test_real_httpx_connect_error_waits_for_the_proxy(monkeypatch):
    """A connection refused is an outage like a 5xx: never charged, retried until the proxy is dead."""
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
    assert clock.t - t0 >= config.PROXY_DEAD_S and ledger.spent == 0 and ledger.exhausted


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
    assert fake.calls == 1 and ledger.spent == TIMEOUT_CHARGE


def test_real_openai_api_timeout_charged_not_retried():
    c, fake, ledger = make(_raiser(lambda: openai.APITimeoutError(request=_req())))
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and ledger.spent == TIMEOUT_CHARGE


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
    assert ledger.spent == r.cost
    asyncio.run(c.chat("m", [{"role": "user", "content": "hi"}], 16, response_format={"type": "json_object"}))
    assert len(seen) == 3 and "response_format" not in seen[-1]


def test_400_without_structured_is_unavailable_no_charge():
    async def bad():
        raise _bad_request()
    c, fake, ledger = make(bad)
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 1 and c.structured_ok is True and ledger.spent == 0


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


def test_5xx_retried_after_shared_pause_and_not_charged(monkeypatch):
    c, fake, ledger = make(_script(HTTPErr(500), _ok()))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    assert run(c).text == "ok"
    assert fake.calls == 2 and ledger.spent == 2  # only the served call: 10/10 + 1
    assert clock.t - t0 == config.RETRY_DELAYS[0]


def test_5xx_alone_never_kills_before_dead_s(monkeypatch):
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))
    clock, _ = _fake_time(monkeypatch, c)
    t0 = clock.t
    with pytest.raises(BudgetExhausted):
        run(c)
    assert clock.t - t0 >= config.PROXY_DEAD_S and ledger.exhausted and ledger.spent == 0
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
    assert not ledger.exhausted and ledger.spent == 2


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
    assert ledger.spent == TIMEOUT_CHARGE and not ledger.exhausted and fake.calls == 1
    clock.t = t0 + 299
    with pytest.raises(LLMUnavailable):
        run(c)
    assert not ledger.exhausted and ledger.spent == 2 * TIMEOUT_CHARGE
    with pytest.raises(BudgetExhausted):  # the pause set by the last failure brings it past the limit
        run(c)
    assert ledger.exhausted and ledger.spent == 3 * TIMEOUT_CHARGE


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
    assert ledger.held == 0 and ledger.spent == 5


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

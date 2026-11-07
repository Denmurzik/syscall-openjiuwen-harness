import asyncio
from types import SimpleNamespace

import httpx
import openai
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


def test_5xx_not_retried_charged_and_three_exhaust():
    c, fake, ledger = make(_raiser(lambda: HTTPErr(500)))
    for i in range(2):
        with pytest.raises(LLMUnavailable):
            run(c)
        assert fake.calls == i + 1 and ledger.spent == (i + 1) * TIMEOUT_CHARGE
    with pytest.raises(BudgetExhausted):
        run(c)
    assert fake.calls == 3 and ledger.spent == 3 * TIMEOUT_CHARGE and ledger.exhausted


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


def test_real_httpx_connect_error_retried(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    c, fake, ledger = make(_raiser(lambda: openai.APIConnectionError(
        request=_req())))
    # openai wraps httpx.ConnectError as the cause
    async def bad():
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError as ce:
            try:
                raise openai.APIConnectionError(request=_req()) from ce
            except openai.APIConnectionError as ae:
                raise _wrapped(ae)
    fake.behavior = bad
    with pytest.raises(LLMUnavailable):
        run(c)
    assert fake.calls == 6 and sleeps == [2, 4, 8, 16, 30] and ledger.spent == 0
    assert c._failures == 1


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


def test_success_resets_failure_counter():
    seq = iter([HTTPErr(500), HTTPErr(500), None, HTTPErr(500)])

    async def beh():
        e = next(seq)
        if e is not None:
            raise _wrapped(e)
        return SimpleNamespace(content="ok", usage_metadata=SimpleNamespace(input_tokens=1, output_tokens=1))
    c, fake, ledger = make(beh)
    for expect in (LLMUnavailable, LLMUnavailable):
        with pytest.raises(expect):
            run(c)
    assert run(c).text == "ok"
    with pytest.raises(LLMUnavailable):
        run(c)
    assert not ledger.exhausted


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
    assert ledger.spent == r.cost and c._failures == 0
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

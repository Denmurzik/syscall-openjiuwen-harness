import pytest

from harness import config
from harness.llm import LLMClient


def test_defaults(monkeypatch):
    for k in ("HARNESS_LLM_TIMEOUT", "HARNESS_RETRY_DELAYS", "HARNESS_DEADLINE_S",
              "HARNESS_STOP_NEW_BEFORE_DEADLINE_S", "HARNESS_PHASE1_TIMEOUT_CODING",
              "HARNESS_PHASE1_TIMEOUT_DEFAULT", "HARNESS_PROXY_DEAD_S"):
        monkeypatch.delenv(k, raising=False)
    assert config.llm_timeout() == config.LLM_TIMEOUT
    assert config.deadline_s() == config.DEADLINE_S
    assert config.stop_new_before_deadline_s() == config.STOP_NEW_BEFORE_DEADLINE_S
    assert config.phase1_timeout("coding") == 100 and config.phase1_timeout("math") == 15
    assert config.retry_delays() == config.RETRY_DELAYS
    assert config.proxy_dead_s() == config.PROXY_DEAD_S == 300


def test_overrides_read_at_call_time(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_TIMEOUT", "2.5")
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.1, 0.2")
    monkeypatch.setenv("HARNESS_DEADLINE_S", "45")
    monkeypatch.setenv("HARNESS_STOP_NEW_BEFORE_DEADLINE_S", "10")
    monkeypatch.setenv("HARNESS_PHASE1_TIMEOUT_CODING", "7")
    monkeypatch.setenv("HARNESS_PHASE1_TIMEOUT_DEFAULT", "3")
    assert config.llm_timeout() == 2.5
    assert config.retry_delays() == (0.1, 0.2)
    assert config.deadline_s() == 45 and config.stop_new_before_deadline_s() == 10
    assert config.phase1_timeout("coding") == 7 and config.phase1_timeout("math") == 3
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", "12")
    assert config.proxy_dead_s() == 12
    assert LLMClient("http://x", "k", None).timeout == 2.5
    assert LLMClient("http://x", "k", None, timeout=9).timeout == 9


@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf", "0"])
def test_invalid_falls_back_with_warning(monkeypatch, caplog, bad):
    config._warned.clear()
    monkeypatch.setenv("HARNESS_LLM_TIMEOUT", bad)
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", bad if bad != "0" else "1,x")
    with caplog.at_level("WARNING", logger="harness"):
        assert config.llm_timeout() == config.LLM_TIMEOUT
        assert config.llm_timeout() == config.LLM_TIMEOUT
        if bad != "0":
            assert config.retry_delays() == config.RETRY_DELAYS
    assert sum("HARNESS_LLM_TIMEOUT" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf"])
def test_invalid_proxy_dead_s_falls_back(monkeypatch, bad):
    monkeypatch.setenv("HARNESS_PROXY_DEAD_S", bad)
    assert config.proxy_dead_s() == config.PROXY_DEAD_S


def test_proxy_pause_grows_to_a_minute(monkeypatch):
    monkeypatch.delenv("HARNESS_RETRY_DELAYS", raising=False)
    assert [config.proxy_pause(i) for i in range(8)] == [2, 4, 8, 16, 30, 60, 60, 60]
    monkeypatch.setenv("HARNESS_RETRY_DELAYS", "0.05")
    assert [config.proxy_pause(i) for i in range(3)] == [0.05, 0.1, 0.2]
    assert config.proxy_pause(5000) == 60

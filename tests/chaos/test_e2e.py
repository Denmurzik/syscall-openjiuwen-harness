"""End-to-end runs of run_agent against a hostile fake proxy."""
import pytest

from tests.chaos.conftest import check_common, run_harness
from tests.chaos.fakeproxy import exhausted_after

pytestmark = pytest.mark.chaos

CYCLE = ["normal", "429", "500", "502", "reset", "empty", "malformed", "no_usage", "400_structured",
         "429_retry_after"]

SCENARIOS = {
    "normal": lambda n, b: "normal",
    "429_always": lambda n, b: "429",
    "429_retry_after": lambda n, b: "429_retry_after" if n % 2 else "normal",
    "500_always": lambda n, b: "500",
    "502_always": lambda n, b: "502",
    "400_structured": lambda n, b: "400_structured",
    "reset_always": lambda n, b: "reset",
    "empty_body": lambda n, b: "empty",
    "malformed_json": lambda n, b: "malformed",
    "no_usage": lambda n, b: "no_usage",
    "huge_usage": lambda n, b: "huge_usage",
    "timeout_always": lambda n, b: "timeout",
    "mixed_cycle": lambda n, b: CYCLE[n % len(CYCLE)],
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_hostile_proxy(name, tasks, tmp_path, make_proxy, baseline):
    proxy = make_proxy(SCENARIOS[name], sleep_s=20)
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 20000}, timeout=150)
    check_common(run, tasks, baseline, bound=120)
    assert proxy.requests > 0, "scenario never reached the proxy: phase 2 did not run"
    if name == "huge_usage":
        # once a response reports absurd usage the ledger is spent: only in-flight calls may follow
        assert proxy.requests <= 8, proxy.requests


@pytest.mark.parametrize("code", ["402", "403"])
def test_budget_exhausted_http(code, tasks, tmp_path, make_proxy, baseline):
    k = 3
    proxy = make_proxy(exhausted_after(k, code))
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 50000}, timeout=150)
    check_common(run, tasks, baseline, bound=100)
    # after the first refusal no new unit may start: only the (<= 3) in-flight calls can still be refused
    assert proxy.count(code) <= 5, proxy.actions
    assert proxy.requests <= k + 5


def test_budget_exhausted_hang(tasks, tmp_path, make_proxy, baseline):
    k = 3
    proxy = make_proxy(exhausted_after(k, "hang"), sleep_s=20)
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 50000}, timeout=150)
    check_common(run, tasks, baseline, bound=100)
    # timeouts for HARNESS_PROXY_DEAD_S (3 s here, 2 s per timeout) end the run: two rounds of 3 calls
    assert proxy.count("hang") <= 6, proxy.actions


def test_small_budget_is_respected(tasks, tmp_path, make_proxy, baseline):
    proxy = make_proxy()
    run = run_harness(tasks, tmp_path / "run", proxy, {"BUDGET": 1500}, timeout=150)
    check_common(run, tasks, baseline, bound=100)
    served_cost = proxy.count("normal") * (30 + 20)  # 300 prompt / 10 + 20 completion
    assert served_cost <= 1500 * 1.5 + 200, (served_cost, proxy.requests)


def test_no_proxy_at_all(tasks, tmp_path, baseline):
    """LLM_BASE_URL pointing at a closed port (connection refused for every call)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    run = run_harness(tasks, tmp_path / "run", None,
                      {"LLM_BASE_URL": f"http://127.0.0.1:{port}/v1", "LLM_API_KEY": "k", "BUDGET": 20000},
                      timeout=150)
    check_common(run, tasks, baseline, bound=100)


def test_deadline_ends_run_while_proxy_hangs(tasks, tmp_path, make_proxy, baseline):
    """Short run deadline + a proxy that hangs far longer than the client timeout."""
    proxy = make_proxy(lambda n, b: "hang", sleep_s=60)
    run = run_harness(tasks, tmp_path / "run", proxy,
                      {"BUDGET": 50000, "HARNESS_LLM_TIMEOUT": 30, "HARNESS_DEADLINE_S": 45,
                       "HARNESS_STOP_NEW_BEFORE_DEADLINE_S": 10, "HARNESS_PROXY_DEAD_S": 1000},
                      timeout=150)
    check_common(run, tasks, baseline, bound=90)

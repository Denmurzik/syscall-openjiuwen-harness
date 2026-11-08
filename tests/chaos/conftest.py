import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.chaos.fakeproxy import Proxy

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("_runner.py")
sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(items):
    for it in items:
        if "tests/chaos" in it.nodeid.replace("\\", "/"):
            it.add_marker(pytest.mark.chaos)


@dataclass
class Run:
    rc: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool
    out: Path


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=20)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def harness_env(tmp: Path, proxy: Proxy | None, env_extra: dict | None) -> dict:
    env = dict(os.environ)
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "TASKS", "OUT", "BUDGET"):
        env.pop(k, None)
    env.update({"CHAOS_ROOT": str(ROOT), "HARNESS_LOG_DIR": str(tmp / "hlogs"),
                "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "CHAOS_LLM_TIMEOUT": "2", "CHAOS_RETRY_DELAYS": "0.05,0.05,0.05,0.05,0.05"})
    if proxy is not None:
        env["LLM_BASE_URL"] = proxy.base_url
        env["LLM_API_KEY"] = "test-key"
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    return env


def start(tasks_dir: Path, tmp: Path, proxy: Proxy | None = None, env_extra: dict | None = None,
          out: Path | None = None):
    """Popen the harness (own process group); stdout/stderr go to files (orphans cannot block pipes)."""
    out = out or (tmp / "answers.json")
    so, se = open(tmp / "stdout.txt", "wb"), open(tmp / "stderr.txt", "wb")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, str(RUNNER), "--tasks-dir", str(tasks_dir), "--out", str(out)],
                            env=harness_env(tmp, proxy, env_extra), stdout=so, stderr=se,
                            stdin=subprocess.DEVNULL, cwd=str(tmp), **kw)
    so.close()
    se.close()
    return proc, out, time.monotonic()


def collect(proc, tmp: Path, out: Path, t0: float, timed_out: bool) -> Run:
    rd = lambda n: (tmp / n).read_text(encoding="utf-8", errors="replace")  # noqa: E731
    return Run(proc.returncode, rd("stdout.txt"), rd("stderr.txt"), time.monotonic() - t0, timed_out, out)


def run_harness(tasks_dir: Path, tmp: Path, proxy: Proxy | None = None, env_extra: dict | None = None,
                timeout: float = 120, out: Path | None = None) -> Run:
    tmp.mkdir(parents=True, exist_ok=True)
    proc, out, t0 = start(tasks_dir, tmp, proxy, env_extra, out)
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        try:
            proc.wait(10)
        except Exception:
            pass
    finally:
        if proc.poll() is None:
            kill_tree(proc)
    return collect(proc, tmp, out, t0, timed_out)


def answers_on_disk(tasks_dir: Path) -> dict:
    res = {}
    for d in sorted(Path(tasks_dir).iterdir()):
        a = d / "answer.txt"
        if a.is_file():
            res[d.name] = a.read_bytes()
    return res


@pytest.fixture(scope="session")
def gen_tasks_dir(tmp_path_factory) -> Path:
    from tools.gen_tasks import generate
    base = tmp_path_factory.mktemp("gen30")
    generate(base, 30, 7, 0.2, 0.5, "easy")
    return base / "tasks"


@pytest.fixture
def tasks(gen_tasks_dir, tmp_path) -> Path:
    d = tmp_path / "tasks"
    shutil.copytree(gen_tasks_dir, d)
    return d


@pytest.fixture(scope="session")
def baseline(gen_tasks_dir, tmp_path_factory) -> dict:
    """Phase-1-only run (no LLM): the answers every hostile run must still produce."""
    tmp = tmp_path_factory.mktemp("baseline")
    d = tmp / "tasks"
    shutil.copytree(gen_tasks_dir, d)
    r = run_harness(d, tmp / "run", None, timeout=240)
    assert r.rc == 0, r.stderr
    ans = answers_on_disk(d)
    assert len(ans) >= 5, "baseline solved too little to be a useful reference"
    return ans


@pytest.fixture
def make_proxy():
    made = []

    def make(behavior=None, sleep_s=30.0) -> Proxy:
        p = Proxy(behavior, sleep_s).start()
        made.append(p)
        return p

    yield make
    for p in made:
        p.stop()


def check_common(run: Run, tasks_dir: Path, baseline: dict, bound: float):
    assert not run.timed_out, f"run did not end within the bound; stderr tail: {run.stderr[-800:]}"
    assert run.rc == 0, run.stderr[-800:]
    assert "Traceback" not in run.stderr, run.stderr[-1500:]
    assert run.elapsed < bound, f"took {run.elapsed:.0f}s"
    got = answers_on_disk(tasks_dir)
    missing = sorted(set(baseline) - set(got))
    assert not missing, f"phase-1 answers missing: {missing}"
    for tid, raw in got.items():
        assert raw.endswith(b"\n") and raw.strip(), f"{tid}: empty/partial answer.txt {raw!r}"
    data = json.loads(run.out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert set(data) == set(got), f"sink and disk disagree: {sorted(set(data) ^ set(got))}"
    for tid, v in data.items():
        assert v == got[tid].decode("utf-8").strip()
    return got, data

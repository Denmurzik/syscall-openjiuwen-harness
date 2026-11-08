"""SIGKILL the whole run mid-flight: every answer on disk must be complete, the answers file valid JSON."""
import json
import random
import shutil
import time
from pathlib import Path

import pytest

from tests.chaos.conftest import answers_on_disk, kill_tree, start

pytestmark = pytest.mark.chaos


def _assert_consistent(tasks: Path, out: Path):
    got = answers_on_disk(tasks)
    for tid, raw in got.items():
        assert raw.endswith(b"\n") and raw.strip(), f"{tid}: partial answer.txt {raw!r}"
        raw.decode("utf-8")
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))  # must parse after an abrupt kill
        assert isinstance(data, dict)
        for tid, v in data.items():
            assert isinstance(v, str) and v.strip(), f"{tid}: empty value in answers file"
    return got


@pytest.mark.parametrize("after_n", [1, 4, 10])
def test_kill_during_phase1(after_n, tasks, tmp_path):
    tmp = tmp_path / "run"
    tmp.mkdir()
    proc, out, t0 = start(tasks, tmp)
    deadline = time.monotonic() + 100
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            if len(answers_on_disk(tasks)) >= after_n:
                break
            time.sleep(0.01)
        kill_tree(proc)
        proc.wait(15)
    finally:
        kill_tree(proc)
    time.sleep(0.5)  # orphans (if any) get a moment to misbehave
    got = _assert_consistent(tasks, out)
    assert len(got) >= after_n


def test_kill_during_phase2(tasks, tmp_path, make_proxy, baseline):
    """Kill while the proxy hangs (calls in flight): phase-1 answers survive intact."""
    proxy = make_proxy(lambda n, b: "hang", sleep_s=60)
    tmp = tmp_path / "run"
    tmp.mkdir()
    proc, out, t0 = start(tasks, tmp, proxy, {"BUDGET": 50000, "CHAOS_LLM_TIMEOUT": 60})
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline and proc.poll() is None and proxy.requests < 2:
            time.sleep(0.05)
        assert proxy.requests >= 2, "never reached phase 2"
        kill_tree(proc)
        proc.wait(15)
    finally:
        kill_tree(proc)
    got = _assert_consistent(tasks, out)
    assert set(baseline) <= set(got), sorted(set(baseline) - set(got))
    assert json.loads(out.read_text(encoding="utf-8")).keys() >= set(baseline)


def test_kill_loop_random_points(tasks, tmp_path):
    """Kill the run at 8 different instants and re-validate each time."""
    rng = random.Random(3)
    base = tmp_path / "pristine"
    shutil.copytree(tasks, base)
    for i in range(8):
        work = tmp_path / f"w{i}"
        shutil.copytree(base, work)
        tmp = tmp_path / f"run{i}"
        tmp.mkdir()
        proc, out, t0 = start(work, tmp)
        try:
            time.sleep(1.5 + rng.random() * 0.4 * (i + 1))
            kill_tree(proc)
            proc.wait(15)
        finally:
            kill_tree(proc)
        time.sleep(0.3)
        _assert_consistent(work, out)

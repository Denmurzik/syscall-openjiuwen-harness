import pytest
import os
from harness.proc import run_python


def test_timeout_kills_and_reports(tmp_path):
    r = run_python(["-c", "while True: pass"], cwd=tmp_path, timeout=1)
    assert r.timed_out and r.returncode != 0


def test_captures_stdout_stderr(tmp_path):
    r = run_python(["-c", "import sys; print(42); print('e', file=sys.stderr)"], cwd=tmp_path, timeout=10)
    assert (r.returncode, r.stdout.strip(), r.stderr.strip()) == (0, "42", "e")


def test_long_output_keeps_tail(tmp_path):
    r = run_python(["-c", "print('x' * 20000); print(42)"], cwd=tmp_path, timeout=10)
    assert r.stdout.rstrip().endswith("42") and len(r.stdout) <= 8 * 1024


@pytest.mark.skipif(os.name != "posix", reason="resource module is posix-only")
def test_limits_never_raise_with_low_hard_limit(monkeypatch):
    import resource
    from harness.proc import make_limits
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda r: (10, 100))

    def fake_set(r, v):
        calls.append(v)
        if v[0] > 100:
            raise ValueError("too high")
    monkeypatch.setattr(resource, "setrlimit", fake_set)
    make_limits(60)()
    assert calls and all(v[0] <= 100 for v in calls)
    monkeypatch.setattr(resource, "setrlimit", lambda r, v: (_ for _ in ()).throw(OSError("no")))
    make_limits(60)()  # failures are swallowed

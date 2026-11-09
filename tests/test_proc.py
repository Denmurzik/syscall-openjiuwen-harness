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


def test_child_env_has_no_proxy_key(tmp_path, monkeypatch):
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "BUDGET"):
        monkeypatch.setenv(k, "secret")
    code = "import os; print(sorted(k for k in ('LLM_API_KEY','LLM_BASE_URL','BUDGET') if k in os.environ))"
    r = run_python(["-c", code], cwd=tmp_path, timeout=10)
    assert r.stdout.strip() == "[]"
    r = run_python(["-c", code], cwd=tmp_path, timeout=10, env={"X": "1"})
    assert r.stdout.strip() == "[]"


def test_run_program_child_env_has_no_proxy_key(tmp_path, monkeypatch):
    from harness.sandbox import run_program
    monkeypatch.setenv("LLM_API_KEY", "secret")
    r = run_program("import os; print('LLM_API_KEY' in os.environ)", tmp_path)
    assert r.stdout.strip() == "False"


def test_blas_threads_pinned(tmp_path):
    r = run_python(["-c", "import os; print(os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS'))"],
                   cwd=tmp_path, timeout=10)
    assert r.stdout.split() == ["1", "1"]


@pytest.mark.skipif(os.name != "posix", reason="resource module is posix-only")
def test_fsize_limit_set():
    import resource
    from harness.proc import make_limits
    old = resource.getrlimit(resource.RLIMIT_FSIZE)
    try:
        make_limits(10)()
        assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 1024 ** 3
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, old)


def test_pandas_imports_under_run_python(tmp_path):
    pytest.importorskip("pandas")
    r = run_python(["-c", "import pandas; print(pandas.DataFrame({'a':[1]}).sum().a)"], cwd=tmp_path, timeout=60)
    assert r.stdout.strip() == "1", r.stderr

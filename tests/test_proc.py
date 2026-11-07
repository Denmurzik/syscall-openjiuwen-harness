from harness.proc import run_python


def test_timeout_kills_and_reports(tmp_path):
    r = run_python(["-c", "while True: pass"], cwd=tmp_path, timeout=1)
    assert r.timed_out and r.returncode != 0


def test_captures_stdout_stderr(tmp_path):
    r = run_python(["-c", "import sys; print(42); print('e', file=sys.stderr)"], cwd=tmp_path, timeout=10)
    assert (r.returncode, r.stdout.strip(), r.stderr.strip()) == (0, "42", "e")

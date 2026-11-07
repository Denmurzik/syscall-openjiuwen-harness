from pathlib import Path

from harness.sandbox import run_program


def test_run_program_without_workspace(tmp_path: Path):
    res = run_program("print('42')", tmp_path / "no_such_workspace")
    assert res.returncode == 0
    assert res.stdout.strip() == "42"

"""Expert 11, patch e (HARNESS_BIG_WS): a large workspace is linked into the program's directory, not copied."""
import os

import pytest

from harness import sandbox

PROG = "import os\nprint(open('sub/data.txt').read().strip(), os.path.islink('sub/data.txt'))\n"


def _ws(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "data.txt").write_text("42\n", encoding="utf-8")
    return ws


def _can_link(tmp_path) -> bool:
    try:
        os.symlink(tmp_path / "a", tmp_path / "b")
        return True
    except OSError:
        return False


def test_small_workspace_is_copied(tmp_path):
    assert sandbox.run_program(PROG, _ws(tmp_path)).stdout.split() == ["42", "False"]


def test_big_workspace_is_linked(tmp_path, monkeypatch):
    if not _can_link(tmp_path):
        pytest.skip("no symbolic links here")
    monkeypatch.setattr(sandbox, "BIG_WS_BYTES", 1)
    ws = _ws(tmp_path)
    assert sandbox.run_program(PROG, ws).stdout.split() == ["42", "True"]
    assert (ws / "sub" / "data.txt").read_text(encoding="utf-8") == "42\n"     # the run's directory is removed, not the file


def test_switch_off_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "BIG_WS_BYTES", 1)
    monkeypatch.setenv("HARNESS_BIG_WS", "0")
    assert sandbox.run_program(PROG, _ws(tmp_path)).stdout.split() == ["42", "False"]


def test_failed_copy_falls_back_to_links(tmp_path, monkeypatch):
    if not _can_link(tmp_path):
        pytest.skip("no symbolic links here")

    def full(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(sandbox.shutil, "copytree", full)
    assert sandbox.run_program(PROG, _ws(tmp_path)).stdout.split() == ["42", "True"]

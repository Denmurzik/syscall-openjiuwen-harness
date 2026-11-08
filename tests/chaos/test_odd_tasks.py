"""Odd task directories: the run must exit 0, never traceback, never hang."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tests.chaos.conftest import answers_on_disk, run_harness

pytestmark = pytest.mark.chaos


def _ok(run, bound=120, logged_tb_ok=False):
    """logged_tb_ok: the situation is unanswerable by construction, so a handled-and-logged
    traceback (log.exception) is acceptable noise; the run must still exit 0 and finish."""
    assert not run.timed_out, run.stderr[-800:]
    assert run.rc == 0, run.stderr[-800:]
    if not logged_tb_ok:
        assert "Traceback" not in run.stderr, run.stderr[-1500:]
    assert run.elapsed < bound, f"{run.elapsed:.0f}s"


def _subset(tasks: Path, dest: Path, names: list) -> Path:
    dest.mkdir(parents=True)
    for n in names:
        shutil.copytree(tasks / n, dest / n)
    return dest


def _first(tasks, prefix):
    return next(d.name for d in sorted(tasks.iterdir()) if d.name.startswith(prefix))


def _patch_task(tasks, tmp_path, prefix, edit):
    name = _first(tasks, prefix)
    sub = _subset(tasks, tmp_path / "t", [name])
    edit(sub / name / "task.json")
    return sub, name


def _edit_json(**changes):
    def edit(p):
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in changes.items():
            if v is KeyError:
                d.pop(k, None)
            else:
                d[k] = v
        p.write_text(json.dumps(d), encoding="utf-8")
    return edit


def test_zero_tasks(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    r = run_harness(d, tmp_path / "run", timeout=60)
    _ok(r, 30)


def test_tasks_dir_does_not_exist(tmp_path):
    r = run_harness(tmp_path / "nope", tmp_path / "run", timeout=60)
    _ok(r, 30, logged_tb_ok=True)


def test_symlink_loop(tasks, tmp_path):
    d = tasks / _first(tasks, "search") / "workspace"
    try:
        os.symlink(d, d / "loop", target_is_directory=True)
        os.symlink(tasks, tasks / "loop_dir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported here")
    r = run_harness(tasks, tmp_path / "run", timeout=150)
    _ok(r, 120)
    assert len(answers_on_disk(tasks)) >= 5


def test_huge_csv(tasks, tmp_path):
    name = _first(tasks, "data_analysis")
    sub = _subset(tasks, tmp_path / "t", [name])
    csv = next((sub / name / "workspace").glob("*.csv"))
    head = csv.read_text(encoding="utf-8").splitlines()[0]
    chunk = ("2024-01-15,Widget,North,3,9.99\n") * 100000
    with open(csv, "w", encoding="utf-8", newline="\n") as f:
        f.write(head + "\n")
        for _ in range(int(50e6 // len(chunk)) + 1):
            f.write(chunk)
    assert csv.stat().st_size >= 50e6
    r = run_harness(sub, tmp_path / "run", timeout=150)
    _ok(r, 100)


def test_binary_sales_csv(tasks, tmp_path):
    name = _first(tasks, "data_analysis")
    sub = _subset(tasks, tmp_path / "t", [name])
    csv = next((sub / name / "workspace").glob("*.csv"))
    csv.write_bytes(os.urandom(200_000) + b"\x00\xff\xfe")
    r = run_harness(sub, tmp_path / "run", timeout=100)
    _ok(r, 60)
    a = sub / name / "answer.txt"
    assert not a.exists() or a.read_text(encoding="utf-8").strip() != "", "empty answer for binary csv"


@pytest.mark.xfail(strict=True, reason="load_tasks reads task.json as plain utf-8: a BOM makes json.loads fail "
                   "and the whole task is skipped (fix: encoding='utf-8-sig')")
def test_task_json_bom(tasks, tmp_path):
    sub, name = _patch_task(tasks, tmp_path, "math",
                            lambda p: p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes()))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists(), "task.json with a UTF-8 BOM was skipped"


def test_task_json_extra_fields(tasks, tmp_path):
    sub, name = _patch_task(tasks, tmp_path, "math",
                            _edit_json(extra={"a": [1, 2]}, tags=None, weird="\u0000 "))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists()


@pytest.mark.xfail(strict=True, reason="load_tasks requires data['reward']: a missing key skips the task "
                   "although reward only orders phase 2 (fix: default 1.0)")
def test_task_json_missing_reward(tasks, tmp_path):
    sub, name = _patch_task(tasks, tmp_path, "math", _edit_json(reward=KeyError))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    # reward is only a phase-2 priority hint; a free deterministic answer should not be lost over it
    assert (sub / name / "answer.txt").exists(), "task with missing reward is skipped entirely"


def test_task_json_reward_string(tasks, tmp_path):
    sub, name = _patch_task(tasks, tmp_path, "math", _edit_json(reward="3"))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists()


@pytest.mark.xfail(strict=True, reason="float(data['reward']) raises on non-numeric text and the task is "
                   "skipped, losing a free phase-1 answer (fix: fall back to 1.0)")
def test_task_json_reward_garbage(tasks, tmp_path):
    sub, name = _patch_task(tasks, tmp_path, "math", _edit_json(reward="high"))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists(), "task with non-numeric reward is skipped entirely"


def test_task_json_garbage_files(tasks, tmp_path):
    name = _first(tasks, "math")
    sub = _subset(tasks, tmp_path / "t", [name])
    for i, raw in enumerate([b"", b"\xff\xfe\x00", b"[]", b"null", b"{", b"{" * 100000]):
        d = sub / f"broken_{i}"
        d.mkdir()
        (d / "task.json").write_bytes(raw)
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists(), "one good task lost among broken ones"


def test_non_utf8_kb_file(tasks, tmp_path):
    name = _first(tasks, "search")
    sub = _subset(tasks, tmp_path / "t", [name])
    ws = sub / name / "workspace"
    f = next(p for p in sorted(ws.rglob("*.md")))
    f.write_bytes(f.read_bytes() + bytes([10, 67, 97, 102, 233, 32, 198, 255, 254, 128]))
    (ws / "latin1.md").write_bytes(b"Name: Z" + bytes([252]) + b"rich" + bytes([10]) + b"Born: 1990" + bytes([10]))
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40)


@pytest.mark.skipif(os.name == "nt", reason="chmod read-only dirs are not enforced on Windows")
def test_read_only_workspace(tasks, tmp_path):
    for d in tasks.iterdir():
        if d.is_dir():
            os.chmod(d / "workspace", 0o555)
    try:
        r = run_harness(tasks, tmp_path / "run", timeout=150)
        _ok(r, 120)
    finally:
        for d in tasks.iterdir():
            if d.is_dir():
                os.chmod(d / "workspace", 0o755)


@pytest.mark.xfail(os.name == "nt", strict=True,
                   reason="Windows only: read-only attribute on a stale answer.txt makes the fallback open() "
                          "fail (os.replace over it is refused); POSIX replaces it fine")
def test_stale_read_only_answer_file(tasks, tmp_path):
    """answer.txt pre-exists and is read-only (left by an earlier attempt)."""
    name = _first(tasks, "math")
    sub = _subset(tasks, tmp_path / "t", [name])
    a = sub / name / "answer.txt"
    a.write_text("stale\n")
    os.chmod(a, 0o444)
    try:
        r = run_harness(sub, tmp_path / "run", timeout=60)
        _ok(r, 40)
        assert a.read_text().strip() != "stale", "stale read-only answer.txt was not replaced"
    finally:
        os.chmod(a, 0o666)


def test_answer_path_is_a_directory(tasks, tmp_path):
    name = _first(tasks, "math")
    sub = _subset(tasks, tmp_path / "t", [name])
    (sub / name / "answer.txt").mkdir()
    r = run_harness(sub, tmp_path / "run", timeout=60)
    _ok(r, 40, logged_tb_ok=True)


def test_out_path_unwritable(tasks, tmp_path):
    name = _first(tasks, "math")
    sub = _subset(tasks, tmp_path / "t", [name])
    bad = tmp_path / "afile"
    bad.write_text("x")
    r = run_harness(sub, tmp_path / "run", timeout=60, out=bad / "sub" / "answers.json")
    _ok(r, 40)
    assert (sub / name / "answer.txt").exists(), "answers file failure must not block answer.txt"


def test_thousand_tiny_tasks(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    for i in range(1000):
        t = d / f"math_{i:04d}"
        (t / "workspace").mkdir(parents=True)
        (t / "task.json").write_text(json.dumps({
            "task_id": t.name, "task_type": "math", "reward": 1, "check": "exact_int",
            "instruction": f"A warehouse starts with {100 + i} boxes. It ships 3 boxes per day for 2 days. "
                           "How many boxes does it have at the end?"}), encoding="utf-8")
    t0 = time.monotonic()
    r = run_harness(d, tmp_path / "run", timeout=420)
    el = time.monotonic() - t0
    _ok(r, 400)
    got = answers_on_disk(d)
    print(f"1000 tiny tasks: {el:.0f}s, {len(got)} answers")
    assert len(got) >= 900, f"only {len(got)} of 1000 answered"
    data = json.loads(r.out.read_text(encoding="utf-8"))
    assert len(data) == len(got)

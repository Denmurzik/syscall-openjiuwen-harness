import json
import logging
import os
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("harness")


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    check: str | None
    dir: Path
    workspace: Path


def _task_dirs(root: Path) -> list[Path]:
    def ok(d: Path) -> bool:
        return d.is_dir() and d.name != "__MACOSX" and not d.name.startswith(".")

    top = [d for d in sorted(root.iterdir()) if ok(d)]
    if any((d / "task.json").is_file() for d in top):
        return top
    nested = []  # archives may nest as task-dir/task-dir/<tasks>
    for d in top:
        try:
            nested += [c for c in sorted(d.iterdir()) if ok(c) and (c / "task.json").is_file()]
        except OSError:
            pass
    return nested or top


def _reward(data: dict, d: Path) -> float:
    try:
        v = float(data["reward"])
        if v == v and abs(v) != float("inf"):
            return v
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    log.warning("task dir %s: missing or non-numeric reward, using 1.0", d)
    return 1.0


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks = []
    for d in _task_dirs(Path(tasks_dir)):
        try:
            data = json.loads((d / "task.json").read_text(encoding="utf-8-sig"))
            tasks.append(Task(
                task_id=str(data["task_id"]),
                task_type=str(data["task_type"]),
                instruction=str(data["instruction"]),
                reward=_reward(data, d),
                check=data.get("check"),
                dir=d,
                workspace=d / "workspace",
            ))
        except Exception as e:
            log.warning("skipping task dir %s: %r", d, e)
    return sorted(tasks, key=lambda t: t.task_id)


def _fsync_path(path: Path, flags: int = os.O_RDONLY) -> None:
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_sink_path: Path | None = None
_sink_answers: dict[str, str] = {}
_sink_lock = threading.Lock()


def set_answer_sink(path: Path | None) -> None:
    """Also mirror every answer into one JSON file {task_id: answer} (None disables)."""
    global _sink_path
    with _sink_lock:
        _sink_answers.clear()
        if path is None:
            _sink_path = None
            return
        p = Path(path)
        if str(path).endswith(("/", "\\")) or p.is_dir():
            p = p / "answers.json"
        _sink_path = p


def _write_sink(task_id: str, answer: str) -> None:
    try:
        with _sink_lock:
            if _sink_path is None:
                return
            _sink_answers[task_id] = answer
            data = json.dumps(_sink_answers, ensure_ascii=False, indent=1)
            tmp = _sink_path.with_name(f".{_sink_path.name}.{os.getpid()}.tmp")
            try:
                _sink_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, _sink_path)
            except OSError:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                with open(_sink_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(data)
    except Exception as e:
        log.warning("answers file write failed: %r", e)


def write_answer(task: Task, answer: str) -> None:
    try:
        _write_answer(task, answer)
    finally:
        _write_sink(task.task_id, answer.strip())


def _write_answer(task: Task, answer: str) -> None:
    data = answer.strip() + "\n"
    target = task.dir / "answer.txt"
    tmp = task.dir / f".answer.txt.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, target)
        except PermissionError:
            os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)  # stale read-only answer.txt (Windows)
            os.replace(tmp, target)
        if os.name == "posix":
            _fsync_path(task.dir)
        return
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def copy_task(task: Task, dest_root: Path) -> Task:
    dest = Path(dest_root) / task.dir.name
    shutil.copytree(task.dir, dest)
    return Task(task.task_id, task.task_type, task.instruction, task.reward, task.check,
                dest, dest / "workspace")

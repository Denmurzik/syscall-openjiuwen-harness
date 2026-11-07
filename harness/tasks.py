import json
import logging
import os
import shutil
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


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks = []
    for d in sorted(Path(tasks_dir).iterdir()):
        if not d.is_dir():
            continue
        try:
            data = json.loads((d / "task.json").read_text(encoding="utf-8"))
            tasks.append(Task(
                task_id=str(data["task_id"]),
                task_type=str(data["task_type"]),
                instruction=str(data["instruction"]),
                reward=float(data["reward"]),
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


def write_answer(task: Task, answer: str) -> None:
    data = answer.strip() + "\n"
    target = task.dir / "answer.txt"
    tmp = task.dir / f".answer.txt.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
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

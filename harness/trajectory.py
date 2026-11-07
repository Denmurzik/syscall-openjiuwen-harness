import json
import threading
from pathlib import Path


class Trajectory:
    """Append-only JSONL event log; every record is flushed immediately."""

    def __init__(self, log_dir: Path):
        self.path = Path(log_dir) / "trajectory.jsonl"
        self._lock = threading.Lock()
        self._file = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        except OSError:
            self._file = None

    def record(self, event: dict) -> None:
        if self._file is None:
            return
        try:
            line = json.dumps(event, ensure_ascii=False, default=str)
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None

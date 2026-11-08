import json
import threading
from contextvars import ContextVar
from pathlib import Path

# model calls of the running phase-2 unit; the scheduler sets a fresh list per unit, the LLM client appends
CALLS: ContextVar[list | None] = ContextVar("harness_llm_calls", default=None)


def record_call(model: str, prompt_tokens: int, completion_tokens: int, cost: float,
                error: str | None = None) -> None:
    calls = CALLS.get()
    if calls is None:
        return
    entry = {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "cost": cost}
    if error:
        entry["error"] = error
    calls.append(entry)


class Trajectory:
    """Append-only JSONL event log; every record is flushed immediately."""

    def __init__(self, log_dir: Path):
        self.path = Path(log_dir) / "trajectory.jsonl"
        self._lock = threading.Lock()
        self._file = None
        self.cost_total = 0.0  # sum of the `cost` fields written so far
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
                self.cost_total += float(event.get("cost") or 0)
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

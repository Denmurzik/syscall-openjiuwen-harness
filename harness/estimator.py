"""Rough phase-2 token estimates (tuned in task 12)."""
from harness.tasks import Task

CHARS_PER_TOKEN = 3
PROMPT_WEIGHT = 10  # prompt tokens count 1/10

JSON_BASE = 80
PROGRAM_BASE = 120
CODING_BASE = 200
CODING_EXTRA_CHARS = 2000


def _read(path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def estimate_json_tokens(template: str) -> float:
    return len(template) / CHARS_PER_TOKEN / PROMPT_WEIGHT + JSON_BASE


def estimate_tokens(task: Task) -> float:
    try:
        if task.task_type == "coding":
            ws = task.workspace
            chars = len(_read(ws / "mylib.py"))
            try:
                tests = sorted(ws.glob("test_*.py"))
            except OSError:
                tests = []
            chars += sum(len(_read(p)) for p in tests)
            return (chars + CODING_EXTRA_CHARS) / CHARS_PER_TOKEN / PROMPT_WEIGHT + CODING_BASE
        from harness.llm_routes import describe_workspace
        ctx = len(task.instruction) + len(describe_workspace(task.workspace))
        return ctx / CHARS_PER_TOKEN / PROMPT_WEIGHT + PROGRAM_BASE
    except Exception:
        return float(PROGRAM_BASE)

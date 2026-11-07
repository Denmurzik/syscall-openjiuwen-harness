"""Model arbitration among behaviourally different green fixes of a coding task.

One call per task, all ambiguous functions in one prompt, the answer is option numbers.
The deterministic choice is already on disk; it is replaced only after the chosen variant
passes the full visible test run (tests are never touched).
"""
import asyncio
import re
import shutil
from pathlib import Path

from harness.llm import LLMUnavailable  # BudgetExhausted is deliberately not caught here
from harness.llm_routes import MODEL_FAST, _run_candidate, write_mylib
from harness.result import SolveResult
from harness.solvers.coding_solver import _segment, _splice, _units
from harness.tasks import Task

MAX_FUNCS = 4          # numbers + commas must fit in max_tokens
MAX_TOKENS = 8
SYSTEM = ("Pick, for each function, the option that implements what the function name means. "
          "Reply with the option numbers only, comma-separated, in the order of the functions.")


def _prompt(items: list[dict], asserts: list[str]) -> str:
    parts = []
    for it in items:
        opts = "\n".join(f"Option {i + 1}:\n{c.strip()}" for i, c in enumerate(it["candidates"]))
        parts.append(f"Function `{it['func']}`, buggy version:\n{it.get('buggy', '').strip()}\n\n"
                     f"Candidate fixes (all pass the visible tests):\n{opts}")
    tests = "\n".join(a for a in asserts if a)
    if tests:
        parts.append(f"Visible tests:\n{tests}")
    return "\n\n".join(parts)


def _regex(items: list[dict]) -> str:
    return ",".join("[1-%d]" % len(it["candidates"]) for it in items)


def parse_choice(text: str, items: list[dict]) -> list[int] | None:
    """Zero-based option per function, from the first digits in the reply; None if unusable."""
    digits = re.findall(r"\d", text or "")
    if len(digits) < len(items):
        return None
    picks = [int(d) - 1 for d in digits[:len(items)]]
    if any(not 0 <= p < len(it["candidates"]) for p, it in zip(picks, items)):
        return None
    return picks


async def _ask(client, items: list[dict], asserts: list[str]) -> list[int] | None:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _prompt(items, asserts)}]
    if not getattr(client, "structured_output_refused", False):
        try:
            r = await client.chat(MODEL_FAST, messages, MAX_TOKENS,
                                  extra_body={"guided_regex": _regex(items)})
            picks = parse_choice(r.text, items)
            if picks is not None:
                return picks   # garbage under guided_regex: ask once more without it
        except LLMUnavailable as e:
            if "400" not in str(e):
                raise
            try:
                client.structured_output_refused = True
            except Exception:
                pass
    r = await client.chat(MODEL_FAST, messages, MAX_TOKENS)
    return parse_choice(r.text, items)


def _apply(src: str, items: list[dict], picks: list[int]) -> str | None:
    """`src` with each function whose pick differs from the current choice replaced."""
    for it, p in zip(items, picks):
        if p == it["chosen"]:
            continue
        units = {u.qualname: u for u in _units(src)}
        u = units.get(it["func"])
        if u is None:
            return None
        src = _splice(src, u, it["candidates"][p])
    return src


async def arbitrate(task: Task, extra: dict, client) -> SolveResult:
    items = list((extra or {}).get("ambiguous") or [])[:MAX_FUNCS]
    items = [it for it in items if len(it.get("candidates", ())) >= 2]
    if not items:
        return SolveResult("solved", answer="DONE", note="nothing to arbitrate")
    try:
        picks = await _ask(client, items, list(extra.get("visible_asserts") or []))
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
    if picks is None:
        return SolveResult("solved", answer="DONE", note="arbitration reply unusable, kept")
    if all(p == it["chosen"] for p, it in zip(picks, items)):
        return SolveResult("solved", answer="DONE", note="arbitration kept the choice")
    ws = Path(task.workspace)
    mylib = ws / "mylib.py"
    try:
        current = mylib.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return SolveResult("solved", answer="DONE", note=f"arbitration skipped: {e}")
    # try all changes, then each one alone: a combination may break what each fixes alone
    tries = [list(picks)]
    changed = [i for i, (p, it) in enumerate(zip(picks, items)) if p != it["chosen"]]
    if len(changed) > 1:
        tries += [[p if j == i else it["chosen"] for j, (p, it) in enumerate(zip(picks, items))]
                  for i in changed]
    for pk in tries:
        try:
            new = _apply(current, items, pk)
            if new is None or new == current:
                continue
            report = await asyncio.to_thread(_run_candidate, ws, new)
        except Exception:
            continue
        if report.ok:
            write_mylib(mylib, new)
            shutil.rmtree(ws / "__pycache__", ignore_errors=True)
            return SolveResult("solved", answer="DONE", note=f"arbitration chose options {[p + 1 for p in pk]}")
    return SolveResult("solved", answer="DONE", note="arbitrated choice failed the tests, kept")

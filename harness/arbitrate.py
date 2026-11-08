"""Model arbitration among behaviourally different green fixes of a coding task.

One call per model per task (MODEL_FAST and MODEL_ALT, concurrently, independently), all
ambiguous functions in one prompt, the answer is option numbers. A function's phase-1 choice
changes only when both models name the same other option; any disagreement, unusable reply or
unavailable model keeps it. The deterministic choice is already on disk; it is replaced only
after the agreed variant passes the full visible test run (tests are never touched).
"""
import ast
import asyncio
import difflib
import re
import shutil
import textwrap
from pathlib import Path

from harness.llm import LLMUnavailable  # BudgetExhausted is deliberately not caught here
from harness.config import ARBITRATE_MODELS
from harness.llm_routes import _run_candidate, write_mylib
from harness.result import SolveResult
from harness.solvers.coding_solver import _segment, _splice, _units
from harness.tasks import Task

MAX_FUNCS = 4          # numbers + commas must fit in max_tokens
MAX_TOKENS = 8
SYSTEM = ("Pick, for each function, the option that implements what the function name means. "
          "Reply with the option numbers only, comma-separated, in the order of the functions.")


def _unparse(src: str) -> str:
    """Same formatting on both sides, so only real edits show up in the diff."""
    try:
        return ast.unparse(ast.parse(textwrap.dedent(src)))
    except SyntaxError:
        return src


def _changes(buggy: str, cand: str) -> str:
    """Changed lines of a candidate against the buggy function, one short line per edit."""
    old, new = _unparse(buggy).splitlines(), _unparse(cand).splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        was = " / ".join(x.strip() for x in old[i1:i2]) or "(nothing)"
        now = " / ".join(x.strip() for x in new[j1:j2]) or "(removed)"
        out.append(f"line {i1 + 1}: `{was}` -> `{now}`")
    return "; ".join(out) or "(same as the buggy version)"


def _short(v: str, n: int = 24) -> str:
    return v if len(v) <= n else v[:n - 1] + "…"


def _prompt(items: list[dict], asserts: list[str]) -> str:
    parts = []
    for it in items:
        opts = "\n".join(f"Option {i + 1}: {_changes(it.get('buggy', ''), c)}"
                         for i, c in enumerate(it["candidates"]))
        block = (f"Function `{it['func']}`, buggy version:\n{it.get('buggy', '').strip()}\n\n"
                 f"Candidate fixes (all pass the visible tests), as edits of the buggy version:\n{opts}")
        calls, outs = it.get("calls") or [], it.get("outputs") or []
        if calls and outs:
            rows = [f"{c} -> " + " | ".join(_short(o[k]) if k < len(o) else "?" for o in outs)
                    for k, c in enumerate(calls[:5])]
            block += f"\n\nResults of options 1..{len(outs)} on sample calls:\n" + "\n".join(rows)
        parts.append(block)
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


async def _ask(client, model: str, items: list[dict], asserts: list[str]) -> list[int] | None:
    """Structured output and its 400 fallback live in LLMClient; here only a garbage reply
    under the regex gets one more try without it."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _prompt(items, asserts)}]
    structured = getattr(client, "structured_ok", True)
    if structured:
        r = await client.chat(model, messages, MAX_TOKENS,
                              extra_body={"guided_regex": _regex(items)})
        picks = parse_choice(r.text, items)
        if picks is not None or not getattr(client, "structured_ok", True):
            return picks
    r = await client.chat(model, messages, MAX_TOKENS)
    return parse_choice(r.text, items)


async def _agreed(client, items: list[dict], asserts: list[str]) -> list[int]:
    """Per function: the option every model named, else the phase-1 choice. LLMUnavailable only
    when no model was reachable; BudgetExhausted (and anything unexpected) propagates."""
    replies = await asyncio.gather(*(_ask(client, m, items, asserts) for m in ARBITRATE_MODELS),
                                   return_exceptions=True)
    for r in replies:
        if isinstance(r, BaseException) and not isinstance(r, LLMUnavailable):
            raise r
    if all(isinstance(r, LLMUnavailable) for r in replies):
        raise replies[0]
    picks = [it["chosen"] for it in items]
    if any(r is None or isinstance(r, BaseException) for r in replies):
        return picks
    for i, it in enumerate(items):
        named = {r[i] for r in replies}
        if len(named) == 1:
            picks[i] = named.pop()
    return picks


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
        picks = await _agreed(client, items, list(extra.get("visible_asserts") or []))
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
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

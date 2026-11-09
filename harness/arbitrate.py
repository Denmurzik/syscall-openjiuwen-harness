"""Model arbitration among behaviourally different green fixes of a coding task, and the independent
check of a green single fix that feeds it (`verify_fix`).

One call per model per task (MODEL_FAST and MODEL_ALT, concurrently, independently), all
ambiguous functions in one prompt, the answer is option numbers. A function's phase-1 choice
changes only when both models name the same other option; any disagreement, unusable reply or
unavailable model keeps it. The deterministic choice is already on disk; it is replaced only
after the agreed variant passes the full visible test run (tests are never touched).
"""
import ast
import asyncio
import dataclasses
import difflib
import re
import shutil
import textwrap
from pathlib import Path

from harness.llm import LLMUnavailable  # BudgetExhausted is deliberately not caught here
from harness.config import ARBITRATE_MODELS, MODEL_FAST, VERIFY_MAX_TOKENS
from harness.llm_routes import _cap, _run_candidate, apply_patch, write_mylib
from harness.result import SolveResult
from harness.solvers.coding_solver import _norm, _segment, _splice, _units
from harness.solvers.probe import behaviour
from harness.tasks import Task

MAX_FUNCS = 4          # numbers + commas must fit in max_tokens
MAX_TOKENS = 8
MAX_ROWS = 5           # probe rows shown per function, after every row where the options differ
VERIFY_TESTS_CAP = 2000  # chars of visible tests per function shown to the model writing its own fix
VERIFY_SYSTEM = ("Fix the bugs in these Python functions. The visible tests are only examples, hidden tests "
                 "check other inputs: write what the function name means, not what merely fits the examples. "
                 "Return only the corrected functions, no explanations, in this format, one block per function:\n"
                 "FUNC <name>\n<complete new text of the function>")
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
            cells = [[_short(o[k]) if k < len(o) else "?" for o in outs] for k in range(len(calls))]
            differ = [k for k, row in enumerate(cells) if len(set(row)) > 1]
            same = [k for k in range(len(calls)) if k not in differ]
            # every row that tells the options apart, then the others up to the limit
            order = differ + same[:max(0, MAX_ROWS - len(differ))]
            rows = [f"{calls[k]} -> " + " | ".join(cells[k]) for k in order]
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


def _votes(replies: list) -> str:
    """Each model's reply as 1-based option numbers, for the note."""
    def one(r):
        if isinstance(r, BaseException):
            return "unavailable"
        return "unusable" if r is None else ",".join(str(p + 1) for p in r)
    return ", ".join(f"{m} {one(r)}" for m, r in zip(ARBITRATE_MODELS, replies))


async def _agreed(client, items: list[dict], asserts: list[str]) -> tuple[list[int], str, str]:
    """Per function: the option every model named, else the phase-1 choice; plus why the
    choice stays where it is ("one model unavailable", "unusable reply", "models disagree",
    "models confirmed") and each model's vote. LLMUnavailable only when no model was reachable;
    BudgetExhausted (and anything unexpected) propagates."""
    replies = await asyncio.gather(*(_ask(client, m, items, asserts) for m in ARBITRATE_MODELS),
                                   return_exceptions=True)
    for r in replies:
        if isinstance(r, BaseException) and not isinstance(r, LLMUnavailable):
            raise r
    if all(isinstance(r, LLMUnavailable) for r in replies):
        raise replies[0]
    picks = [it["chosen"] for it in items]
    votes = _votes(replies)
    if any(isinstance(r, BaseException) for r in replies):
        return picks, "one model unavailable", votes
    if any(r is None for r in replies):
        return picks, "unusable reply", votes
    agreed = True
    for i, it in enumerate(items):
        named = {r[i] for r in replies}
        if len(named) == 1:
            picks[i] = named.pop()
        else:
            agreed = False
    return picks, "models confirmed" if agreed else "models disagree", votes


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
        picks, why, votes = await _agreed(client, items, list(extra.get("visible_asserts") or []))
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
    res = await _decide(task, items, picks, why)
    return dataclasses.replace(res, extra={"votes": votes})


async def _decide(task: Task, items: list[dict], picks: list[int], why: str) -> SolveResult:
    if all(p == it["chosen"] for p, it in zip(picks, items)):
        return SolveResult("solved", answer="DONE", note=f"arbitration kept the choice ({why})")
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


def _verify_prompt(items: list[dict]) -> str:
    return "\n\n".join(f"Function `{it['func']}`, buggy version:\n{it['buggy']}\n\n"
                        f"Visible tests:\n{_cap(it.get('tests', ''), VERIFY_TESTS_CAP)}" for it in items)


def _differing(ws: Path, current: str, reply: str, items: list[dict]) -> list[dict]:
    """Arbitration items for the functions whose fix in `reply` passes the visible tests and behaves differently
    from ours on the probe inputs. Our fix is the phase-1 choice (option 1)."""
    theirs = apply_patch(current, reply)
    if theirs is None:
        return []
    ours_units, their_units = ({u.qualname: u for u in _units(s)} for s in (current, theirs))
    out = []
    for it in items:
        func, inputs = it["func"], it["inputs"]
        u, v = ours_units.get(func), their_units.get(func)
        if u is None or v is None:
            continue
        ours = textwrap.dedent(_segment(current, u)).strip("\n")
        cand = textwrap.dedent(_segment(theirs, v)).strip("\n")
        if _norm(ours) == _norm(cand):
            continue
        new = _splice(current, u, cand)   # only this function of the reply, in the file as it is now
        behaves = [behaviour(current, func, inputs), behaviour(new, func, inputs)]
        if behaves[0] == behaves[1] or not _run_candidate(ws, new).ok:
            continue
        out.append({"func": func, "buggy": it["buggy"], "candidates": [ours, cand], "chosen": 0,
                    "calls": [f"{func}({', '.join(map(repr, a))})" for a in inputs], "outputs": behaves})
    return out


async def verify_fix(task: Task, extra: dict, client) -> SolveResult:
    """MODEL_FAST writes its own fix from the buggy function and the visible tests, without seeing ours. Same
    behaviour on the probe inputs: ours is confirmed, no more calls. Different and green: 2c arbitration decides,
    so ours is replaced only when both models agree on the other. Any doubt keeps ours. `unparsed` only when a
    model was unavailable, with what a later try can resume from."""
    if extra.get("ambiguous"):   # the candidate is paid for already, only the arbitration is left
        return await _judge(task, extra, client)
    items = list(extra.get("verify") or [])[:MAX_FUNCS]
    ws = Path(task.workspace)
    try:
        current = (ws / "mylib.py").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return SolveResult("solved", answer="DONE", note=f"check skipped: {e}")
    try:
        r = await client.chat(MODEL_FAST, [{"role": "system", "content": VERIFY_SYSTEM},
                                           {"role": "user", "content": _verify_prompt(items)}], VERIFY_MAX_TOKENS)
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
    if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= VERIFY_MAX_TOKENS:
        # at the limit the reply may be cut, and a cut one can still parse with branches lost
        return SolveResult("solved", answer="DONE", note="check: reply hit the output limit, kept")
    try:
        differ = await asyncio.to_thread(_differing, ws, current, r.text, items)
    except Exception as e:
        return SolveResult("solved", answer="DONE", note=f"check failed, kept: {e!r}"[:200])
    if not differ:
        return SolveResult("solved", answer="DONE", note="check: no other green behaviour, kept")
    return await _judge(task, {"ambiguous": differ, "visible_asserts": list(extra.get("visible_asserts") or [])},
                        client)


async def _judge(task: Task, extra: dict, client) -> SolveResult:
    """2c arbitration of ours against the model's candidate; the note carries each model's vote."""
    res = await arbitrate(task, extra, client)
    if res.status != "solved":
        return SolveResult("unparsed", note=res.note, extra=extra)
    votes = (res.extra or {}).get("votes")
    return SolveResult("solved", answer=res.answer, note=f"{res.note}; votes: {votes}"[:200] if votes else res.note)

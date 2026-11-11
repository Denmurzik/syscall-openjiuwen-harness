"""End-of-run audit of green coding functions across the task set (HARNESS_AUDIT_FORMS).

The jury's coding tasks are cut from one pool of functions, and hidden tests catch bugs that the one visible assert
does not. For the six functions of the organizers' examples the known text settles it (`_restore_known`). For any
other function the set itself is the evidence: when the same function (same name, same parameters) stands in two or
more different forms in files whose visible tests are all green, and the forms behave differently on sample calls,
at most one of them is right. Forms that give the same results on every sample call are one option. MODEL_FAST and
MODEL_ALT are shown the options and the differing calls and name one each (8 tokens, as 2c); only when both name the
same option is its text put into the files that hold another one, and only where the visible tests stay green with
it. An option held green by fewer files than another (the set votes against it) wins only when MODEL_REASON names it
too. No rival forms in the set: no call, nothing changes. Any doubt keeps every file as it is.
"""
import ast
import asyncio
import logging
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import ARBITRATE_MODELS, MODEL_REASON
from harness.llm import LLMUnavailable
from harness.solvers import coding_solver as cs
from harness.solvers.probe import behaviour, inputs_from_call, visible_call
from harness.solvers.testrun import run_tests

log = logging.getLogger("harness")

MAX_FORMS = 3            # options of one function shown to the models, by votes
MAX_ROWS = 6             # differing sample calls shown
MAX_TOKENS = 8
RUN_TIMEOUT = 10.0       # one run of a task's visible tests
CENSUS_BUDGET_S = 60.0
MIN_HOLDERS = 2          # files that must hold the function green under a visible test, all forms together
REASON_TOKENS = 1500     # output limit of the third opinion on a minority pick
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
SYSTEM = ("Several versions of one Python function were found in different files; hidden tests check other inputs "
          "than the examples, edge cases included. Pick the version that implements what the function name means "
          "for every input. Reply with the option number only.")

# edge inputs by the type of a visible argument, tried besides the variants of the visible call
_EDGE = {
    str: ["", "a", "ab", "abca", "level", "queue", "Hello World", "AEIOU xyz", "one  two", " x y "],
    int: [0, 1, 2, 3, 4, 6, 7, 10, -3],
    list: [[], [5], [3, 5], [-3, -8], [1, 3, 2], [2, 1], [4, 1, 3, 2], [0, 0, 5], [-2, 3], [1, 2, 3, 4], [2, 2, 1]],
}


@dataclass
class Holder:
    task: object
    ws: Path
    test_file: str
    test_src: str
    forms: dict = field(default_factory=dict)   # name -> (AST dump, text, tested)


def _holder(task, timeout: float = RUN_TIMEOUT) -> Holder | None:
    """One coding workspace whose visible tests are all green: its top-level self-contained functions defined once,
    undecorated, names the examples do not show."""
    ws = Path(task.workspace)
    if not (ws / "mylib.py").is_file():
        return None
    tests = sorted(p.name for p in ws.glob("test_*.py"))
    test_file = "test_mylib.py" if (ws / "test_mylib.py").is_file() else (tests[0] if tests else None)
    if test_file is None:
        return None
    src = (ws / "mylib.py").read_text(encoding="utf-8")
    test_src = (ws / test_file).read_text(encoding="utf-8")
    tree = ast.parse(src)
    refs = cs._test_refs(test_src)
    if any(not r for r in refs.values()):
        return None   # a test that cannot be laid on functions: nothing is known about this file
    skipped = cs._skipped_tests(ast.parse(test_src))
    if not run_tests(ws, test_file, timeout=timeout).ok:
        return None
    tested = set().union(set(), *(r for f, r in refs.items() if f not in skipped))
    names = cs._module_names(tree)
    defined: dict = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[n.name] = defined.get(n.name, 0) + 1
    h = Holder(task, ws, test_file, test_src)
    for node in tree.body:
        if type(node) is not ast.FunctionDef or node.decorator_list or defined[node.name] != 1 \
                or node.name in cs._EXAMPLE_NAMES or not cs._self_contained(node, names):
            continue
        text = ast.get_source_segment(src, node)
        if text:
            h.forms[node.name] = (ast.dump(node), text, node.name in tested)
    return h


def census(tasks: list, budget_s: float = CENSUS_BUDGET_S) -> tuple[list, dict]:
    """(holders, rivals). rivals: name -> the forms to judge [{dump, text, votes, untested}], tested forms first by
    votes, for functions that stand in more than one form with the same parameters, green under a visible test in at
    least MIN_HOLDERS files in all. An unfinished census gives nothing (the files left out could hold the votes
    against)."""
    stop = time.monotonic() + budget_s
    holders = []
    for t in tasks:
        left = stop - time.monotonic()
        if left <= 0:
            return [], {}
        try:
            h = _holder(t, min(RUN_TIMEOUT, left))
        except Exception:
            h = None
        if h is not None:
            holders.append(h)
    if time.monotonic() > stop:
        return [], {}   # the last run may have been cut by the time left: its file could hold a vote
    forms: dict = {}
    for h in holders:
        for name, (dump, text, tested) in h.forms.items():
            e = forms.setdefault(name, {}).setdefault(dump, {"dump": dump, "text": text, "votes": 0, "untested": 0})
            e["votes" if tested else "untested"] += 1
    rivals = {}
    for name, by_dump in forms.items():
        fs = sorted(by_dump.values(), key=lambda e: (-e["votes"], -e["untested"], e["text"]))
        if len(fs) < 2 or sum(e["votes"] for e in fs) < MIN_HOLDERS:
            continue
        try:
            params = {ast.dump(ast.parse(e["text"]).body[0].args) for e in fs}
        except Exception:
            continue
        if len(params) != 1:
            continue   # one name, different signatures: not the same function
        rivals[name] = fs
    return holders, rivals


def _inputs(name: str, holders: list) -> list[tuple]:
    """Sample calls: the visible calls of the function in the set, their variants, and edge values of the same
    types."""
    seen, out = set(), []

    def add(args):
        k = repr(args)
        if k not in seen and len(out) < 60:
            seen.add(k)
            out.append(args)

    calls = []
    for h in holders:
        if name in h.forms:
            vc = visible_call(h.test_src, name)
            if vc and repr(vc[0]) not in {repr(c) for c in calls}:
                calls.append(vc[0])
    for args in calls[:3]:
        for seed in (0, 1):
            for a in inputs_from_call(args, n=8, seed=seed):
                add(a)
        for i, v in enumerate(args):
            kind = list if isinstance(v, list) else (int if isinstance(v, int) and not isinstance(v, bool) else type(v))
            for e in _EDGE.get(kind, ()):
                add(tuple(args[:i]) + (e,) + tuple(args[i + 1:]))
    return out


def options(name: str, forms: list, holders: list) -> tuple[list, list, list[str]]:
    """(shown, rest, rows). The forms are run on the sample calls and those that give the same results on all of
    them make one option: its text is the most-held one, its votes those of all its forms together. `shown`: the
    options asked about, by votes. With two or more options tested somewhere, the MAX_FORMS best of those (an option
    no visible test of the set ever ran is not shown: it is in `rest` and takes the winner); with one, it and up to
    MAX_FORMS - 1 untested options. Tested options beyond MAX_FORMS are in neither and stay as they are. `rows`: the
    sample calls on which the shown options differ (calls where one raises go last). Fewer than two shown options:
    nothing to ask."""
    inputs = _inputs(name, holders)
    if not inputs:
        return [], [], []
    by_outs: dict = {}
    for f in forms:   # `forms` come most-held first: an option's first form is its text
        outs = tuple(behaviour(f["text"] + "\n", name, inputs))
        o = by_outs.setdefault(outs, {"text": f["text"], "dumps": set(), "votes": 0, "untested": 0, "outs": outs})
        o["dumps"].add(f["dump"])
        o["votes"] += f["votes"]
        o["untested"] += f["untested"]
    opts = sorted(by_outs.values(), key=lambda o: (-o["votes"], -o["untested"], o["text"]))
    tested = [o for o in opts if o["votes"]]
    untested = [o for o in opts if not o["votes"]]
    if len(tested) >= 2:
        shown, rest = tested[:MAX_FORMS], untested
    else:
        shown, rest = tested + untested[:MAX_FORMS - 1], []
    if len(shown) < 2:
        return shown, rest, []
    outs = [o["outs"] for o in shown]
    differ = [k for k in range(len(inputs)) if len({o[k] for o in outs}) > 1]
    rnd = random.Random(0)
    clean = [k for k in differ if not any(o[k].startswith("<exc") for o in outs)]
    if len(clean) > MAX_ROWS:
        clean = sorted(rnd.sample(clean, MAX_ROWS))
    picked = (clean + [k for k in differ if k not in clean])[:MAX_ROWS]
    rows = [f"{name}({', '.join(map(repr, inputs[k]))}) -> " + " | ".join(o[k][:40] for o in outs) for k in picked]
    return shown, rest, rows


def _prompt(name: str, forms: list, rows: list[str]) -> str:
    opts = "\n\n".join(f"Option {i + 1}:\n{f['text'].strip()}" for i, f in enumerate(forms))
    return (f"Function `{name}`.\n\n{opts}\n\nResults of options 1..{len(forms)} on sample calls:\n"
            + "\n".join(rows))


def reason_estimate(prompt: str) -> int:
    """What the ledger must hold for the third opinion: its prompt and its whole output limit."""
    return (len(SYSTEM) + len(prompt)) // 3 + REASON_TOKENS


async def _ask(client, model: str, prompt: str, n: int, reason: bool = False) -> int | None:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if reason:
        r = await client.chat(model, messages, REASON_TOKENS, reasoning_effort="low")
        if getattr(r, "truncated", False):
            return None
        text = _THINK.sub("", r.text or "").strip()
        digits = [c for c in text[-12:] if c.isdigit()]
        return int(digits[-1]) - 1 if digits and 1 <= int(digits[-1]) <= n else None
    kw = {"extra_body": {"guided_regex": "[1-%d]" % n}} if getattr(client, "structured_ok", True) else {}
    r = await client.chat(model, messages, MAX_TOKENS, **kw)
    digits = [c for c in (r.text or "") if c.isdigit()]
    if not digits or not 1 <= int(digits[0]) <= n:
        return None
    return int(digits[0]) - 1


async def decide(client, name: str, forms: list, holders: list, may_reason=None) -> tuple[dict | None, str]:
    """(winner, note). `winner`: {"text", "votes", "losers"}, the option (`options`) both arbitration models named
    and the AST dumps of the forms it replaces (the other shown options and `rest`). An option with fewer votes than
    the top one (the set holds another behaviour green more often) needs MODEL_REASON's word too, and
    `may_reason(estimate)` must allow that call (spend gate, ledger). None: forms that do not differ, a model
    unavailable, an unusable reply, a disagreement, or a minority pick not confirmed. BudgetExhausted propagates."""
    shown, rest, rows = await asyncio.to_thread(options, name, forms, holders)
    if not rows:
        return None, "forms do not differ on the sample calls"
    prompt = _prompt(name, shown, rows)
    replies = await asyncio.gather(*(_ask(client, m, prompt, len(shown)) for m in ARBITRATE_MODELS),
                                   return_exceptions=True)
    for r in replies:
        if isinstance(r, BaseException) and not isinstance(r, LLMUnavailable):
            raise r
    votes = ", ".join("unavailable" if isinstance(r, BaseException) else ("unusable" if r is None else str(r + 1))
                      for r in replies)
    if any(isinstance(r, BaseException) or r is None for r in replies) or len(set(replies)) != 1:
        return None, f"no agreement ({votes}) on {[o['votes'] for o in shown]}"
    pick = replies[0]
    win = shown[pick]
    tally = f"option {pick + 1} of votes {[o['votes'] for o in shown]}"
    winner = {"text": win["text"], "votes": win["votes"],
              "losers": set().union(*(o["dumps"] for o in shown + rest if o is not win))}
    if win["votes"] < max(o["votes"] for o in shown):
        # the set itself votes against the pick: more files hold another behaviour green. Where a name can be read
        # two ways (case of vowels, the first Fibonacci number) the pool's own text is the common one, so
        # overturning it takes a third, reasoning opinion
        if may_reason is not None and not may_reason(reason_estimate(prompt)):
            return None, f"minority {tally}: no room for {MODEL_REASON}"
        try:
            third = await _ask(client, MODEL_REASON, prompt, len(shown), reason=True)
        except LLMUnavailable:
            third = None
        if third != pick:
            return None, f"minority {tally} not confirmed by {MODEL_REASON} ({third if third is None else third + 1})"
        return winner, f"all three models chose minority {tally}"
    return winner, f"both models chose {tally}"


def apply(name: str, winner: dict, holders: list, stop: float | None = None) -> list[str]:
    """Put the winning text into every file whose form of `name` is one of `winner["losers"]`, where the visible
    tests stay green with it. No new file is tried after the monotonic time `stop`. Returns the task ids changed."""
    from harness.llm_routes import _run_candidate, write_mylib
    changed = []
    for h in holders:
        if name not in h.forms or h.forms[name][0] not in winner["losers"]:
            continue
        if stop is not None and time.monotonic() > stop:
            break
        try:
            src = (h.ws / "mylib.py").read_text(encoding="utf-8")   # another function may have been replaced since
            unit = next((u for u in cs._units(src) if u.qualname == name), None)
            if unit is None:
                continue
            new = cs._splice(src, unit, winner["text"])
            if new == src or not _run_candidate(h.ws, new).ok:
                continue
            write_mylib(h.ws / "mylib.py", new)
            shutil.rmtree(h.ws / "__pycache__", ignore_errors=True)
            changed.append(h.task.task_id)
        except Exception:
            log.exception("audit: cannot replace %s in %s", name, h.task.task_id)
    return changed

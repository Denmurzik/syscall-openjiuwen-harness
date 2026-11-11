"""End-of-run audit of green coding functions across the task set (HARNESS_AUDIT_FORMS).

The jury's coding tasks are cut from one pool of functions, and hidden tests catch bugs that the one visible assert
does not. For the six functions of the organizers' examples the known text settles it (`_restore_known`). For any
other function the set itself is the evidence: when the same function (same name, same parameters) stands in two or
more different forms in files whose visible tests are all green, and the forms behave differently on sample calls,
at most one of them is right. Forms that give the same results on every sample call are one option, shown as its
most-held form, and its files are never rewritten when it wins; its votes are those of that form alone (a majority is
counted on identical texts only). MODEL_FAST and MODEL_ALT (the options in reverse order) are shown the options and
the differing calls and name one each (8 tokens, as 2c); only when both name the same option is its text put into the
files that hold another shown option, and only where every test file stays green with it. Files whose form was not
shown are never rewritten. An option with fewer votes than another (the set votes against it) wins only when
MODEL_REASON names it too. No rival forms in the set: no call, nothing changes. Any doubt keeps every file as it is.
"""
import ast
import asyncio
import logging
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import ARBITRATE_MODELS, MODEL_REASON
from harness.llm import LLMUnavailable
from harness.solvers import coding_solver as cs
from harness.solvers.probe import behaviour, inputs_from_call, visible_call
from harness.solvers.testrun import run_in_dir

log = logging.getLogger("harness")

MAX_FORMS = 3            # options of one function shown to the models, by votes
MAX_PROBED = 2 * MAX_FORMS   # forms of one function run on the sample calls, most-held first
MAX_ROWS = 6             # differing sample calls shown
MAX_TOKENS = 8
RUN_TIMEOUT = 10.0       # one run of a test file
CENSUS_BUDGET_S = 60.0
MIN_HOLDERS = 2          # files that must hold the function green under a visible test, all forms together
REASON_TOKENS = 1500     # output limit of the third opinion on a minority pick
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LONE = re.compile(r"[^\w]*(?:option\s*)?(\d)[^\w]*", re.I)
_OPTION = re.compile(r"option\s*(\d)", re.I)
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
    test_files: list
    test_srcs: list
    forms: dict = field(default_factory=dict)   # name -> (AST dump, text, tested)
    votes: bool = True   # False: the same code and tests as an earlier holder (it takes replacements, gives no vote)


def _key(ws: Path, test_files: list) -> tuple:
    """Workspaces with the same code and tests (AST) count once, as in `coding_solver.mine_known_texts`."""
    return (ast.dump(ast.parse((ws / "mylib.py").read_bytes())),
            "\0".join(ast.dump(ast.parse((ws / f).read_bytes())) for f in test_files))


def _holder(task, timeout: float = RUN_TIMEOUT) -> Holder | None:
    """One coding workspace whose test files are all green: its top-level self-contained functions defined once,
    undecorated, names the examples do not show."""
    ws = Path(task.workspace)
    if not (ws / "mylib.py").is_file():
        return None
    test_files = sorted(p.name for p in ws.glob("test_*.py"))
    if not test_files:
        return None
    src = (ws / "mylib.py").read_text(encoding="utf-8")
    test_srcs = [(ws / f).read_text(encoding="utf-8") for f in test_files]
    tree = ast.parse(src)
    tested: set = set()
    for test_src in test_srcs:
        refs = cs._test_refs(test_src)
        if any(not r for r in refs.values()):
            return None   # a test that cannot be laid on functions: nothing is known about this file
        skipped = cs._skipped_tests(ast.parse(test_src))
        tested |= set().union(set(), *(r for f, r in refs.items() if f not in skipped))
    if not _green(ws, test_files, timeout):
        return None
    names = cs._module_names(tree)
    defined: dict = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[n.name] = defined.get(n.name, 0) + 1
    h = Holder(task, ws, test_files, test_srcs)
    for node in tree.body:
        if type(node) is not ast.FunctionDef or node.decorator_list or defined[node.name] != 1 \
                or node.name in cs._EXAMPLE_NAMES or not cs._self_contained(node, names):
            continue
        text = ast.get_source_segment(src, node)
        if text:
            h.forms[node.name] = (ast.dump(node), text, node.name in tested)
    return h


def _green(ws: Path, test_files: list, timeout: float, mylib_text: str | None = None) -> bool:
    """Every test file of `ws` passes, run in a throw-away copy (with `mylib_text` as mylib.py when given)."""
    tmp = Path(tempfile.mkdtemp(prefix="audit_"))
    try:
        shutil.copytree(ws, tmp / "ws", ignore=shutil.ignore_patterns("__pycache__"))
        if mylib_text is not None:
            (tmp / "ws" / "mylib.py").write_text(mylib_text, encoding="utf-8", newline="\n")
        return all(run_in_dir(tmp / "ws", f, timeout).ok for f in test_files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def census(tasks: list, budget_s: float = CENSUS_BUDGET_S) -> tuple[list, dict]:
    """(holders, rivals). rivals: name -> the forms to judge [{dump, text, votes, untested}], most-held first, for
    functions that stand in more than one form with the same parameters, green under a visible test in at least
    MIN_HOLDERS files in all. A workspace with the same code and tests as an earlier one is a holder without a vote.
    An unfinished census gives nothing (the files left out could hold the votes against)."""
    stop = time.monotonic() + budget_s
    holders = []
    first: dict = {}   # workspace key -> the holder that voted for it (None: not a holder)
    for t in tasks:
        left = stop - time.monotonic()
        if left <= 0:
            return [], {}
        try:
            ws = Path(t.workspace)
            key = _key(ws, sorted(p.name for p in ws.glob("test_*.py")))
        except Exception:
            continue
        if key in first:
            h0 = first[key]
            if h0 is not None:
                holders.append(Holder(t, ws, h0.test_files, h0.test_srcs, dict(h0.forms), votes=False))
            continue
        try:
            h = _holder(t, min(RUN_TIMEOUT, left))
        except Exception:
            h = None
        first[key] = h
        if h is not None:
            holders.append(h)
    if time.monotonic() > stop:
        return [], {}   # the last run may have been cut by the time left: its file could hold a vote
    forms: dict = {}
    for h in holders:
        for name, (dump, text, tested) in h.forms.items():
            e = forms.setdefault(name, {}).setdefault(dump, {"dump": dump, "text": text, "votes": 0, "untested": 0})
            if h.votes:
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
            for test_src in h.test_srcs:
                vc = visible_call(test_src, name)
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


def options(name: str, forms: list, holders: list) -> tuple[list, list[tuple]]:
    """(shown, rows). The MAX_PROBED most-held forms are run on the sample calls; those that give the same results on
    all of them make one option: its text is its most-held form, its votes that form's alone (behaviour never adds
    votes), `dumps` all its forms. `shown`: the options asked about, by votes. With two or more options tested
    somewhere, the MAX_FORMS best of those; with one, it and up to MAX_FORMS - 1 untested options. Forms of options
    not shown are never rewritten. `rows`: (call, [result per shown option]) for the sample calls on which the shown
    options differ (calls where one raises go last). Fewer than two shown options: nothing to ask."""
    inputs = _inputs(name, holders)
    if not inputs:
        return [], []
    by_outs: dict = {}
    for f in forms[:MAX_PROBED]:   # most-held first: an option's first form is its text and its votes
        outs = tuple(behaviour(f["text"] + "\n", name, inputs))
        o = by_outs.get(outs)
        if o is None:
            o = by_outs[outs] = {"text": f["text"], "dump": f["dump"], "dumps": set(), "votes": f["votes"],
                                 "untested": f["untested"], "outs": outs}
        o["dumps"].add(f["dump"])
    opts = sorted(by_outs.values(), key=lambda o: (-o["votes"], -o["untested"], o["text"]))
    tested = [o for o in opts if o["votes"]]
    untested = [o for o in opts if not o["votes"]]
    shown = tested[:MAX_FORMS] if len(tested) >= 2 else tested + untested[:MAX_FORMS - 1]
    if len(shown) < 2:
        return shown, []
    outs = [o["outs"] for o in shown]
    differ = [k for k in range(len(inputs)) if len({o[k] for o in outs}) > 1]
    rnd = random.Random(0)
    clean = [k for k in differ if not any(o[k].startswith("<exc") for o in outs)]
    if len(clean) > MAX_ROWS:
        clean = sorted(rnd.sample(clean, MAX_ROWS))
    picked = (clean + [k for k in differ if k not in clean])[:MAX_ROWS]
    rows = [(f"{name}({', '.join(map(repr, inputs[k]))})", [o[k][:40] for o in outs]) for k in picked]
    return shown, rows


def _prompt(name: str, opts: list, rows: list[tuple], order: list[int]) -> str:
    """The options and rows in `order` (indices into the shown options)."""
    text = "\n\n".join(f"Option {i + 1}:\n{opts[j]['text'].strip()}" for i, j in enumerate(order))
    lines = [f"{call} -> " + " | ".join(res[j] for j in order) for call, res in rows]
    return (f"Function `{name}`.\n\n{text}\n\nResults of options 1..{len(opts)} on sample calls:\n"
            + "\n".join(lines))


def reason_estimate(prompt: str) -> int:
    """What the ledger must hold for the third opinion: its prompt and its whole output limit."""
    return (len(SYSTEM) + len(prompt)) // 3 + REASON_TOKENS


def unit_estimate(forms: list) -> int:
    """Expected cost of the two arbitration calls on `forms` (the prompt at 3 characters a token)."""
    chars = len(SYSTEM) + sum(len(f["text"]) for f in forms[:MAX_FORMS]) + 80 * MAX_ROWS
    return (chars // 3 + MAX_TOKENS) * len(ARBITRATE_MODELS)


def reason_pick(text: str, n: int) -> int | None:
    """MODEL_REASON's option (0-based): a lone digit (or "Option k") on the last line, else the last "option k" of
    the reply. Anything else: None."""
    text = _THINK.sub("", text or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    m = _LONE.fullmatch(lines[-1])
    if m:
        d = int(m.group(1))
    else:
        found = _OPTION.findall(text)
        if not found:
            return None
        d = int(found[-1])
    return d - 1 if 1 <= d <= n else None


async def _ask(client, model: str, prompt: str, n: int, reason: bool = False) -> int | None:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if reason:
        r = await client.chat(model, messages, REASON_TOKENS, reasoning_effort="low")
        if getattr(r, "truncated", False):
            return None
        return reason_pick(r.text, n)
    kw = {"extra_body": {"guided_regex": "[1-%d]" % n}} if getattr(client, "structured_ok", True) else {}
    r = await client.chat(model, messages, MAX_TOKENS, **kw)
    digits = [c for c in (r.text or "") if c.isdigit()]
    if not digits or not 1 <= int(digits[0]) <= n:
        return None
    return int(digits[0]) - 1


async def decide(client, name: str, forms: list, holders: list, may_reason=None) -> tuple[dict | None, str]:
    """(winner, note). `winner`: {"text", "votes", "losers"}, the option (`options`) both arbitration models named
    and the AST dumps of the forms it replaces: those of the other shown options only. MODEL_FAST sees the options by
    votes, MODEL_ALT in reverse order (its reply is mapped back), so a bias to the first place does not make them
    agree. An option with fewer votes than the top one (the set holds another text green more often) needs
    MODEL_REASON's word too, and `may_reason(estimate)` must allow that call (spend gate, ledger). None: forms that do
    not differ, a model unavailable, an unusable reply, a disagreement, or a minority pick not confirmed.
    BudgetExhausted propagates."""
    shown, rows = await asyncio.to_thread(options, name, forms, holders)
    if not rows:
        return None, "forms do not differ on the sample calls"
    n = len(shown)
    orders = [list(range(n)) if i % 2 == 0 else list(range(n))[::-1] for i in range(len(ARBITRATE_MODELS))]
    asks = [_ask(client, m, _prompt(name, shown, rows, order), n) for m, order in zip(ARBITRATE_MODELS, orders)]
    raw = await asyncio.gather(*asks, return_exceptions=True)
    for r in raw:
        if isinstance(r, BaseException) and not isinstance(r, LLMUnavailable):
            raise r
    replies = [r if isinstance(r, BaseException) or r is None else order[r] for r, order in zip(raw, orders)]
    votes = ", ".join("unavailable" if isinstance(r, BaseException) else ("unusable" if r is None else str(r + 1))
                      for r in replies)
    if any(isinstance(r, BaseException) or r is None for r in replies) or len(set(replies)) != 1:
        return None, f"no agreement ({votes}) on {[o['votes'] for o in shown]}"
    pick = replies[0]
    win = shown[pick]
    tally = f"option {pick + 1} of votes {[o['votes'] for o in shown]}"
    winner = {"text": win["text"], "votes": win["votes"],
              "losers": set().union(*(o["dumps"] for o in shown if o is not win))}
    if win["votes"] < max(o["votes"] for o in shown):
        # the set itself votes against the pick: more files hold another text green. Where a name can be read two
        # ways (case of vowels, the first Fibonacci number) the pool's own text is the common one, so overturning
        # it takes a third, reasoning opinion
        prompt = _prompt(name, shown, rows, orders[0])
        if may_reason is not None and not may_reason(reason_estimate(prompt)):
            return None, f"minority {tally}: no room for {MODEL_REASON}"
        try:
            third = await _ask(client, MODEL_REASON, prompt, n, reason=True)
        except LLMUnavailable:
            third = None
        if third != pick:
            return None, f"minority {tally} not confirmed by {MODEL_REASON} ({third if third is None else third + 1})"
        return winner, f"all three models chose minority {tally}"
    return winner, f"both models chose {tally}"


def apply(name: str, winner: dict, holders: list, stop: float | None = None) -> list[str]:
    """Put the winning text into every file whose form of `name` is one of `winner["losers"]`, where every test file
    stays green with it. No new file is tried after the monotonic time `stop`. Returns the task ids changed."""
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
            if new == src or not _green(h.ws, h.test_files, 20.0, new):
                continue
            cs._write_mylib(h.ws / "mylib.py", new)
            shutil.rmtree(h.ws / "__pycache__", ignore_errors=True)
            changed.append(h.task.task_id)
        except Exception:
            log.exception("audit: cannot replace %s in %s", name, h.task.task_id)
    return changed

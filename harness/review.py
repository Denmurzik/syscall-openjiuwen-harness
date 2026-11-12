"""Review of answered coding tasks for bugs the visible tests do not catch (expert 9, idea a).

The jury's hidden tests check inputs the visible tests do not. Phase 1 and the ladder stop at the first green run, so a
bug that no visible assert reaches stays in the file. Here MODEL_FAST reads the green file once and rewrites the
functions it finds wrong; a silent reply ("NONE") ends the review. A function is replaced only when MODEL_ALT, asked
the same on its own, rewrote that function too, the two rewrites behave alike on trial inputs wherever the file's
function returns a value, on one of them they differ from it, and the visible tests stay green (`_evidence`). A known
text (the examples' functions, texts mined from the set) and a method are never replaced. Any doubt keeps the file.

Round 19 (task 82, HARNESS_REVIEW_REASON): where FAST and ALT both rewrote a function and their rewrites disagree on
the trial inputs, MODEL_REASON is asked the same, once per file; a function is replaced when its rewrite and one of
theirs pass the same rules."""
import ast
import asyncio
import re
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

from harness import config
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from harness.llm import LLMUnavailable
from harness.llm_routes import (_FUNC_HEAD, _THINK, _cap, _first_def, _target_file, _test_file,
                                apply_patch, write_mylib)
from harness.solvers import coding_solver as cs
from harness.solvers.probe import _MAX_REPR, behaviour, visible_call
from harness.solvers.testrun import run_in_dir

REVIEW_SYSTEM = (
    "You review a Python file whose visible tests all pass. The tests are only examples: hidden tests call the same "
    "functions with other inputs (edge cases, empty and repeated values, larger numbers, other branches). "
    "For each function decide whether it does what its name, docstring and comments say for every valid input. "
    "Return the corrected text of each function that is wrong, no explanations, in this format, one block per "
    "function:\nFUNC <name>\n<complete new text of the function>\n"
    "For a method use FUNC <Class.method>. Keep the signature and the docstring. Do not touch a function that is "
    "right: no restyling, no renaming, no checks or behaviour the docstring does not ask for. "
    "If every function is right, reply with the single word NONE.")
REVIEW_MAX_TOKENS = 700
MAX_FILE_CHARS = 9000   # a larger file is not reviewed: the reply could not hold its functions
RUN_TIMEOUT = 20        # one run of one visible test file
PROBE_TIMEOUT = 10      # the trial inputs of one version of the function
MIN_LEFT_S = 60         # no model call with less time than this before the stop: its rewrite could not be checked
REASON_TOKENS = 3000    # output limit of MODEL_REASON's third opinion (low reasoning effort, then the FUNC blocks)
REASON_SPEND_LIMIT = 0.5   # no third opinion once this share of the budget is spent (the stage's own gate)


def _left(stop: float | None, cap: float) -> float:
    """Seconds a step may take: `cap`, cut to what is left before the monotonic time `stop`."""
    return cap if stop is None else min(cap, stop - time.monotonic())


def _test_files(ws: Path) -> list[str]:
    """Every visible test file of the workspace (as the audit runs them), else phase 1's one."""
    names = sorted({p.name for p in ws.glob("test_*.py")} | {p.name for p in ws.glob("*_test.py")})
    return names or [_test_file(ws)]


def _green(ws: Path, text: str, target: str, stop: float | None = None) -> bool:
    """Every visible test file passes with `text` as the target file, run in a throw-away copy; out of time is
    not green."""
    tmp = Path(tempfile.mkdtemp(prefix="review_"))
    try:
        shutil.copytree(ws, tmp / "ws", ignore=shutil.ignore_patterns("__pycache__"))
        (tmp / "ws" / target).write_text(text, encoding="utf-8", newline="\n")
        for f in _test_files(ws):
            limit = _left(stop, RUN_TIMEOUT)
            if limit <= 1 or not run_in_dir(tmp / "ws", f, limit).ok:
                return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _texts(src: str) -> dict:
    return {u.qualname: textwrap.dedent(cs._segment(src, u)).strip("\n") for u in cs._units(src)}


def known_texts(mined: dict | None = None) -> dict:
    """name -> known text: the examples' and the final texts mined from the set (`scheduler._mine_texts`:
    {"texts": ..., "final": [...]}). A mined text that is not final can be the silent bug the set's tasks share
    (`sorted(set(xs))` for an order-keeping dedupe): it stays open to the review. The main process never installs the
    mined texts (other in-process code reads the known tables), so they come here as an argument. Example names keep
    their texts."""
    out = dict(cs._CANON_TEXT)
    texts = (mined or {}).get("texts") if isinstance(mined, dict) else None
    final = set((mined or {}).get("final") or ()) if isinstance(mined, dict) else set()
    for name, text in (texts or {}).items():
        if isinstance(name, str) and isinstance(text, str) and name in final and name not in cs._EXAMPLE_NAMES:
            out[name] = text
    return out


def needs_review(src: str, known: dict | None = None) -> bool:
    """False for a file whose every function is a known text (`known_texts`: the examples' six, the texts mined from
    the set): their silent slips are phase 1's work, at no cost."""
    known = cs._CANON_TEXT if known is None else known
    try:
        units = cs._units(src)
    except (SyntaxError, ValueError, RecursionError):
        return False
    if not units or len(src) > MAX_FILE_CHARS:
        return False
    for u in units:
        text = known.get(u.qualname)
        if text is None or cs._norm(text) != cs._norm(cs._segment(src, u)):
            return True
    return False


def estimate(src: str, tests: str) -> int:
    return (len(src) + len(tests) + len(REVIEW_SYSTEM)) // 30 + 150


def _messages(src: str, tests: str, target: str, test_name: str) -> list[dict]:
    user = f"{target}:\n{_cap(src)}\n\n{test_name} (all pass):\n{_cap(tests)}"
    return [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}]


_EXTRA = re.compile(r"^[ \t]*(?:import[ \t]+\w|from[ \t]+[\w.]+[ \t]+import\b|(?:async[ \t]+)?def[ \t]+\w|class[ \t]+\w|@)")


def _dropped(src: str, reply: str) -> set:
    """Names whose FUNC block holds code `apply_patch` drops: an import before or after the function, a helper
    function or class next to it. The kept function would call a name the file does not define. An import line the
    file already has, or a helper the file already holds in the same text, is no loss."""
    patch = _THINK.sub("", reply.replace("\r\n", "\n"))
    heads = list(_FUNC_HEAD.finditer(patch))
    have = {ln.strip() for ln in src.split("\n")}
    try:
        old = _texts(src)
    except (SyntaxError, ValueError, RecursionError):
        return {h.group(1) for h in heads}
    out = set()
    for i, h in enumerate(heads):
        lines = patch[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(patch)].split("\n")
        found = _first_def("\n".join(lines))
        if found is None:
            continue
        start = next(k for k, ln in enumerate(lines) if ln.lstrip().startswith(("def ", "async def ", "@")))
        rest = lines[:start] + lines[start + len(found[0].split("\n")):]
        for k, ln in enumerate(rest):
            if not _EXTRA.match(ln):
                continue
            if ln.lstrip().startswith(("import", "from")) and ln.strip() in have:
                continue
            if ln.lstrip().startswith(("def ", "async def ")):
                helper = _first_def("\n".join(rest[k:]))
                if helper is not None:
                    hname = helper[1].body[0].name
                    if hname in old and cs._norm(old[hname]) == cs._norm(helper[0]):
                        continue
            out.add(h.group(1))
            break
    return out


def _rewrites(src: str, reply: str) -> dict:
    """qualname -> new text, for the functions of `reply` that differ from the file's (as code, not as spelling).
    A function whose block carries code that would be dropped (`_dropped`) is left out."""
    if reply is None or reply.strip().strip(".`'\"").upper() == "NONE":
        return {}
    patched = apply_patch(src, reply)
    if patched is None:
        return {}
    old, new = _texts(src), _texts(patched)
    lost = _dropped(src, reply)
    return {n: t for n, t in new.items() if n in old and n not in lost and cs._norm(old[n]) != cs._norm(t)}


def _spliced(src: str, name: str, text: str) -> str | None:
    for u in cs._units(src):
        if u.qualname == name:
            out = cs._splice(src, u, text)
            try:
                ast.parse(out)
            except (SyntaxError, ValueError):
                return None
            return out
    return None


_INTS = [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 25, 30, 45, 100, 400, 1900, 2000, 2023, 2024, -1, -7]
_STRS = ["", "a", "abc", "Hello World", "a b a", "The the THE cat", "aab", "  two  spaces ", "xyz XYZ", "racecar",
         "a=1; b = 2", "aaabcc", "listen", "Silent"]
_INT_LISTS = [[], [1], [1, 2, 3], [3, 1, 2], [2, 2, 1], [5, 5, 3], [1, 2, 3, 4, 5], [-2, -5, -1], [3, 3, 1, 1], [0, 0],
              [4, 1, 3, 2], [7, 7, 7, 6, 1], [10, -10, 10]]
_STR_LISTS = [[], ["a"], ["b", "a", "b"], ["x", "Y", "x"], ["bb", "a", "ccc"]]
_NESTED = [[], [[1, 2], [2, 3]], [[1, 5], [2, 3]], [[5, 8], [1, 3]], [1, [2, [3, [4]]]], [[1], [2, 3], []], [[3, 4], [1, 2]]]
BATTERY_CAP = 400


def _pool(v) -> list:
    """Values of the shape of one visible argument."""
    if isinstance(v, bool):
        return [True, False]
    if isinstance(v, int):
        return _INTS + [v + 1, v - 1, v * 2, v + 7]
    if isinstance(v, float):
        return [0.0, 0.5, 1.0, 2.5, 10.0, -1.5, 100.0, v * 2]
    if isinstance(v, str):
        return _STRS + [v.upper(), v.title(), v[::-1], v + " " + v, v[:1]]
    if isinstance(v, list):
        own = [v[::-1], v + v[:1], v[:1], v + v]
        if any(isinstance(x, (list, tuple)) for x in v):
            return _NESTED + own
        if v and all(isinstance(x, str) for x in v):
            return _STR_LISTS + own
        return _INT_LISTS + own
    return [v]


def _battery(args: tuple | None, arity: int) -> list[tuple]:
    """Trial inputs: with a visible call, values of the same shapes (one argument varied at a time, then pairs of
    positions together); without one, common shapes for the number of parameters."""
    out: list[tuple] = []
    if args is not None:
        pools = [_pool(a) for a in args]
        out.append(tuple(args))
        for i, pool in enumerate(pools):
            out += [tuple(args[:i]) + (x,) + tuple(args[i + 1:]) for x in pool]
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                for x in pools[i][:22:2]:
                    for y in pools[j][:22]:
                        t = list(args)
                        t[i], t[j] = x, y
                        out.append(tuple(t))
    elif arity == 1:
        out = [(x,) for x in _INTS + _STRS + _INT_LISTS + _STR_LISTS + _NESTED]
    elif arity == 2:
        for xs, ys in ((_INTS, _INTS), (_INT_LISTS, _INTS[:10]), (_STRS, _INTS[:10]), (_STRS, _STRS),
                       (_INT_LISTS, _INT_LISTS)):
            out += [(x, y) for x in xs[:12] for y in ys[:12]]
    elif 3 <= arity <= 4:
        small = [0, 1, 2, 5, 10]
        out = [tuple(small[(i >> (2 * k)) % 5] for k in range(arity)) for i in range(0, 5 ** arity, 3)]
    seen, uniq = set(), []
    for t in out:
        key = repr(t)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq[:BATTERY_CAP]


def _arity(src: str, name: str) -> int:
    """Positional parameters without defaults of a top-level function; -1 when it takes *args or keyword-only ones."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            if a.vararg or a.kwarg or a.kwonlyargs:
                return -1
            return len(a.posonlyargs) + len(a.args) - len(a.defaults)
    return -1


# exceptions a correct function raises on a value outside its domain: the only ones that count as behaviour
_VALUE_EXC = frozenset({"ValueError", "KeyError", "IndexError", "ZeroDivisionError"})
# a probe value of a type the function does not take: harmless where the file's function raises on it too
_SHAPE_EXC = frozenset({"TypeError", "AttributeError"})


def _exc(out: str) -> str | None:
    return out[5:-1] if out.startswith("<exc:") and out.endswith(">") else None


def _broken(ours: list, theirs: list) -> bool:
    """The rewrite raises, where the file's function does not raise the same, an exception that shows the rewrite
    broken rather than different: a name it does not define (an import or helper that was dropped), a timeout, a
    crash, a recursion, anything but a value-domain error. A type error counts only where the file's function
    returns a value (the probe value fits the function)."""
    for o, t in zip(ours, theirs):
        e = _exc(t)
        if e is None or e in _VALUE_EXC or t == o:
            continue
        if e in _SHAPE_EXC and o.startswith("<exc:"):
            continue
        return True
    return False


class _Trial:
    """Trial inputs of one function of the file and each version's outcomes on them, one subprocess a version,
    remembered by the file's text (the third opinion probes the same versions again). `ok`: the function can be
    probed (not a method, at least 3 inputs)."""

    def __init__(self, src: str, tests: str, name: str):
        self.src, self.name, self.memo = src, name, {}
        self.inputs: list = []
        if "." in name:
            return
        call = visible_call(tests, name)
        try:
            self.inputs = _battery(call[0] if call else None, _arity(src, name))
        except (SyntaxError, ValueError, RecursionError, TypeError):
            self.inputs = []

    @property
    def ok(self) -> bool:
        return len(self.inputs) >= 3

    def cut(self, out: str) -> bool:
        """A value cut at the probe's repr limit: it shows neither agreement nor a difference."""
        return len(out) >= _MAX_REPR

    def outcomes(self, text: str | None, stop: float | None) -> list[str] | None:
        """The outcomes of the file with `text` as the function (None: the file's own); None when out of time."""
        full = self.src if text is None else _spliced(self.src, self.name, text)
        if full is None:
            return None
        if full not in self.memo:
            limit = _left(stop, PROBE_TIMEOUT)
            if limit < PROBE_TIMEOUT / 2:
                return None
            self.memo[full] = behaviour(full, self.name, self.inputs, timeout=limit)
        return self.memo[full]


def _verdict(src: str, tests: str, name: str, a: str, b: str, stop: float | None = None,
             trial: "_Trial | None" = None) -> tuple[list[str], bool]:
    """(`_evidence`, disagree): disagree is True when the two rewrites are two opinions rather than one: one of them
    shows itself broken and the other does not, they return different values where the file's function returns one,
    or one raises where the other returns a value. Only then can a third opinion side with one of them."""
    trial = trial if trial is not None else _Trial(src, tests, name)
    if not trial.ok or _spliced(src, name, a) is None or _spliced(src, name, b) is None:
        return [], False
    outs = []
    for text in (None, a, b):
        o = trial.outcomes(text, stop)
        if o is None:
            return [], False
        outs.append(o)
    ours, ba, bb = outs
    broken = _broken(ours, ba), _broken(ours, bb)
    if broken[0] or broken[1]:
        return [], broken[0] != broken[1]
    valid = [i for i, o in enumerate(ours) if not o.startswith("<exc:")]
    # where one rewrite raises and the other returns, or they raise different exceptions, they differ on what to do
    # with a degenerate input ([5, 5] for "second largest distinct"): that is not a disagreement about the function
    raises = [(ba[i].startswith("<exc:"), bb[i].startswith("<exc:")) for i in range(len(ours))]
    # a value cut at the probe's repr limit can hide a difference after the cut: such an input shows neither
    # agreement nor a difference, it is set aside too
    both = [i for i in valid if raises[i][0] == raises[i][1] and not trial.cut(ba[i]) and not trial.cut(bb[i])]
    if len(both) < 3 or 2 * len(both) < len(valid):
        return [], len(valid) >= 3 and any(raises[i][0] != raises[i][1] for i in valid)
    if any(ba[i] != bb[i] and not raises[i][0] for i in both):
        return [], True
    if not any(ba[i] == bb[i] != ours[i] and (_exc(ba[i]) is None or _exc(ba[i]) in _VALUE_EXC) for i in both):
        return [], False
    ok = [t for t, k in ((a, 0), (b, 1)) if not any(raises[i][k] for i in valid)]
    if ok:
        return ok, False
    # both rewrites raise where ours returns a value: stored only when they raise the same value-domain exception
    # on every such input, each raises it explicitly, and the function's text or its visible tests name it
    spots = [i for i in valid if raises[i][0] or raises[i][1]]
    excs = {_exc(ba[i]) for i in spots}
    if not all(ba[i] == bb[i] and _exc(ba[i]) in _VALUE_EXC for i in spots):
        return [], False
    seg = _texts(src).get(name, "")
    named = [ln for ln in tests.splitlines() if name in ln]
    if not all(e in seg or any(e in ln for ln in named) for e in excs):
        return [], False
    return [t for t in (a, b) if all(re.search(r"\braise\s+" + e + r"\b", t) for e in excs)], False


def _evidence(src: str, tests: str, name: str, a: str, b: str, stop: float | None = None) -> list[str]:
    """The rewrites of one function that may be stored, best first; [] keeps the file. The two rewrites are one
    opinion and the file's function is shown wrong. Trial inputs: the visible call and values of its shapes (or
    common shapes for a function the tests do not call). A rewrite that shows itself broken (`_broken`) ends it. On
    every input where the file's function returns a value the two rewrites must return the same value or both raise
    (inputs where only one of them raises are set aside, at most half), and on at least one input their common
    outcome must differ from what the file's function returns. So a rewrite that only changes what happens where the
    function raises today (a guard for empty input, another exception type) is no evidence. Only a rewrite that
    raises on no input where the file's function returns a value is stored: one that drops a guard of ours would
    turn a hidden test of that input red. The one exception: where both rewrites raise the same value-domain
    exception on every such input, the function's text or a visible test line naming it names that exception
    ("ValueError if the lengths differ"), and the stored text says `raise <Name>`. A method cannot be probed: kept.
    Out of time before `stop`: kept."""
    return _verdict(src, tests, name, a, b, stop)[0]


def _known(src: str, name: str, known: dict | None = None) -> bool:
    """The file holds the known text of this function (an example's, or one mined from the set): never replaced."""
    text = (cs._CANON_TEXT if known is None else known).get(name)
    return text is not None and cs._norm(text) == cs._norm(_texts(src).get(name, ""))


def _decide(ws: Path, src: str, tests: str, target: str, first: dict, second: dict,
            known: dict | None = None, stop: float | None = None, disputed: list | None = None,
            trials: dict | None = None) -> tuple[str, list[str]]:
    """The file with every agreed rewrite that keeps the visible tests green, and their names. No step starts after
    the monotonic time `stop`. `disputed` (if given) collects the names whose two rewrites disagree (`_verdict`);
    `trials` (if given) keeps each name's `_Trial` for a later look at the same versions."""
    cur, changed = src, []
    trials = {} if trials is None else trials
    for name in sorted(set(first) & set(second)):
        if stop is not None and time.monotonic() > stop:
            break
        if _known(src, name, known):
            continue
        trial = trials.setdefault(name, _Trial(src, tests, name))
        texts, split = _verdict(src, tests, name, first[name], second[name], stop, trial)
        for text in texts:
            cand = _spliced(cur, name, text)
            if cand is not None and _green(ws, cand, target, stop):
                cur = cand
                changed.append(name)
                break
        if split and disputed is not None:
            disputed.append(name)
    return cur, changed


def _settle(ws: Path, src: str, cur: str, tests: str, target: str, first: dict, second: dict, third: dict,
            names: list, trials: dict, stop: float | None = None) -> tuple[str, list[str]]:
    """The disputed functions (`names`) where the third rewrite and FAST's, else ALT's, pass `_verdict` as one
    opinion: that pair's text goes into `cur` when the visible tests stay green."""
    changed = []
    for name in names:
        if name not in third:
            continue
        for other in (first[name], second[name]):
            if stop is not None and time.monotonic() > stop:
                return cur, changed
            texts, _ = _verdict(src, tests, name, third[name], other, stop, trials.get(name))
            for text in texts:
                cand = _spliced(cur, name, text)
                if cand is not None and _green(ws, cand, target, stop):
                    cur = cand
                    changed.append(name)
                    break
            if name in changed:
                break
    return cur, changed


def _may_reason(client, msgs: list[dict]) -> bool:
    """The third opinion fits the stage's gates: under REASON_SPEND_LIMIT of the budget and affordable (a client
    without a ledger, as in the tests, always may)."""
    ledger = getattr(client, "ledger", None)
    if ledger is None:
        return True
    est = sum(len(m["content"]) for m in msgs) // 40 + REASON_TOKENS
    return not ledger.exhausted and ledger.spent < REASON_SPEND_LIMIT * ledger.budget and ledger.can_afford(est)


async def _third(client, msgs: list[dict], src: str) -> tuple[dict, str]:
    """MODEL_REASON's rewrites on the same prompt (no other reply shown), and a word for the note."""
    try:
        r = await client.chat(MODEL_REASON, msgs, REASON_TOKENS, reasoning_effort="low")
    except LLMUnavailable:
        return {}, "unavailable"
    if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= REASON_TOKENS:
        return {}, "hit the output limit"
    third = _rewrites(src, r.text)
    return third, str(sorted(third))


async def review(task, client, known: dict | None = None, stop: float | None = None) -> tuple[list[str], str]:
    """Names of the functions replaced in the task's file, and a note for the trajectory. `known`: `known_texts`
    (default the examples' texts only). `stop`: the monotonic time after which no probe or test run starts."""
    ws = Path(task.workspace)
    try:
        test_name = _test_file(ws)
        tests = (ws / test_name).read_text(encoding="utf-8")
        target = _target_file(ws, tests)
        path = ws / target
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [], f"review skipped: {e!r}"[:200]
    if not await asyncio.to_thread(_green, ws, src, target, stop):
        return [], "review skipped: the tests are not green"
    if stop is not None and stop - time.monotonic() < MIN_LEFT_S:
        return [], "review: too close to the stop to check a rewrite, kept"
    msgs = _messages(src, tests, target, test_name)
    try:
        r = await client.chat(MODEL_FAST, msgs, REVIEW_MAX_TOKENS)
        if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= REVIEW_MAX_TOKENS:
            return [], "review: reply hit the output limit, kept"
        first = _rewrites(src, r.text)
        if not first:
            return [], "review: nothing found"
        r = await client.chat(MODEL_ALT, msgs, REVIEW_MAX_TOKENS)
        if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= REVIEW_MAX_TOKENS:
            return [], "review: second reply hit the output limit, kept"
        second = _rewrites(src, r.text)
    except LLMUnavailable as e:
        return [], f"review: llm unavailable: {e}"[:200]
    if not set(first) & set(second):
        return [], f"review: no agreement ({sorted(first)} / {sorted(second)}), kept"[:200]
    disputed, trials = [], {}
    new, changed = await asyncio.to_thread(_decide, ws, src, tests, target, first, second, known, stop, disputed,
                                           trials)
    tail = ""
    if disputed and config.review_reason():
        if stop is not None and stop - time.monotonic() < MIN_LEFT_S:
            tail = f"; {sorted(disputed)} disputed, too close to the stop for {MODEL_REASON}"
        elif not _may_reason(client, msgs):
            tail = f"; {sorted(disputed)} disputed, no room for {MODEL_REASON}"
        else:
            third, word = await _third(client, msgs, src)
            new, more = await asyncio.to_thread(_settle, ws, src, new, tests, target, first, second, third,
                                                sorted(disputed), trials, stop)
            changed += more
            tail = f"; {sorted(disputed)} disputed, {MODEL_REASON} {word}, sided {sorted(more)}"
    if not changed or new == src:
        return [], f"review: no evidence against {sorted(set(first) & set(second))}{tail}, kept"[:300]
    if path.read_text(encoding="utf-8") != src:   # someone else rewrote the file meanwhile: leave it
        return [], "review: the file changed meanwhile, kept"
    write_mylib(path, new)
    shutil.rmtree(ws / "__pycache__", ignore_errors=True)
    return changed, f"review: replaced {changed} ({sorted(first)} / {sorted(second)}){tail}"[:300]

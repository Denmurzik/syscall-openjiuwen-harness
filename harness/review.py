"""Review of answered coding tasks for bugs the visible tests do not catch (expert 9, idea a).

The jury's hidden tests check inputs the visible tests do not. Phase 1 and the ladder stop at the first green run, so a
bug that no visible assert reaches stays in the file. Here MODEL_FAST reads the green file once and rewrites the
functions it finds wrong; a silent reply ("NONE") ends the review. A function is replaced only when MODEL_ALT, asked
the same on its own, rewrote that function too, the two rewrites behave alike on trial inputs wherever the file's
function returns a value, on one of them they differ from it, and the visible tests stay green (`_evidence`). A known
text (the examples' functions, texts mined from the set) and a method are never replaced. Any doubt keeps the file.

Round 19 (task 82, HARNESS_REVIEW_REASON): where FAST and ALT both rewrote a function and their rewrites disagree on
the trial inputs, MODEL_REASON is asked the same about those functions only (`_focus`), once per file; a function is
replaced when its rewrite and one of theirs pass the same rules. Task 87: a function only one of them rewrote
(one-sided) is asked about too, after the disputes, when that rewrite alone would pass the rules (`_lone`).

Round 19 (task 84, HARNESS_REVIEW_METHODS): a method is checked too, on scenarios built from the visible tests that
call it (`_scenes`): the objects are built and the methods called as the tests do, with other literal values, and the
values the tests look at, the calls' results and the objects' public attributes are compared under the same rules. A
method no visible test calls has no scenario: it is dropped before MODEL_ALT is asked."""
import ast
import asyncio
import copy
import itertools
import json
import re
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

from harness import config
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON
from harness.llm import LLMUnavailable
from harness.llm_routes import (_FUNC_HEAD, _THINK, _cap, _deep_tests, _first_def, _read_tests, _target_file,
                                _test_file, apply_patch, write_mylib)
from harness.solvers import coding_solver as cs
from harness.proc import run_python
from harness.solvers.probe import _ADDR, _MAX_REPR, behaviour, visible_call
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
HANG_CONFIRM_S = 5.0    # HARNESS_REVIEW_PROBES: the alarm of the run that confirms a hang of the file, per input
HANG_CONFIRM_MAX = 2    # inputs confirmed at most (each may take HANG_CONFIRM_S)
MIN_LEFT_S = 60         # no model call with less time than this before the stop: its rewrite could not be checked
REASON_TOKENS = 2000    # output limit of MODEL_REASON's third opinion (low reasoning effort, then the FUNC blocks)
REASON_SPEND_LIMIT = 0.5   # no third opinion once this share of the budget is spent (the stage's own gate)
REASON_MAX_CALLS = 6       # third opinions a run, at most (1 to 2K units each: a set where every file disputes would
REASON_MAX_UNITS = 8000    # otherwise be held back by the 50 % gate alone), and the units they may spend in all
REASON_DISPUTE_CALLS = 2   # of the cap, kept for files with a dispute: a file with only one-sided cases (task 87)
REASON_DISPUTE_UNITS = 2000  # may not take them


def _left(stop: float | None, cap: float) -> float:
    """Seconds a step may take: `cap`, cut to what is left before the monotonic time `stop`."""
    return cap if stop is None else min(cap, stop - time.monotonic())


def _test_files(ws: Path) -> list[str]:
    """Every visible test file of the workspace (as the audit runs them), else those below the root
    (HARNESS_DEEP_TESTS), else phase 1's one."""
    names = sorted({p.name for p in ws.glob("test_*.py")} | {p.name for p in ws.glob("*_test.py")})
    return names or _deep_tests(ws) or [_test_file(ws)]


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


_NUM_TEXT = ["0", "1", "-1", "7", "10", "99", "100", "150", "151", "999", "1000", "0.5", "2.5", "-3.5", "12.50"]


def _str_variants(v: str) -> list:
    """HARNESS_REVIEW_PROBES: strings of the same build as a visible argument, where the generic strings only make
    the function raise: each sign flipped ('+02:00' -> '-02:00'), each run of digits replaced by other numbers
    (' 42 ' -> ' 151 ', ' -1 '), the same text with its blanks stripped. For a text that is one number, other
    numbers as text. At most 40."""
    if not config.review_probes() or not v or len(v) > 80:
        return []
    out = []
    for i, ch in enumerate(v):
        if ch in "+-" and len(out) < 6:
            out.append(v[:i] + ("-" if ch == "+" else "+") + v[i + 1:])
    runs = list(re.finditer(r"\d+", v))
    for m in runs[:5]:
        n = int(m.group())
        width = len(m.group())
        for alt in (n + 1, max(n - 1, 0), 0, n * 10 + 9, 151, 999):
            out.append(v[:m.start()] + str(alt).zfill(width) + v[m.end():])
    if len(runs) == 1 and not re.search(r"[+-]\s*\d", v):
        m = runs[0]
        out.append(v[:m.start()] + "-" + m.group() + v[m.end():])
        out.append(v[:m.start()] + "-1" + v[m.end():])
    core = v.strip()
    if core != v:
        out.append(core)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", core):
        out += _NUM_TEXT + [f" {core} ", core + "x"]
    seen, uniq = {v}, []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:40]


def _pool(v) -> list:
    """Values of the shape of one visible argument."""
    if isinstance(v, bool):
        return [True, False]
    if isinstance(v, int):
        return _INTS + [v + 1, v - 1, v * 2, v + 7]
    if isinstance(v, float):
        return [0.0, 0.5, 1.0, 2.5, 10.0, -1.5, 100.0, v * 2]
    if isinstance(v, str):
        return _STRS + [v.upper(), v.title(), v[::-1], v + " " + v, v[:1]] + _str_variants(v)
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


# ---- HARNESS_REVIEW_PROBES: visible arguments that are not literals (Decimal("10.00"), date(2024, 2, 10)) ----
_CTOR_MODULES = frozenset({"decimal", "datetime", "fractions"})
_DEC = ["0", "1", "0.5", "0.05", "0.125", "1.005", "2.675", "12.5", "10.00", "-1.5", "100", "0.01", "0.015", "7",
        "33.3", "1.25", "20"]
_DATES = [(2024, 2, 29), (2023, 2, 28), (2024, 12, 31), (2023, 12, 1), (2024, 1, 1), (2024, 1, 31), (2024, 5, 10),
          (2024, 5, 11), (2024, 5, 12), (2024, 5, 13), (2024, 3, 31), (2024, 11, 30), (2025, 6, 15), (2000, 2, 29),
          (1999, 12, 31)]
_TIMES = [(12, 0), (23, 59), (0, 0), (6, 15), (18, 45)]
# Reads {"src", "func", "head", "inputs", "alarm"}: `head` is the test file's own imports of decimal, datetime and
# fractions, each input the source of one argument tuple, evaluated with those names; `alarm`: seconds per input
# (1.0). Output as probe._CHILD's, except that the alarm is the review's own: a BaseException that the function's
# `except Exception` does not swallow, shown as "<exc:ProbeAlarm>" (a `raise TimeoutError` of the function stays
# "<exc:TimeoutError>")
_EXPR_CHILD = r"""
import copy, io, json, sys
req = json.loads(sys.stdin.read())
real = sys.stdout
def emit(s):
    real.write("@@PROBE@@" + json.dumps(s) + "\n"); real.flush()
class _ProbeAlarm(BaseException):
    pass
try:
    import signal
    def _alarm(*a):
        raise _ProbeAlarm()
    signal.signal(signal.SIGALRM, _alarm)
    have_alarm = True
except Exception:
    have_alarm = False
sys.stdout = io.StringIO()
ns, env = {"__name__": "mylib"}, {}
try:
    exec(req["head"], env)
    exec(compile(req["src"], "mylib.py", "exec"), ns)
    fn = ns[req["func"]]
except BaseException as e:
    for _ in req["inputs"]:
        emit("<exc:%s>" % type(e).__name__)
    sys.exit(0)
for text in req["inputs"]:
    try:
        args = eval(text, env)
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, float(req.get("alarm", 1.0)))
        out = repr(fn(*copy.deepcopy(args)))[:LIMIT]
    except _ProbeAlarm:
        out = "<exc:ProbeAlarm>"
    except BaseException as e:
        out = "<exc:%s>" % type(e).__name__
    finally:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    sys.stdout = io.StringIO()
    emit(out)
""".replace("LIMIT", str(_MAX_REPR))


def _expr_behaviour(src: str, func: str, head: str, inputs: list[str], timeout: float,
                    alarm: float = 1.0) -> list[str]:
    """`probe.behaviour` for inputs given as source text (`_visible_exprs`), `alarm` seconds per input."""
    if not inputs:
        return []
    req = json.dumps({"src": src, "func": func, "head": head, "inputs": list(inputs), "alarm": alarm})
    with tempfile.TemporaryDirectory(prefix="probe_") as d:
        res = run_python(["-c", _EXPR_CHILD], cwd=Path(d), timeout=timeout, stdin=req, env={"PYTHONHASHSEED": "0"})
    out: list[str] = []
    for line in res.stdout.splitlines():
        if line.startswith("@@PROBE@@"):
            try:
                out.append(_ADDR.sub("", str(json.loads(line[9:]))))
            except ValueError:
                break
    fill = "<exc:Timeout>" if res.timed_out else "<exc:Crash>"
    return (out + [fill] * len(inputs))[:len(inputs)]


def _ctor(node) -> str | None:
    """'Decimal', 'date', ... for a call `Decimal("1.0")`, `datetime.date(2024, 1, 2)` with literal arguments."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
    if name not in ("Decimal", "Fraction", "date", "datetime", "timedelta"):
        return None
    try:
        for a in [*node.args, *(k.value for k in node.keywords)]:
            ast.literal_eval(a)
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        return None
    return name


def _expr_pool(text: str) -> list[str]:
    """Other values of the kind of one visible argument, as source text."""
    node = ast.parse(text, mode="eval").body
    kind = _ctor(node)
    if kind is None:
        return [repr(x) for x in _pool(ast.literal_eval(node))]
    fn = ast.get_source_segment(text, node.func)
    if kind == "Decimal":
        return [f"{fn}({x!r})" for x in _DEC]
    if kind == "Fraction":
        return [f"{fn}({a}, {b})" for a, b in ((0, 1), (1, 2), (1, 3), (2, 4), (-3, 4), (7, 1), (5, 10), (22, 7))]
    if kind == "date":
        return [f"{fn}({y}, {m}, {d})" for y, m, d in _DATES]
    if kind == "datetime":
        return [f"{fn}({y}, {m}, {d}, {hh}, {mm})" for (y, m, d), (hh, mm) in zip(_DATES, _TIMES * 3)]
    return [f"{fn}(days={d})" for d in (0, 1, 7, 30, 365, -1)] + [f"{fn}(hours={h})" for h in (1, 25, -3)]


def _visible_exprs(tests: str, name: str) -> tuple[str, list[str]] | None:
    """(head, argument sources) of the first call of `name` in the tests whose arguments are literals or Decimal /
    Fraction / date / datetime / timedelta built from literals, at least one of them built; `head`: the test file's
    imports of decimal, datetime and fractions. None when there is no such call."""
    try:
        tree = ast.parse(tests)
    except (SyntaxError, ValueError, RecursionError):
        return None
    head = []
    for node in tree.body:
        mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
        if mods and all(m.split(".")[0] in _CTOR_MODULES for m in mods):
            head.append(ast.get_source_segment(tests, node) or "")
    if not head:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.keywords or _callee(node) != name or not node.args:
            continue
        built = 0
        for a in node.args:
            if _ctor(a) is not None:
                built += 1
                continue
            try:
                ast.literal_eval(a)
            except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
                built = -100
                break
        if built > 0:
            return "\n".join(head), [ast.get_source_segment(tests, a) or "" for a in node.args]
    return None


def _expr_battery(args: list[str]) -> list[str]:
    """`_battery` over argument sources: the visible call, one argument varied at a time, then pairs."""
    pools = [_expr_pool(a) for a in args]
    out = [list(args)]
    for i, pool in enumerate(pools):
        out += [args[:i] + [x] + args[i + 1:] for x in pool]
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            for x in pools[i][:16]:
                for y in pools[j][:16]:
                    t = list(args)
                    t[i], t[j] = x, y
                    out.append(t)
    seen, uniq = set(), []
    for t in out:
        key = "(" + ", ".join(t) + ",)"
        if key not in seen:
            seen.add(key)
            uniq.append(key)
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


# ---- class methods (round 19, task 84): scenarios built from the visible tests ----
# Reads {"src", "mod", "head", "scenes", "out"} from stdin. Each scene is the body of one visible test (setUp first)
# with other literal values: the target file is loaded afresh as module `mod`, the test file's own preamble `head`
# runs, then the scene. Its outcome: the values the asserts look at and the calls' return values in order, then the
# state of what the scene left bound (objects of the module by their public attributes); an exception that escapes
# the scene is its outcome "<exc:Type>". One JSON line per scene goes to the file `out`, so a hang keeps the rest.
_SCENE_CHILD = r'''
import contextlib, io, json, sys, types
req = json.loads(sys.stdin.read())
LIM, MOD = %d, req["mod"]


class _Alarm(BaseException):
    pass


try:
    import signal

    def _ring(*a):
        raise _Alarm()
    signal.signal(signal.SIGALRM, _ring)
    have_alarm = True
except Exception:
    have_alarm = False


def _ours(t):
    return getattr(t, "__module__", None) == MOD


def scene(code):
    trace, cut = [], []

    def show(v, d=0):
        t = type(v)
        if d < 3 and _ours(t) and t.__repr__ is object.__repr__ and hasattr(v, "__dict__"):
            return t.__name__ + "{" + ", ".join("%%s=%%s" %% (k, show(x, d + 1)) for k, x in sorted(vars(v).items())
                                                if not k.startswith("_")) + "}"
        if d < 3 and isinstance(v, (list, tuple)) and any(_ours(type(x)) for x in v):
            inner = ", ".join(show(x, d + 1) for x in v)
            return ("[%%s]" if isinstance(v, list) else "(%%s)") %% inner
        try:
            s = repr(v)
        except _Alarm:
            raise
        except BaseException as e:
            s = "<repr:%%s>" %% type(e).__name__
        if len(s) >= LIM:
            cut.append(1)
            s = s[:LIM]
        return s

    def rec(*vals, **kw):
        for v in vals:
            trace.append(show(v))
        return vals[0] if vals else None

    class Raises:
        exception = value = None

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is not None and issubclass(t, _Alarm):
                return False
            trace.append("<raised:%%s>" %% (t.__name__ if t else "none"))
            self.exception = self.value = v
            return True

    def raises(exc=None, fn=None, *a, **k):
        if fn is None:
            return Raises()
        k.pop("msg", None)
        try:
            r = fn(*a, **k)
        except _Alarm:
            raise
        except BaseException as e:
            trace.append("<raised:%%s>" %% type(e).__name__)
            return None
        trace.append("<raised:none>")
        trace.append(show(r))

    class Self:
        def __getattr__(self, n):
            if n in ("assertRaises", "assertWarns"):
                return raises
            if n in ("assertRaisesRegex", "assertWarnsRegex"):
                return lambda exc, rx=None, fn=None, *a, **k: raises(exc, fn, *a, **k)
            if n.startswith("assert") or n == "fail":
                return rec
            if n == "subTest":
                return lambda *a, **k: contextlib.nullcontext()
            raise AttributeError(n)

    class Pytest:
        @staticmethod
        def raises(exc, *a, **k):
            k.pop("match", None)
            return raises(exc, *a, **k) if a else Raises()

        @staticmethod
        def approx(x, *a, **k):
            return x

    mod = types.ModuleType(MOD)
    mod.__file__ = MOD + ".py"
    sys.modules[MOD] = mod
    # a module of a package (`pkg.core`, as the tests import it): its parent packages, empty, lead to it
    parts = MOD.split(".")
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        setattr(sys.modules[name], parts[i], sys.modules[".".join(parts[:i + 1])])
    me = Self()
    try:
        exec(compile(req["src"], MOD + ".py", "exec"), mod.__dict__)
        ns = {"__name__": "__scene__", "_rec": rec, "pytest": Pytest}
        exec(compile(req["head"], "head", "exec"), ns)
        ns["self"] = me
        base = set(ns)
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 1.0)
        exec(compile(code, "scene", "exec"), ns)
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        left = [k for k in sorted(ns) if k not in base and not k.startswith("_")
                and not isinstance(ns[k], (types.ModuleType, type, types.FunctionType))]
        state = ["%%s=%%s" %% (k, show(ns[k])) for k in left]
        state += ["self.%%s=%%s" %% (k, show(v)) for k, v in sorted(vars(me).items()) if not k.startswith("_")]
        # the objects' single-underscore attributes: the rewrites must agree on them too, but a difference from ours
        # is read on the public part only (`_Trial.public`)
        bound = [(k, ns[k]) for k in left] + [("self." + k, v) for k, v in sorted(vars(me).items())
                                               if not k.startswith("_")]
        private = ["%%s.%%s=%%s" %% (k, a, show(x)) for k, v in bound if _ours(type(v)) and hasattr(v, "__dict__")
                   for a, x in sorted(vars(v).items()) if a.startswith("_") and not a.startswith("__")]
        out = json.dumps(trace + state) + ("\x1e" + json.dumps(private) if private else "") + ("<cut>" if cut else "")
    except BaseException as e:
        out = "<exc:%%s>" %% ("TimeoutError" if isinstance(e, _Alarm) else type(e).__name__)
    finally:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return out


real = sys.stdout
with open(req["out"], "w", encoding="utf-8") as f:
    for code in req["scenes"]:
        sys.stdout = io.StringIO()
        try:
            out = scene(code)
        except BaseException as e:
            out = "<exc:%%s>" %% type(e).__name__
        sys.stdout = real
        f.write(json.dumps(out) + "\n")
        f.flush()
''' % _MAX_REPR
SCENE_CAP = 8           # visible tests a method's scenarios come from


def _scene_outcomes(src: str, mod: str, head: str, scenes: list[str], timeout: float) -> list[str]:
    """The outcome of each scene (`_SCENE_CHILD`) with `src` as the module `mod`, all in one subprocess; the scenes a
    hang or a crash left unrun are "<exc:Timeout>" / "<exc:Crash>"."""
    if not scenes:
        return []
    with tempfile.TemporaryDirectory(prefix="scene_") as d:
        out = Path(d) / "out.jsonl"
        req = json.dumps({"src": src, "mod": mod, "head": head, "scenes": scenes, "out": str(out)})
        res = run_python(["-c", _SCENE_CHILD], cwd=Path(d), timeout=timeout, stdin=req, env={"PYTHONHASHSEED": "0"})
        got = []
        try:
            lines = out.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            try:
                got.append(_ADDR.sub("", str(json.loads(line))))
            except ValueError:
                break
    fill = "<exc:Timeout>" if res.timed_out else "<exc:Crash>"
    return (got + [fill] * len(scenes))[:len(scenes)]


class _Scene(ast.NodeTransformer):
    """A test body as a scene: `assert a == b` records both sides, a bare call records what it returns (the
    `self.assert*` calls record their arguments in the child)."""

    def visit_Assert(self, node):
        test = node.test
        vals = [test.left, *test.comparators] if isinstance(test, ast.Compare) else [test]
        return ast.copy_location(ast.Expr(ast.Call(ast.Name("_rec", ast.Load()), vals, [])), node)

    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Call) and not (isinstance(v.func, ast.Attribute) and isinstance(v.func.value, ast.Name)
                                            and v.func.value.id == "self"):
            node.value = ast.Call(ast.Name("_rec", ast.Load()), [self.generic_visit(v)], [])
            return node
        return self.generic_visit(node)


def _literal(node) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        return False


def _slots(body: list) -> list:
    """The literal values of a scene that are inputs: constants and literal lists, not the expected values (direct
    literal arguments of an assert, of `_rec` from an `assert`, of `approx`), not f-string parts or None."""
    out, skip = [], set()
    for node in (n for stmt in body for n in ast.walk(stmt)):
        if isinstance(node, ast.Call):
            name = _callee(node)
            if name == "_rec" or name == "approx" or (name or "").startswith("assert") and name not in _RAISES:
                skip.update(id(a) for a in node.args if _literal(a))
        elif isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
    seen = set()

    def walk(node):
        if id(node) in skip or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, (ast.List, ast.Tuple)) and isinstance(getattr(node, "ctx", None), ast.Load) \
                and node.elts and _literal(node):
            out.append(node)
            return
        if isinstance(node, ast.Constant) and node.value is not None and not isinstance(node.value, bytes) \
                and not isinstance(node.value, complex) and not (isinstance(node.value, str) and len(node.value) > 40):
            out.append(node)
            return
        for child in ast.iter_child_nodes(node):
            walk(child)
    for stmt in body:
        walk(stmt)
    return out


class _Put(ast.NodeTransformer):
    def __init__(self, subs: dict):
        self.subs = subs

    def visit(self, node):
        k = getattr(node, "_slot", None)
        if k is not None and k in self.subs:
            return ast.copy_location(ast.parse(repr(self.subs[k]), mode="eval").body, node)
        return super().visit(node)


def _refers(body: list, cls: str, meth: str) -> bool:
    nodes = [n for stmt in body for n in ast.walk(stmt)]
    names = {getattr(n, "id", None) for n in nodes if isinstance(n, ast.Name)} | \
            {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    if cls not in names:
        return False
    if meth == "__init__":
        return any(isinstance(n, ast.Call) and _callee(n) == cls for n in nodes)
    return meth in {n.attr for n in nodes if isinstance(n, ast.Attribute)}


def _scenes(tests: str, name: str) -> tuple[str, list[str]]:
    """(preamble, scenes) for the method `name` ("Class.method"): the visible tests that build the class and call
    the method (setUp first), as they are and with other values of the same types in their literal inputs (one at a
    time, then two together), at most BATTERY_CAP scenes. A test that takes arguments (fixtures, parametrize) is
    left out; a dunder other than __init__ has no scene."""
    cls, _, meth = name.partition(".")
    if not cls or not meth or "." in meth or (meth.startswith("__") and meth != "__init__"):
        return "", []
    try:
        tree = ast.parse(tests)
    except (SyntaxError, ValueError, RecursionError):
        return "", []

    def is_test(f):
        return isinstance(f, ast.FunctionDef) and f.name.startswith("test")

    head, bodies = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(is_test(f) for f in node.body):
            setup = next((f.body for f in node.body if isinstance(f, ast.FunctionDef) and f.name == "setUp"), [])
            for f in node.body:
                a = f.args if is_test(f) else None
                if a is not None and len(a.args) == 1 and not (a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs):
                    bodies.append(setup + f.body)
        elif is_test(node):
            a = node.args
            if not (a.args or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs):
                bodies.append(node.body)
        elif isinstance(node, ast.Expr) or isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            continue    # `unittest.main()`, a docstring, the main guard: not the tests' setting
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0])
                == "pytest" for a in node.names):
            continue
        elif not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list):
            head.append(node)
    per = []
    for body in [b for b in bodies if _refers(b, cls, meth)][:SCENE_CAP]:
        mod = ast.Module(body=[_Scene().visit(copy.deepcopy(s)) for s in body], type_ignores=[])
        ast.fix_missing_locations(mod)
        slots = _slots(mod.body)
        for k, node in enumerate(slots):
            node._slot = k
        values = [ast.literal_eval(n) for n in slots]
        pools = [_pool(v) for v in values]
        subs = itertools.chain(
            [{}], ({i: x} for i, pool in enumerate(pools) for x in pool if x != values[i]),
            ({i: x, j: y} for i in range(len(pools)) for j in range(i + 1, len(pools))
             for x in pools[i][:12:2] for y in pools[j][:12] if x != values[i] and y != values[j]))
        texts = []
        for sub in subs:
            if len(texts) >= BATTERY_CAP:
                break
            try:
                texts.append(ast.unparse(_Put(sub).visit(copy.deepcopy(mod))))
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
        per.append(texts)
    scenes, seen = [], set()
    for row in itertools.zip_longest(*per):      # every test gets its share of the cap
        for t in row:
            if t is not None and t not in seen and len(scenes) < BATTERY_CAP:
                seen.add(t)
                scenes.append(t)
    if not scenes:
        return "", []
    return ast.unparse(ast.Module(body=head, type_ignores=[])), scenes


def _module_name(target: str) -> str:
    """The name the tests import the target by: `mylib` for mylib.py, `pkg.core` for pkg/core.py, `pkg` for
    pkg/__init__.py, `wallet.ledger` for src/wallet/ledger.py (HARNESS_DEEP_TESTS layouts)."""
    parts = list(Path(target).with_suffix("").parts)
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    if len(parts) > 1 and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _loads_alone(ws: Path, target: str, src: str) -> bool:
    """The target file loads on its own, as the probes and scenes run it (a throw-away directory, nothing of the
    workspace beside it): no relative import, no import of another module or package of the workspace. A file in the
    root always counts (round 18: its review is never skipped)."""
    if "/" not in target:
        return True
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            return False
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for m in names:
            top = m.split(".")[0]
            if any((base / top).is_dir() or (base / f"{top}.py").is_file() for base in (ws, ws / "src")):
                return False
    return True


class _Trial:
    """Trial inputs of one function of the file and each version's outcomes on them, one subprocess a version,
    remembered by the file's text (the third opinion probes the same versions again). A method (with
    HARNESS_REVIEW_METHODS) has scenes for inputs (`_scenes`). `ok`: it can be probed (at least 3 inputs)."""

    def __init__(self, src: str, tests: str, name: str, target: str = "mylib.py"):
        self.src, self.name, self.memo = src, name, {}
        self.inputs: list = []
        self.method, self.mod, self.head = "." in name, _module_name(target), ""
        self.expr_head: str | None = None   # HARNESS_REVIEW_PROBES: the inputs are source text (`_visible_exprs`)
        # round 19: ours is probed twice and an input where it differs from itself is set aside (`steady`). Every
        # round-19 switch off: one run of ours, as in round 18
        self.twice = config.review_reason() or config.review_methods()
        if self.method:
            if config.review_methods():
                self.head, self.inputs = _scenes(tests, name)
            return
        call = visible_call(tests, name)
        try:
            self.inputs = _battery(call[0] if call else None, _arity(src, name))
            if config.review_probes() and call:
                # a parameter with a default: calls that leave it out, one after another in the same process, so a
                # mutable default that keeps what earlier calls put in it shows
                need, args = _arity(src, name), call[0]
                if 1 <= need < len(args):
                    short = [tuple(args[:need])] + [(x,) + tuple(args[1:need]) for x in _pool(args[0])[:6]]
                    self.inputs = self.inputs[:BATTERY_CAP - len(short)] + short
            elif config.review_probes():
                found = _visible_exprs(tests, name)
                if found is not None:
                    self.expr_head, self.inputs = found[0], _expr_battery(found[1])
        except (SyntaxError, ValueError, RecursionError, TypeError, MemoryError):
            self.inputs = []

    @property
    def ok(self) -> bool:
        return len(self.inputs) >= 3

    def cut(self, out: str) -> bool:
        """A value cut at the probe's repr limit: it shows neither agreement nor a difference."""
        return out.endswith("<cut>") if self.method else len(out) >= _MAX_REPR

    def public(self, out: str) -> str:
        """What a difference from ours is read on: a scene's outcome without its private state (`_SCENE_CHILD`
        puts the single-underscore attributes after a \\x1e); a function's outcome as it is."""
        return out.split("\x1e", 1)[0] if self.method else out

    def outcomes(self, text: str | None, stop: float | None, role: str = "ours") -> list[str] | None:
        """The outcomes of the file with `text` as the function (None: the file's own); None when out of time.
        A run is remembered by `role` and text: two roles with the same text are two runs, so a rewrite whose output
        is not the same on every run does not agree with itself."""
        full = self.src if text is None else _spliced(self.src, self.name, text)
        if full is None:
            return None
        key = (role, full)
        if key not in self.memo:
            limit = _left(stop, PROBE_TIMEOUT)
            if limit < PROBE_TIMEOUT / 2:
                return None
            self.memo[key] = (_scene_outcomes(full, self.mod, self.head, self.inputs, limit) if self.method
                              else _expr_behaviour(full, self.name, self.expr_head, self.inputs, limit)
                              if self.expr_head is not None
                              else behaviour(full, self.name, self.inputs, timeout=limit))
        return self.memo[key]

    def hangs(self, idx: list[int], stop: float | None) -> list[int]:
        """HARNESS_REVIEW_PROBES: the inputs among `idx` (where ours ran into the one-second alarm) on which ours,
        run again alone in the review's own child, still runs into its alarm after HANG_CONFIRM_S seconds: a slow
        but correct function, or a busy machine, returns there and is no evidence. At most HANG_CONFIRM_MAX inputs;
        none when the time left cannot hold the run."""
        idx = idx[:HANG_CONFIRM_MAX]
        if not idx or self.method:
            return []
        texts = [self.inputs[i] if self.expr_head is not None else repr(tuple(self.inputs[i])) for i in idx]
        key = ("ours hang", tuple(texts))
        if key not in self.memo:
            need = HANG_CONFIRM_S * len(texts) + 5
            limit = _left(stop, need)
            if limit < need:
                return []
            self.memo[key] = _expr_behaviour(self.src, self.name, self.expr_head or "", texts, limit,
                                             alarm=HANG_CONFIRM_S)
        return [i for i, o in zip(idx, self.memo[key]) if o == "<exc:ProbeAlarm>"]

    def steady(self, stop: float | None) -> list[bool] | None:
        """Per input: ours gives the same outcome on a second run. An input where it does not (random, time) shows
        neither agreement nor a difference. All True with every round-19 switch off (round 18: one run)."""
        first = self.outcomes(None, stop)
        if first is None or not self.twice:
            return None if first is None else [True] * len(first)
        again = self.outcomes(None, stop, "ours again")
        return None if again is None else [x == y for x, y in zip(first, again)]


def _decorators(text: str) -> list[str] | None:
    """The decorator list of the function in `text`, as AST dumps; None when it does not parse."""
    try:
        node = ast.parse(textwrap.dedent(text)).body[0]
    except (SyntaxError, ValueError, RecursionError, IndexError):
        return None
    return [ast.dump(d) for d in getattr(node, "decorator_list", [])]


def _verdict(src: str, tests: str, name: str, a: str, b: str, stop: float | None = None,
             trial: "_Trial | None" = None, roles: tuple = ("first", "second")) -> tuple[list[str], bool]:
    """(`_evidence`, disagree): disagree is True when the two rewrites are two opinions rather than one: one of them
    shows itself broken and the other does not, they return different values where the file's function returns one,
    or one raises where the other returns a value. Only then can a third opinion side with one of them. Where ours
    differs from itself on some input (`_Trial.steady`), nothing is judged. `roles`: under which names the runs of
    `a` and `b` are remembered (`_Trial.outcomes`). A method rewrite that changes the method's decorators (drops
    `@property`) is no evidence."""
    trial = trial if trial is not None else _Trial(src, tests, name)
    if not trial.ok or _spliced(src, name, a) is None or _spliced(src, name, b) is None:
        return [], False
    if trial.method and not (_decorators(_texts(src).get(name, "")) == _decorators(a) == _decorators(b)):
        return [], False
    steady = trial.steady(stop)
    if steady is None:
        return [], False
    outs = []
    for text, role in ((a, roles[0]), (b, roles[1])):
        o = trial.outcomes(text, stop, role)
        if o is None:
            return [], False
        outs.append(o)
    if not all(steady):
        # ours is not the same on every run (random, time): an input where it happened to repeat itself can still
        # hide chance, the rewrites' agreement there as well as their difference from ours (probe_memo: a wrong
        # `random.choice(xs[:-1] or xs)` "differs" on [1, 2] when ours drew 2 twice). Not judged, no third opinion
        return [], False
    return _judge(src, tests, name, a, b, trial, trial.outcomes(None, stop), *outs, stop=stop)


def _judge(src: str, tests: str, name: str, a: str, b: str, trial: "_Trial", ours: list, ba: list,
           bb: list, stop: float | None = None) -> tuple[list[str], bool]:
    """`_verdict`'s rules on the outcomes of ours and of the two rewrites."""
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
    # HARNESS_REVIEW_PROBES: an input on which the file's function ran into the one-second alarm while both rewrites
    # return the same value shows it wrong as well, once a second run of ours alone confirms the hang with the
    # review's own alarm (`_Trial.hangs`): a TimeoutError it raises itself is a raise like any other (round 18, a
    # change only where ours raises is no evidence), and a slow but correct function returns given more time
    hung = [i for i, o in enumerate(ours) if o in ("<exc:TimeoutError>", "<exc:ProbeAlarm>") and _exc(ba[i]) is None
            and ba[i] == bb[i] and not trial.cut(ba[i])] if config.review_probes() and not trial.method else []
    # HARNESS_REVIEW_PROBES: where the file's function raises a value-domain exception its own text does not name and
    # both rewrites raise another one that the text does name ("Raises ValueError when the second number is zero",
    # the file divides and lets ZeroDivisionError out), the file is shown wrong; the stored text must say `raise <Name>`
    seg = _texts(src).get(name, "")
    swap = {_exc(ba[i]) for i in range(len(ours))
            if _exc(ours[i]) in _VALUE_EXC and _exc(ba[i]) in _VALUE_EXC and ba[i] == bb[i] and ba[i] != ours[i]
            and _exc(ba[i]) in seg and _exc(ours[i]) not in seg} if config.review_probes() and not trial.method else set()
    if not swap and not any(ba[i] == bb[i] and trial.public(ba[i]) != trial.public(ours[i])
                            and (_exc(ba[i]) is None or _exc(ba[i]) in _VALUE_EXC) for i in both):
        if not (hung and trial.hangs(hung, stop)):   # the confirming run only when nothing else shows ours wrong
            return [], False

    def says(t):
        return all(re.search(r"\braise\s+" + e + r"\b", t) for e in swap)

    ok = [t for t, k in ((a, 0), (b, 1)) if not any(raises[i][k] for i in valid) and says(t)]
    if ok:
        return ok, False
    # both rewrites raise where ours returns a value: stored only when they raise the same value-domain exception
    # on every such input, each raises it explicitly, and the function's text or its visible tests name it
    spots = [i for i in valid if raises[i][0] or raises[i][1]]
    excs = {_exc(ba[i]) for i in spots}
    if not all(ba[i] == bb[i] and _exc(ba[i]) in _VALUE_EXC for i in spots):
        return [], False
    named = _naming(tests, name)
    if not all(e in seg or any(e in ln for ln in named) for e in excs):
        return [], False
    return [t for t in (a, b) if all(re.search(r"\braise\s+" + e + r"\b", t) for e in excs) and says(t)], False


_RAISES = frozenset({"raises", "assertRaises", "assertRaisesRegex"})


def _callee(call: ast.Call) -> str | None:
    """The called name: `f(...)` -> f, `x.f(...)` -> f."""
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


def _naming(tests: str, name: str) -> list[str]:
    """The test text that may name an exception for the function `name` (a method by its own name): the lines that
    hold the name as a whole word (`mean` is not `mean_abs`), and the head of every `with pytest.raises(X):` /
    `with self.assertRaises(X):` block whose last statement is the call of the function (no other call in its
    arguments), though the call stands on another line.
    HARNESS_REVIEW_NAMED off: round 18, the lines that hold the name anywhere."""
    short = name.rsplit(".", 1)[-1]
    if not config.review_named():
        return [ln for ln in tests.splitlines() if short in ln]
    word = re.compile(r"(?<!\w)" + re.escape(short) + r"(?!\w)")
    out = [ln for ln in tests.splitlines() if word.search(ln)]
    try:
        tree = ast.parse(tests)
    except (SyntaxError, ValueError, RecursionError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee(node) in _RAISES and any(
                getattr(a, "attr", None) == short or getattr(a, "id", None) == short for a in node.args[1:]):
            out.append(ast.get_source_segment(tests, node) or "")   # assertRaises(X, f, ...) over several lines
            continue
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        heads = [it.context_expr for it in node.items
                 if isinstance(it.context_expr, ast.Call) and _callee(it.context_expr) in _RAISES]
        if not heads:
            continue
        # the block names the exception for `short` only when the call of `short` is what raises: the outermost call
        # of the block's last statement, with no other call in its arguments (`mean(parse("1,x"))` raises in parse)
        call = getattr(node.body[-1], "value", None) if node.body else None
        if isinstance(call, ast.Call) and _callee(call) == short and not any(
                isinstance(c, ast.Call) for a in [*call.args, *(k.value for k in call.keywords)] for c in ast.walk(a)):
            out += [ast.get_source_segment(tests, h) or "" for h in heads]
    return out


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
    exception on every such input, the function's text or the visible tests name that exception for it
    ("ValueError if the lengths differ", `_naming`), and the stored text says `raise <Name>`. A method cannot be
    probed: kept.
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
        trial = trials.setdefault(name, _Trial(src, tests, name, target))
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
            names: list, trials: dict, stop: float | None = None) -> tuple[str, list[str], dict]:
    """The disputed or one-sided functions (`names`) where the third rewrite and FAST's, else ALT's (whichever
    rewrote it), pass `_verdict` as one opinion: that pair's text goes into `cur` when the visible tests stay green.
    Also name -> the side taken ("FAST" / "ALT"), for the note."""
    changed, sides = [], {}
    for name in names:
        if name not in third:
            continue
        for other, role, side in ((first.get(name), "first", "FAST"), (second.get(name), "second", "ALT")):
            if other is None:
                continue
            if stop is not None and time.monotonic() > stop:
                return cur, changed, sides
            texts, _ = _verdict(src, tests, name, third[name], other, stop, trials.get(name), ("third", role))
            for text in texts:
                cand = _spliced(cur, name, text)
                if cand is not None and _green(ws, cand, target, stop):
                    cur = cand
                    changed.append(name)
                    sides[name] = side
                    break
            if name in changed:
                break
    return cur, changed, sides


def _lone(ws: Path, src: str, tests: str, target: str, first: dict, second: dict, known: dict | None,
          stop: float | None, trials: dict) -> list[str]:
    """The one-sided functions (round 19, task 87): rewritten by only one of FAST and ALT, not known texts, whose
    rewrite would pass every rule if a second opinion agreed with it (`_verdict` of the rewrite against a second run
    of itself: a difference from ours where ours returns a value, nothing broken, the raise rule) and keeps the
    visible tests green. A guard-only or a red rewrite is no case: asking MODEL_REASON about it could store nothing."""
    out = []
    for name in sorted(set(first) ^ set(second)):
        if stop is not None and time.monotonic() > stop:
            break
        if _known(src, name, known):
            continue
        text, role = (first[name], "first") if name in first else (second[name], "second")
        trial = trials.setdefault(name, _Trial(src, tests, name, target))
        texts, _ = _verdict(src, tests, name, text, text, stop, trial, (role, role + " again"))
        cand = _spliced(src, name, texts[0]) if texts else None
        if cand is not None and _green(ws, cand, target, stop):
            out.append(name)
    return out


def _may_reason(client, msgs: list[dict], lone_only: bool = False) -> bool:
    """The third opinion fits the stage's gates: under REASON_SPEND_LIMIT of the budget and affordable (a client
    without a ledger, as in the tests, passes this part), and under the run's cap: REASON_MAX_CALLS calls and
    REASON_MAX_UNITS units, tallied on the client (one client a run). A True reserves one call. A file with only
    one-sided cases (`lone_only`) leaves the last REASON_DISPUTE_CALLS calls and REASON_DISPUTE_UNITS units to
    disputes: when the cap binds, a dispute goes first."""
    calls, units = getattr(client, "_review_reason", (0, 0.0))
    keep_calls, keep_units = (REASON_DISPUTE_CALLS, REASON_DISPUTE_UNITS) if lone_only else (0, 0)
    if calls >= REASON_MAX_CALLS - keep_calls or units >= REASON_MAX_UNITS - keep_units:
        return False
    ledger = getattr(client, "ledger", None)
    if ledger is not None:
        est = sum(len(m["content"]) for m in msgs) // 40 + REASON_TOKENS
        if ledger.exhausted or ledger.spent >= REASON_SPEND_LIMIT * ledger.budget or not ledger.can_afford(est):
            return False
    client._review_reason = (calls + 1, units)
    return True


def _focus(msgs: list[dict], names: list) -> list[dict]:
    """The review prompt with the question narrowed to the disputed functions (no other reply shown). Asked about
    the whole file, MODEL_REASON weighs every function and can spend its whole output limit reasoning (far8
    coding_8004: 3000 tokens, no reply); asked about one, 150-1000 tokens."""
    ask = f"\n\nLook only at {', '.join(names)}: rewrite it if it is wrong, else reply NONE."
    return [*msgs[:-1], {**msgs[-1], "content": msgs[-1]["content"] + ask}]


async def _third(client, msgs: list[dict], src: str) -> tuple[dict, str]:
    """MODEL_REASON's rewrites on `msgs` (`_focus`), and a word for the note."""
    try:
        r = await client.chat(MODEL_REASON, msgs, REASON_TOKENS, reasoning_effort="low")
    except LLMUnavailable:
        return {}, "unavailable"
    calls, units = getattr(client, "_review_reason", (1, 0.0))
    client._review_reason = (calls, units + (r.cost or 0))
    if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= REASON_TOKENS:
        return {}, "hit the output limit"
    third = _rewrites(src, r.text)
    return third, str(sorted(third))


async def review(task, client, known: dict | None = None, stop: float | None = None) -> tuple[list[str], str]:
    """Names of the functions replaced in the task's file, and a note for the trajectory. `known`: `known_texts`
    (default the examples' texts only). `stop`: the monotonic time after which no probe or test run starts."""
    ws = Path(task.workspace)
    try:
        test_name, tests = _read_tests(ws)
        target = _target_file(ws, tests)
        path = ws / target
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [], f"review skipped: {e!r}"[:200]
    if not await asyncio.to_thread(_green, ws, src, target, stop):
        return [], "review skipped: the tests are not green"
    if not _loads_alone(ws, target, src):   # no probe or scene could load it: every rewrite would be "no evidence"
        return [], "review skipped: the package module imports what a probe cannot load"
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
        if config.review_methods():
            blind = sorted(n for n in first if "." in n and not _Trial(src, tests, n, target).ok)
            first = {n: t for n, t in first.items() if n not in blind}
            if not first:
                return [], f"review: no visible test calls {blind}, kept"[:200]
        r = await client.chat(MODEL_ALT, msgs, REVIEW_MAX_TOKENS)
        if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= REVIEW_MAX_TOKENS:
            return [], "review: second reply hit the output limit, kept"
        second = _rewrites(src, r.text)
    except LLMUnavailable as e:
        return [], f"review: llm unavailable: {e}"[:200]
    both = set(first) & set(second)
    if not both and not config.review_reason():
        return [], f"review: no agreement ({sorted(first)} / {sorted(second)}), kept"[:200]
    disputed, trials = [], {}
    new, changed = src, []
    if both:
        new, changed = await asyncio.to_thread(_decide, ws, src, tests, target, first, second, known, stop,
                                               disputed, trials)
    # task 87: a function only one of them rewrote is asked about too, after the disputes
    lone = await asyncio.to_thread(_lone, ws, src, tests, target, first, second, known, stop, trials) \
        if config.review_reason() else []
    if changed and new != src:
        # the agreed fixes are written before the third opinion is asked: a failure or a cancel there keeps them
        if path.read_text(encoding="utf-8") != src:   # someone else rewrote the file meanwhile: leave it
            return [], "review: the file changed meanwhile, kept"
        write_mylib(path, new)
        shutil.rmtree(ws / "__pycache__", ignore_errors=True)
    else:
        changed, new = [], src
    tail = ""
    asks = sorted(disputed) + lone          # the disputes first
    if asks and config.review_reason():
        what = "; " + ", ".join(([f"{sorted(disputed)} disputed"] if disputed else [])
                                + ([f"{lone} one-sided"] if lone else []))
        if stop is not None and stop - time.monotonic() < MIN_LEFT_S:
            tail = f"{what}, too close to the stop for {MODEL_REASON}"
        elif not _may_reason(client, _focus(msgs, asks), lone_only=not disputed):
            tail = f"{what}, no room for {MODEL_REASON}"
        else:
            third, word = await _third(client, _focus(msgs, asks), src)
            cur, more, sides = await asyncio.to_thread(_settle, ws, src, new, tests, target, first, second, third,
                                                       asks, trials, stop)
            side = (f" with {sides[more[0]]}" if len(more) == 1 else f" with {sides}") if more else ""
            tail = f"{what}, {MODEL_REASON} {word}, sided {sorted(more)}{side}"
            if more:
                if path.read_text(encoding="utf-8") != new:
                    tail += ", the file changed meanwhile"
                else:
                    write_mylib(path, cur)
                    shutil.rmtree(ws / "__pycache__", ignore_errors=True)
                    changed += more
    if not changed:
        if not both:
            return [], f"review: no agreement ({sorted(first)} / {sorted(second)}){tail}, kept"[:300]
        return [], f"review: no evidence against {sorted(both)}{tail}, kept"[:300]
    return changed, f"review: replaced {changed} ({sorted(first)} / {sorted(second)}){tail}"[:300]

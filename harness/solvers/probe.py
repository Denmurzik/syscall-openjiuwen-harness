"""Behavioural probing of a mylib function: run it on trial inputs in one subprocess."""
import ast
import copy
import json
import random
import re
import tempfile
from pathlib import Path

from harness.proc import run_python

_MARK = "@@PROBE@@"
_MAX_REPR = 200
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")

# Reads {"src", "func", "inputs"} (inputs as repr text) from stdin; prints one marked JSON
# line per input, so partial results survive a hang or a hard kill.
_CHILD = r'''
import ast, copy, io, json, sys
req = json.loads(sys.stdin.read())
real = sys.stdout
mark = %r
def emit(s):
    real.write(mark + json.dumps(s) + "\n"); real.flush()
try:
    import signal
    def _alarm(*a):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    have_alarm = True
except Exception:
    have_alarm = False
sys.stdout = io.StringIO()
ns = {"__name__": "mylib"}
try:
    exec(compile(req["src"], "mylib.py", "exec"), ns)
    fn = ns[req["func"]]
except BaseException as e:
    for _ in ast.literal_eval(req["inputs"]):
        emit("<exc:%%s>" %% type(e).__name__)
    sys.exit(0)
for args in ast.literal_eval(req["inputs"]):
    try:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 1.0)
        out = repr(fn(*copy.deepcopy(args)))[:%d]
    except BaseException as e:
        out = "<exc:%%s>" %% type(e).__name__
    finally:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    sys.stdout = io.StringIO()
    emit(out)
''' % (_MARK, _MAX_REPR)


def behaviour(mylib_src: str, func: str, inputs: list[tuple], timeout: float = 5) -> list[str]:
    """repr of func(*args) (or "<exc:Type>") for each input, all in one subprocess."""
    if not inputs:
        return []
    req = json.dumps({"src": mylib_src, "func": func, "inputs": repr(list(inputs))})
    with tempfile.TemporaryDirectory(prefix="probe_") as d:
        res = run_python(["-c", _CHILD], cwd=Path(d), timeout=timeout, stdin=req,
                         env={"PYTHONHASHSEED": "0"})
    out: list[str] = []
    for line in res.stdout.splitlines():
        if line.startswith(_MARK):
            try:
                out.append(_ADDR.sub("", str(json.loads(line[len(_MARK):]))))
            except ValueError:
                break
    fill = "<exc:Timeout>" if res.timed_out else "<exc:Crash>"
    return (out + [fill] * len(inputs))[:len(inputs)]


def _vary(v, rng: random.Random):
    """A few same-shaped variants of one argument."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [0, 1, -1, v + 1, v - 1, v * 2, rng.randint(-10, 20)]
    if isinstance(v, float):
        return [0.0, v + 1, -v, v * 2.5]
    if isinstance(v, list):
        if v and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return [[0], [2, 2], [-1, 5, 3, 3], [100, -100, 7, 8, 9], list(reversed(v)),
                    [rng.randint(-9, 9) for _ in range(rng.randint(2, 6))],
                    [rng.randint(0, 9) for _ in range(rng.randint(3, 7))], v + v[:1]]
        return [list(reversed(v)), v + v[:1], v[:1]]
    if isinstance(v, str):
        return ["a b", "Abba", "  hi  there ", "Zz yY", "q", v.upper(), v[::-1], v + " " + v]
    if isinstance(v, tuple):
        return [tuple(reversed(v))]
    return []


def inputs_from_call(args: tuple, n: int = 6, seed: int = 0) -> list[tuple]:
    """The original argument tuple first, then up to n-1 distinct same-shaped variants."""
    rng = random.Random(seed)
    out: list[tuple] = [tuple(args)]
    seen = {repr(out[0])}
    pool: list[tuple] = []
    for i, a in enumerate(args):
        for v in _vary(a, rng):
            pool.append(tuple(args[:i]) + (v,) + tuple(args[i + 1:]))
    # joint variants help when the function relates its arguments
    if len(args) > 1:
        alts = [_vary(a, rng) for a in args]
        for _ in range(n):
            pool.append(tuple((rng.choice(alt) if alt else a) for a, alt in zip(args, alts)))
    rng.shuffle(pool)
    for p in pool:
        if len(out) >= n:
            break
        key = repr(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def visible_call(test_src: str, func: str) -> tuple[tuple, str] | None:
    """First call of mylib's `func` with literal arguments in the tests: (args, statement source)."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None
    imported: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name == "mylib"}
        elif isinstance(node, ast.ImportFrom) and node.module == "mylib":
            imported |= {a.asname or a.name for a in node.names if a.name == func}
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.stmt) or isinstance(stmt, (ast.ClassDef, ast.FunctionDef,
                                                               ast.AsyncFunctionDef)):
            continue
        for c in ast.walk(stmt):
            f = getattr(c, "func", None)
            if not isinstance(c, ast.Call) or f is None or c.keywords:
                continue
            hit = ((isinstance(f, ast.Attribute) and f.attr == func
                    and isinstance(f.value, ast.Name) and f.value.id in modules)
                   or (isinstance(f, ast.Name) and f.id in imported))
            if not hit:
                continue
            try:
                args = tuple(ast.literal_eval(a) for a in c.args)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            seg = ast.get_source_segment(test_src, stmt) or ""
            return copy.deepcopy(args), seg.strip()[:240]
    return None

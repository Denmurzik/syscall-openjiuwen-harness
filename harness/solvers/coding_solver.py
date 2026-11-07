"""Coding tasks: repair single-token bugs in workspace/mylib.py by inverse mutation.

0 LLM tokens. The failing tests tell which mylib functions to mutate; every candidate
is evaluated in a temporary copy of the workspace, never in the real one.
"""
import ast
import os
import shutil
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from harness import config
from harness.result import SolveResult
from harness.solvers.mutations import Mutant, tagged_mutants
from harness.solvers.testrun import TestReport, run_in_dir
from harness.tasks import Task

CANDIDATE_CAP = 3000
CAND_TIMEOUT_MAX = 5.0
BASE_TIMEOUT = 20.0


class _OutOfTime(Exception):
    pass


@dataclass(frozen=True)
class _Unit:
    qualname: str
    owner: str          # name a test would use: top-level function or class name
    start: int          # 0-based first line (decorators included)
    end: int            # exclusive
    col: int


def _units(src: str) -> list[_Unit]:
    tree = ast.parse(src)
    out: list[_Unit] = []

    def add(node, qual, owner):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        out.append(_Unit(qual, owner, first - 1, node.end_lineno, node.col_offset))

    func = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, func):
            add(node, node.name, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, func):
                    add(sub, f"{node.name}.{sub.name}", node.name)
    return out


def _segment(src: str, u: _Unit) -> str:
    return "".join(src.splitlines(keepends=True)[u.start:u.end])


def _splice(src: str, u: _Unit, new_func: str) -> str:
    lines = src.splitlines(keepends=True)
    body = textwrap.indent(new_func.rstrip("\n") + "\n", " " * u.col)
    return "".join(lines[:u.start]) + body + "".join(lines[u.end:])


def _test_refs(test_src: str) -> dict[str, set[str]]:
    """test method name -> mylib names it references (empty set = unknown)."""
    tree = ast.parse(test_src)
    modules: set[str] = set()
    imported: dict[str, str] = {}
    star = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "mylib":
                    modules.add(a.asname or "mylib")
        elif isinstance(node, ast.ImportFrom) and node.module == "mylib":
            for a in node.names:
                if a.name == "*":
                    star = True
                else:
                    imported[a.asname or a.name] = a.name
    refs: dict[str, set[str]] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names: set[str] = set()
            for n in ast.walk(fn):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                        and n.value.id in modules:
                    names.add(n.attr)
                elif isinstance(n, ast.Name) and n.id in imported:
                    names.add(imported[n.id])
            refs.setdefault(fn.name, set()).update(set() if star else names)
    return refs


class _Run:
    def __init__(self, sandbox: Path, test_file: str, deadline: float):
        self.sandbox = sandbox
        self.test_file = test_file
        self.deadline = deadline
        self.cand_timeout = CAND_TIMEOUT_MAX
        self.verify_timeout = BASE_TIMEOUT
        self.count = 0

    def _remaining(self) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0.2:
            raise _OutOfTime()
        return left

    def run(self, src: str, only: list[str] | None, timeout: float) -> TestReport:
        timeout = min(timeout, self._remaining())
        (self.sandbox / "mylib.py").write_text(src, encoding="utf-8", newline="\n")
        return run_in_dir(self.sandbox, self.test_file, timeout, only)

    def candidate(self, src: str, only: list[str]) -> TestReport:
        self.count += 1
        return self.run(src, only, self.cand_timeout)

    def verify(self, src: str, failing: list[str]) -> TestReport | None:
        """Full run; accept only when failures shrink to a proper subset of `failing`."""
        rep = self.run(src, None, self.verify_timeout)
        if rep.ok:
            return rep
        if any(f.startswith("<") for f in rep.failed):
            return None
        return rep if set(rep.failed) < set(failing) else None


def _single(src: str, units: list[_Unit]) -> Iterator[tuple[str, Mutant]]:
    for u in units:
        for m in tagged_mutants(_segment(src, u)):
            yield _splice(src, u, m.source), m


def _pairs(src: str, units: list[_Unit]) -> Iterator[tuple[str, Mutant]]:
    for u in units:
        seg = _segment(src, u)
        original = ast.unparse(ast.parse(textwrap.dedent(seg)))
        for m1 in tagged_mutants(seg):
            for m2 in tagged_mutants(m1.source):
                if m2.source == original:
                    continue
                yield _splice(src, u, m2.source), Mutant(
                    m2.source, "pair", m1.inverse and m2.inverse, max(m1.prio, m2.prio))


def _search(ctx: _Run, gen: Iterator[tuple[str, Mutant]], failing: list[str],
            only: list[str], capped: bool) -> tuple[str, TestReport] | None:
    """Evaluate candidates. Ranking: green+inverse (returned at once) > green > partial
    inverse > partial (a partial fix leaves other tests red)."""
    slots: dict[tuple[bool, bool], tuple[str, TestReport]] = {}
    for cand, mut in gen:
        if capped and ctx.count >= CANDIDATE_CAP:
            break
        if not ctx.candidate(cand, only).ok:
            continue
        full = ctx.verify(cand, failing)
        if full is None:
            continue
        if full.ok and mut.inverse:
            return cand, full
        slots.setdefault((full.ok, mut.inverse), (cand, full))
    for key in ((True, False), (False, True), (False, False)):
        if key in slots:
            return slots[key]
    return None


def _callees(src: str, units: list[_Unit], owner: str) -> list[str]:
    """Owners (mylib functions/classes) referenced by name inside `owner`'s code."""
    owners = {u.owner for u in units}
    names: list[str] = []
    for u in units:
        if u.owner != owner:
            continue
        for n in ast.walk(ast.parse(textwrap.dedent(_segment(src, u)))):
            ident = n.id if isinstance(n, ast.Name) else None
            if ident in owners and ident != owner and ident not in names:
                names.append(ident)
    return names


def _repair_round(ctx: _Run, cur: str, report: TestReport, refs: dict[str, set[str]]
                  ) -> tuple[str, TestReport] | None:
    failing = report.failed
    units = _units(cur)
    all_owners = [u.owner for u in units]
    if any(f not in refs for f in failing):
        return None
    scope: dict[str, list[str]] = {}   # owner -> failing tests to satisfy
    for f in failing:
        for name in sorted(refs[f]):
            if name in all_owners:
                scope.setdefault(name, [])
                if f not in scope[name]:
                    scope[name].append(f)
    called = list(scope)
    # Helpers called by those functions are tried too (breadth first), against the same tests.
    queue = list(called)
    while queue:
        name = queue.pop(0)
        for callee in _callees(cur, units, name):
            if callee not in scope:
                scope[callee] = list(scope[name])
                queue.append(callee)
    ctx.count = 0

    # Phase A: functions called by failing tests, then their helpers, each against its own
    # failing tests. A green candidate wins at once; a partial fix is committed only after
    # every helper has been tried.
    partial = None
    for name, mine in scope.items():
        found = _search(ctx, _single(cur, [u for u in units if u.owner == name]),
                        failing, mine, capped=False)
        if found and found[1].ok:
            return found
        if found and partial is None:
            partial = found
    if partial:
        return partial

    # Phase B: every function of mylib.
    rest = [u for u in units if u.owner not in scope]
    for group in (rest, [u for u in units if u.owner in scope]):
        found = _search(ctx, _single(cur, group), failing, failing, capped=True)
        if found:
            return found

    # Phase C: pairs of mutations inside the called functions.
    found = _search(ctx, _pairs(cur, [u for u in units if u.owner in scope] or units),
                    failing, failing, capped=True)
    if found:
        return found

    # Phase D: the best partial fix (fewest remaining failures) among single mutants.
    best = None
    for name, mine in scope.items():
        for cand, _m in _single(cur, [u for u in units if u.owner == name]):
            rep = ctx.candidate(cand, mine)
            if rep.ok or any(f.startswith("<") for f in rep.failed):
                continue
            if len(rep.failed) < len(mine) and (best is None or len(rep.failed) < best[0]):
                full = ctx.verify(cand, failing)
                if full is not None:
                    best = (len(rep.failed), cand, full)
    return (best[1], best[2]) if best else None


def _write_mylib(path: Path, text: str) -> None:
    tmp = path.with_name(f".mylib.py.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _mylib_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(module aliases, directly imported names) bound from `mylib` by the test file."""
    aliases: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "mylib"}
        elif isinstance(node, ast.ImportFrom) and node.module == "mylib":
            names |= {a.asname or a.name for a in node.names if a.name != "*"}
    return aliases, names


def contradictory_tests(test_src: str) -> bool:
    """True if one test method asserts the same mylib call (literal arguments) equal to two
    different literals, which no implementation can satisfy. Deliberately narrow."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return False
    aliases, names = _mylib_names(tree)
    if not aliases and not names:
        return False

    def is_mylib_call(c: ast.expr) -> bool:
        if not isinstance(c, ast.Call) or not (c.args or c.keywords):  # no-arg calls may be stateful
            return False
        f = c.func
        return ((isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases)
                or (isinstance(f, ast.Name) and f.id in names))

    def literal(n: ast.AST):
        return ast.literal_eval(n)

    seen: dict[tuple, object] = {}
    for cls in [None] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        scope = tree.body if cls is None else cls.body
        for fn in scope:
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "assertEqual" and len(node.args) == 2
                        and is_mylib_call(node.args[0])):
                    continue
                call = node.args[0]
                try:
                    args = [literal(x) for x in call.args]
                    kwargs = [(k.arg, literal(k.value)) for k in call.keywords]
                    expected = literal(node.args[1])
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    continue
                key = (cls.name if cls else None, fn.name, ast.dump(call.func), repr(args), repr(kwargs))
                if key not in seen:
                    seen[key] = expected
                    continue
                prev = seen[key]
                if prev != expected:  # assertEqual semantics: 1 == 1.0 == True
                    return True
    return False


def _solve(task: Task, time_limit: float) -> SolveResult:
    deadline = time.monotonic() + time_limit
    ws = Path(task.workspace)
    mylib = ws / "mylib.py"
    if not mylib.is_file():
        return SolveResult("unparsed", note="mylib.py not found")
    test_file = "test_mylib.py"
    if not (ws / test_file).is_file():
        cands = sorted(p.name for p in ws.glob("test_*.py"))
        if not cands:
            return SolveResult("unparsed", note="no test file")
        test_file = cands[0]
    original = mylib.read_text(encoding="utf-8")
    test_src = (ws / test_file).read_text(encoding="utf-8")
    refs = _test_refs(test_src)
    ast.parse(original)

    tmp = Path(tempfile.mkdtemp(prefix="coding_fix_"))
    try:
        sandbox = tmp / "ws"
        shutil.copytree(ws, sandbox, ignore=shutil.ignore_patterns("__pycache__"))
        ctx = _Run(sandbox, test_file, deadline)
        t0 = time.monotonic()
        report = ctx.run(original, None, BASE_TIMEOUT)
        base_time = time.monotonic() - t0
        ctx.cand_timeout = min(CAND_TIMEOUT_MAX, max(1.5, 4 * base_time))
        ctx.verify_timeout = min(BASE_TIMEOUT, max(ctx.cand_timeout, 2 * base_time))
        if report.ok:
            return SolveResult("solved", answer="DONE", note="tests already pass")
        if any(f.startswith("<") for f in report.failed):
            return SolveResult("unparsed", note=f"cannot localize failures: {report.failed}")
        cur = original
        while not report.ok:
            step = _repair_round(ctx, cur, report, refs)
            if step is None:
                if contradictory_tests(test_src):  # only after repair failed: saves model tokens
                    return SolveResult("infeasible", note="tests contradict each other")
                return SolveResult("unparsed", note=f"no repair found for {report.failed}")
            cur, report = step
        _write_mylib(mylib, cur)
        shutil.rmtree(ws / "__pycache__", ignore_errors=True)
        return SolveResult("solved", answer="DONE", note="repaired by mutation")
    except _OutOfTime:
        return SolveResult("unparsed", note="time limit reached")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def solve(task: Task, time_limit: float = config.CODING_TIME_LIMIT) -> SolveResult:
    try:
        return _solve(task, time_limit)
    except Exception as e:  # never raise out of a solver
        return SolveResult("unparsed", note=f"coding solver error: {e!r}")

"""Coding tasks: repair single-token bugs in workspace/mylib.py by inverse mutation.

0 LLM tokens. The failing tests tell which mylib functions to mutate; every candidate
is evaluated in a temporary copy of the workspace, never in the real one.
"""
import ast
import copy
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
from harness.solvers.probe import behaviour, inputs_from_call, visible_call
from harness.solvers.testrun import TestReport, run_in_dir
from harness.tasks import Task

CANDIDATE_CAP = 3000
CAND_TIMEOUT_MAX = 5.0
BASE_TIMEOUT = 20.0
AMBIG_MUTANT_CAP = 400
AMBIG_MAX_OPTIONS = 9
CRASH_PENALTY = 4.0


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


def _contradicted_callees(test_src: str) -> set[str]:
    """Names of mylib functions that one test method asserts equal to two different literals
    for the same literal arguments. Deliberately narrow."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return set()
    aliases, names = _mylib_names(tree)
    if not aliases and not names:
        return set()
    out: set[str] = set()

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
                    f = call.func
                    out.add(f.attr if isinstance(f, ast.Attribute) else f.id)
    return out


def contradictory_tests(test_src: str) -> bool:
    """True if one test method asserts the same mylib call (literal arguments) equal to two
    different literals, which no implementation can satisfy. Deliberately narrow."""
    return bool(_contradicted_callees(test_src))


_PURE_BUILTINS = {"len", "sum", "min", "max", "sorted", "abs", "range", "str", "int", "float", "round",
                  "bool", "list", "tuple", "dict", "set", "enumerate", "zip", "reversed", "any", "all",
                  "divmod", "pow", "ord", "chr", "map", "filter", "isinstance"}
_SIMPLE = (int, float, str, bool, type(None))


def _simple_literal(v) -> bool:
    """Builtin value without custom equality."""
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_simple_literal(x) for x in v)
    if isinstance(v, dict):
        return all(_simple_literal(k) and _simple_literal(x) for k, x in v.items())
    return False


def _callee_is_pure(tree: ast.Module, name: str) -> bool:
    """Same literal arguments always give the same result: a top-level function that reads only
    its parameters, its own locals, whitelisted builtins and other such functions."""
    top: dict[str, int] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top[n.name] = top.get(n.name, 0) + 1
        else:
            for t in ast.walk(n):
                if isinstance(t, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    top[t.name] = top.get(t.name, 0) + 2  # redefinition in a module-level if/try
                elif isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    top[t.id] = top.get(t.id, 0) + 2
                elif isinstance(t, (ast.Import, ast.ImportFrom)):
                    for a in t.names:
                        key = (a.asname or a.name).split(".")[0]
                        top[key] = top.get(key, 0) + 2
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    todo, seen = [name], set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        fn = funcs.get(cur)
        if fn is None or top.get(cur) != 1 or fn.decorator_list:
            return False
        a = fn.args
        if any(d is not None and not isinstance(d, ast.Constant) for d in a.defaults + a.kw_defaults):
            return False
        local = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
            elif isinstance(node, ast.arg):
                local.add(node.arg)
        for node in ast.walk(fn):
            if node is fn:
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef, ast.Await, ast.Yield,
                                 ast.YieldFrom, ast.With, ast.AsyncWith, ast.Try, ast.Delete)):
                return False
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") or not (isinstance(node.value, ast.Name) and node.value.id in local):
                    return False
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in local:
                    continue
                if node.id in funcs:
                    todo.append(node.id)
                elif node.id not in _PURE_BUILTINS or node.id in top:
                    return False
    return True


_PATCHY = {"mock", "patch", "monkeypatch", "patch_object", "MagicMock", "Mock"}
_REFLECT = {"setattr", "delattr", "reload", "import_module", "__import__", "__dict__", "skipTest", "skip"}
_SETUPS = {"setUp", "setUpClass", "setUpModule", "tearDown", "tearDownClass", "tearDownModule"}


def _plain_test_file(tree: ast.Module, bound: set[str]) -> bool:
    """The only file shape the file-wide proof trusts: imports, classes and a bare
    `unittest.main()` guard at module level; no local imports and no rebinding of `bound`
    (mylib aliases and names) or of `unittest` anywhere."""
    guarded = bound | {"unittest"}
    seen_names: dict[str, int] = {}
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                if a.name == "*":
                    return False
                if not (isinstance(st, ast.ImportFrom) and st.module == "mylib"
                        or isinstance(st, ast.Import) and a.name in ("unittest", "mylib")):
                    return False
                nm = (a.asname or a.name).split(".")[0]
                seen_names[nm] = seen_names.get(nm, 0) + 1
        elif isinstance(st, ast.If):
            if ast.dump(st.test) != ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body) or st.orelse:
                return False
            if not (len(st.body) == 1 and isinstance(st.body[0], ast.Expr)
                    and ast.dump(st.body[0].value) == ast.dump(ast.parse("unittest.main()", mode="eval").body)):
                return False
        elif isinstance(st, ast.ClassDef):
            if not _plain_case(st) or st.name in guarded:
                return False
        elif not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
            return False
    if seen_names.get("unittest") != 1 or any(seen_names.get(n, 0) != 1 for n in bound):
        return False
    if not any(isinstance(st, ast.Import) and any(a.name == "unittest" and a.asname is None for a in st.names)
               for st in tree.body):
        return False
    for cls in [st for st in tree.body if isinstance(st, ast.ClassDef)]:
        for fn in [x for x in cls.body if isinstance(x, ast.FunctionDef)]:
            for st in fn.body:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                    continue
                if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and isinstance(st.value.func, ast.Attribute)
                        and isinstance(st.value.func.value, ast.Name) and st.value.func.value.id == "self"
                        and st.value.func.attr.startswith("assert")):
                    return False
            # the generator's class tests: (lambda s: (s.push(1), s.pop()))(m.Stack())
            params = {a.arg for n in ast.walk(fn) if isinstance(n, ast.Lambda) for a in n.args.args}
            for n in ast.walk(fn):
                if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)):
                    return False
                if isinstance(n, ast.Attribute):
                    v = n.value
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)                             and isinstance(v.func.value, ast.Name) and v.func.value.id in bound:
                        v = v.func.value   # m.Interval(1, 2).length(): a fresh mylib object
                    base = v.id if isinstance(v, ast.Name) else None
                    if n.attr.startswith("_") or (base != "self" and base not in bound | params)                             or (base == "self" and not n.attr.startswith("assert")):
                        return False
    top_imports = {id(st) for st in tree.body if isinstance(st, (ast.Import, ast.ImportFrom))}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in top_imports:
            return False
        if isinstance(n, (ast.Delete, ast.Global, ast.Nonlocal, ast.NamedExpr)):
            return False
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in guarded:
            return False
        if isinstance(n, ast.arg) and n.arg in guarded:
            return False
        if isinstance(n, ast.ExceptHandler) and n.name in guarded:
            return False
        if isinstance(n, ast.Name) and n.id in {"exec", "eval", "compile", "globals", "locals", "vars",
                                                "getattr", "open"}:
            return False
    return True


def _plain_case(cls: ast.ClassDef) -> bool:
    """`class X(unittest.TestCase)` with nothing but a docstring and plain test methods."""
    if cls.decorator_list or cls.keywords or len(cls.bases) != 1:
        return False
    b = cls.bases[0]
    if not (isinstance(b, ast.Attribute) and b.attr == "TestCase"
            and isinstance(b.value, ast.Name) and b.value.id == "unittest"):
        return False
    for st in cls.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue
        if not (isinstance(st, ast.FunctionDef) and st.name.startswith("test") and not st.decorator_list):
            return False
        a = st.args
        if [x.arg for x in a.args] != ["self"] or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            return False
    return True


_REFLECTIVE_MODULES = {"os", "sys", "inspect", "importlib", "builtins"}


def _reflective(tree: ast.AST) -> bool:
    """True when the code can reach interpreter internals: a dunder name or attribute (other
    than `__name__` / `__main__`) or an import of os / sys / inspect / importlib / builtins."""
    for n in ast.walk(tree):
        name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
        if name and name.startswith("__") and name.endswith("__") and name not in ("__name__", "__main__"):
            return True
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] in _REFLECTIVE_MODULES for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in _REFLECTIVE_MODULES:
            return True
    return False


def _mentions_unittest(tree: ast.AST) -> bool:
    """A library that touches `unittest` can change what the assertions do."""
    return any((isinstance(n, ast.Name) and n.id == "unittest")
               or (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "unittest" for a in n.names))
               or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "unittest")
               for n in ast.walk(tree))


def _renames_mylib_name(tree: ast.AST) -> bool:
    """`from mylib import g as f`: the name checked for purity is not the function that runs."""
    return any(isinstance(n, ast.ImportFrom) and n.module == "mylib"
               and any(a.asname and a.asname != a.name for a in n.names) for n in ast.walk(tree))


def provable_contradiction(test_src: str, mylib_src: str) -> bool:
    """Strict proof for an early `infeasible`; any doubt -> False. Needs two plain top-level
    `self.assertEqual(<mylib call with literal args>, <literal>)` statements of one test method,
    the same call, expected literals unequal under Python ==, and a stateless callee."""
    try:
        tree = ast.parse(test_src)
        lib = ast.parse(mylib_src)
    except (SyntaxError, ValueError):
        return False
    aliases, names = _mylib_names(tree)
    if not aliases and not names:
        return False
    if not _plain_test_file(tree, aliases | names):
        return False
    if _renames_mylib_name(tree) or _reflective(tree) or _reflective(lib) \
            or _mentions_unittest(lib):
        return False
    for node in ast.walk(lib):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False
        if isinstance(node, ast.Name) and node.id in {"globals", "locals", "vars", "exec", "eval", "setattr",
                                                      "delattr", "__import__", "sys", "importlib", "builtins"}:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [getattr(node, "module", None) or ""] + [a.name for a in node.names]
            if any(x.split(".")[0] in ("mylib", "sys", "importlib", "builtins") for x in mods):
                return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _PATCHY:
            return False
        if isinstance(node, ast.Attribute) and node.attr in _PATCHY:
            return False
        if isinstance(node, ast.Name) and node.id in _REFLECT:
            return False
        if isinstance(node, ast.Attribute) and node.attr in _REFLECT:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [getattr(node, "module", None) or ""] + [a.name for a in node.names]
            if any(w in m for m in mods for w in ("mock", "patch", "pytest")):
                return False
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            tg = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in tg:
                for x in ast.walk(t):
                    if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in aliases:
                        return False
                    if isinstance(x, ast.Name) and (x.id in aliases or x.id in names):
                        return False  # rebinding the module or an imported name
        if isinstance(node, ast.FunctionDef) and node.name in _SETUPS:
            return False

    def callee(c: ast.expr) -> str | None:
        if not isinstance(c, ast.Call) or not (c.args or c.keywords):
            return None
        f = c.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases:
            return f.attr
        if isinstance(f, ast.Name) and f.id in names:
            return f.id
        return None

    seen: dict[tuple, object] = {}   # file-wide: methods and classes run independently
    cls_count: dict[str, int] = {}
    meth_count: dict[str, int] = {}   # test names are the harness's handle: keep them unique
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            cls_count[n.name] = cls_count.get(n.name, 0) + 1
            for x in n.body:
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    meth_count[x.name] = meth_count.get(x.name, 0) + 1

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and _plain_case(n)
                and cls_count[n.name] == 1]:
        counts: dict[str, int] = {}
        for x in cls.body:
            bound = [x.name] if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else [
                y.id for y in ast.walk(x) if isinstance(y, ast.Name) and isinstance(y.ctx, ast.Store)]
            for nm in bound:
                counts[nm] = counts.get(nm, 0) + 1
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.decorator_list:
                continue
            if not fn.name.startswith("test") or counts.get(fn.name) != 1 or meth_count[fn.name] != 1:
                continue  # never run by unittest, or redefined
            if any(isinstance(n, (ast.Return, ast.Raise)) for n in ast.walk(fn)):
                continue  # asserts may never run
            for st in fn.body:  # direct children only: no if/try/with/for around the asserts
                if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)):
                    continue
                node = st.value
                f = node.func
                if not (isinstance(f, ast.Attribute) and f.attr == "assertEqual"
                        and isinstance(f.value, ast.Name) and f.value.id == "self"
                        and len(node.args) == 2 and not node.keywords):
                    continue
                name = callee(node.args[0])
                if name is None:
                    continue
                call = node.args[0]
                try:
                    args = [ast.literal_eval(x) for x in call.args]
                    kws = [(k.arg, ast.literal_eval(k.value)) for k in call.keywords]
                    expected = ast.literal_eval(node.args[1])
                except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
                    continue
                if any(k is None for k, _ in kws) or not _simple_literal(expected) \
                        or not all(_simple_literal(x) for x in args + [v for _, v in kws]):
                    continue
                key = (name, repr(args), repr(kws))
                if key not in seen:
                    seen[key] = expected
                    continue
                try:
                    differs = bool(seen[key] != expected)
                except Exception:
                    differs = False
                if differs and _callee_is_pure(lib, name):
                    return True
    return False


def _norm(seg: str) -> str:
    try:
        return ast.unparse(ast.parse(textwrap.dedent(seg)))
    except SyntaxError:
        return seg


# ---- tie preference: which of several green fixes looks like the undone bug ----------------
# Ranks one fix against the buggy function (lower = preferred). Used only among fixes that all
# pass every visible test, so it decides ties, never correctness: a comparison flip (< <-> >)
# beats a strictness change (> -> >=, the "first occurrence" side stays strict) beats a negation
# (> -> <=, == -> !=); + <-> - and / <-> // beat other operator swaps; a constant moving towards
# zero beats one moving away; fixes that remove a no-op (x % 1, x // 1, x * 1, --x) win, fixes
# that create one lose. Equal ranks keep the mutant order (constants first).

_CMP_SIDE = {ast.Lt: "<", ast.LtE: "<", ast.Gt: ">", ast.GtE: ">"}
_NATURAL_SWAPS = ({ast.Add, ast.Sub}, {ast.Div, ast.FloorDiv})
_NATURAL_CALLS = {"sorted", "reversed"}


def _int_const(node) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int_const(node.operand)
        return None if v is None else -v
    return None


def _diff_pairs(a, b, out: list) -> None:
    if type(a) is not type(b):
        out.append((a, b))
        return
    if not isinstance(a, ast.AST):
        if a != b:
            out.append((a, b))
        return
    for f in a._fields:
        va, vb = getattr(a, f, None), getattr(b, f, None)
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                out.append((a, b))
                continue
            for x, y in zip(va, vb):
                _diff_pairs(x, y, out)
        elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
            _diff_pairs(va, vb, out)
        elif type(va) is not type(vb) or va != vb:
            out.append((a, b))
            return


def _pair_rank(a, b) -> float:
    va, vb = _int_const(a), _int_const(b)
    if va is not None and vb is not None:
        return 0.5 if abs(vb) > abs(va) else 0.0
    if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
        return 1.0
    ta, tb = type(a), type(b)
    if ta in _CMP_SIDE and tb in _CMP_SIDE:
        same_side = _CMP_SIDE[ta] == _CMP_SIDE[tb]
        same_strict = (ta in (ast.Lt, ast.Gt)) == (tb in (ast.Lt, ast.Gt))
        if same_strict and not same_side:
            return 0.0      # flip
        return 1.0 if same_side else 2.0   # strictness change / negation
    if isinstance(a, ast.operator) and isinstance(b, ast.operator):
        return 0.0 if {ta, tb} in _NATURAL_SWAPS else 2.0
    if isinstance(a, ast.Name) and isinstance(b, ast.Name):
        return 0.0 if {a.id, b.id} <= _NATURAL_CALLS else 1.0
    return 2.0


def _strip_zero(node):
    """`node` without its `+ 0` / `- 0` / `0 +` offsets, as a comparable dump."""
    class _T(ast.NodeTransformer):
        def visit_BinOp(self, b):
            self.generic_visit(b)
            if isinstance(b.op, (ast.Add, ast.Sub)) and _int_const(b.right) == 0:
                return b.left
            if isinstance(b.op, ast.Add) and _int_const(b.left) == 0:
                return b.right
            return b
    return ast.dump(_T().visit(copy.deepcopy(node)))


def _zero_offset_tautology(cmp: ast.Compare) -> bool:
    """`xs[i] >= xs[i + 0]`: a zero offset makes both sides the same expression. A zero offset
    alone is not degenerate: `n - 0` is how a mutation says "drop the spurious - 1"."""
    sides = [cmp.left] + cmp.comparators
    for a, b in zip(sides, sides[1:]):
        if ast.dump(a) != ast.dump(b) and _strip_zero(a) == _strip_zero(b):
            return True
    return False


def _degenerate(tree: ast.AST) -> float:
    """Weight of no-op code: x * 1, x * 0, x // 1, x % 1, x / 1, x ** 1, += 0, a tautology by a zero offset, sum(0 for ...),
    --x (3 each) and -0 (1)."""
    n = 0.0
    for x in ast.walk(tree):
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub):
            if isinstance(x.operand, ast.UnaryOp) and isinstance(x.operand.op, ast.USub):
                n += 3
            elif _int_const(x.operand) == 0:
                n += 1
        elif isinstance(x, (ast.BinOp, ast.AugAssign)):
            right = _int_const(x.right if isinstance(x, ast.BinOp) else x.value)
            left = _int_const(x.left) if isinstance(x, ast.BinOp) else None
            if isinstance(x.op, ast.Mult) and (right in (0, 1) or left in (0, 1)):
                n += 3
            elif isinstance(x.op, (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)) and right == 1:
                n += 3
            elif isinstance(x, ast.AugAssign) and isinstance(x.op, (ast.Add, ast.Sub)) and right == 0:
                n += 3
        elif isinstance(x, ast.Compare) and _zero_offset_tautology(x):
            n += 3
        elif (isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "sum"
              and len(x.args) == 1 and isinstance(x.args[0], ast.GeneratorExp)
              and _int_const(x.args[0].elt) == 0):
            n += 3
    return n


def _edit_rank(buggy: str, cand: str) -> float:
    try:
        a, b = ast.parse(textwrap.dedent(buggy)), ast.parse(textwrap.dedent(cand))
    except SyntaxError:
        return 99.0
    pairs: list = []
    _diff_pairs(a, b, pairs)
    return sum(_pair_rank(x, y) for x, y in pairs) + _degenerate(b) - _degenerate(a)


def _ambiguity(ctx: _Run, original: str, cur: str, test_src: str,
               failing: list[str]) -> tuple[dict | None, str]:
    """Fixed top-level functions with several green fixes that behave differently on probe
    inputs. Candidates are single mutants of the buggy function spliced into the chosen
    solution (other functions stay as chosen); one representative per behaviour cluster, best
    `_edit_rank` first. When another cluster ranks better than the phase-1 choice it becomes the
    choice (after a full green run). Returns (hint or None, the source to keep on disk)."""
    orig_units = {u.qualname: u for u in _units(original)}
    found: list[dict] = []
    asserts: list[str] = []

    def probe_timeout() -> float:
        left = ctx.deadline - time.monotonic()
        if left < 10:
            raise _OutOfTime()
        return min(5.0, left - 2)

    for qual in [u.qualname for u in _units(cur)]:
        u = {x.qualname: x for x in _units(cur)}.get(qual)   # fresh: an earlier switch may move lines
        old = orig_units.get(qual)
        if u is None or old is None or "." in qual or _segment(cur, u) == _segment(original, old):
            continue
        call = visible_call(test_src, qual)
        if call is None:
            continue
        args, stmt = call
        inputs = inputs_from_call(args)
        buggy = textwrap.dedent(_segment(original, old)).strip("\n")
        chosen_seg = textwrap.dedent(_segment(cur, u)).strip("\n")
        clusters: dict[tuple, list] = {}   # behaviour -> [rank, first order, source]
        try:
            chosen_key = tuple(behaviour(cur, qual, inputs, probe_timeout()))
            chosen_norm, chosen_order = _norm(chosen_seg), None
            seen: set[str] = set()
            tried = 0
            for m in tagged_mutants(_segment(original, old)):
                if tried >= AMBIG_MUTANT_CAP:
                    break
                seg = _norm(m.source)
                if seg in seen:
                    continue
                seen.add(seg)
                tried += 1
                if seg == chosen_norm:   # the phase-1 choice itself: its place in the mutant order
                    chosen_order = tried
                    continue
                cand = _splice(cur, u, m.source)
                if not ctx.candidate(cand, failing).ok or not ctx.run(cand, None, ctx.verify_timeout).ok:
                    continue
                key = tuple(behaviour(cand, qual, inputs, probe_timeout()))
                src = textwrap.dedent(m.source).strip("\n")
                rank = _edit_rank(buggy, src)
                if key not in clusters:
                    clusters[key] = [rank, tried, src]
                elif rank < clusters[key][0]:
                    clusters[key][0], clusters[key][2] = rank, src
            chosen_rank = _edit_rank(buggy, chosen_seg)
            if chosen_order is None:
                chosen_order = tried + 1
            if chosen_key in clusters:
                clusters[chosen_key] = [min(chosen_rank, clusters[chosen_key][0]),
                                        min(chosen_order, clusters[chosen_key][1]), chosen_seg]
            else:
                clusters[chosen_key] = [chosen_rank, chosen_order, chosen_seg]
            if len(clusters) < 2:
                continue
            # one crash may be a domain limit ([0] for "second largest", b = 0 for a division);
            # crashing on half of the probes is not
            crash = [k for k in clusters if 2 * sum(v.startswith("<exc:") for v in k) >= len(k)]
            if crash and len(crash) < len(clusters):
                for k in crash:
                    clusters[k][0] += CRASH_PENALTY
            ranked = sorted(clusters, key=lambda k: (clusters[k][0], clusters[k][1]))
            if ranked[0] != chosen_key:
                new = _splice(cur, u, clusters[ranked[0]][2])
                if ctx.run(new, None, ctx.verify_timeout).ok:
                    cur, chosen_key = new, ranked[0]
        except _OutOfTime:
            break   # keep what earlier functions already yielded
        keys = ranked[:AMBIG_MAX_OPTIONS]
        if chosen_key not in keys:
            keys[-1] = chosen_key
        found.append({"func": qual,
                      "buggy": buggy,
                      "candidates": [clusters[k][2] for k in keys],
                      "chosen": keys.index(chosen_key),
                      "calls": [f"{qual}({', '.join(map(repr, a))})" for a in inputs],
                      "outputs": [list(k) for k in keys]})
        asserts.append(stmt)
    if not found:
        return None, cur
    return {"ambiguous": found, "visible_asserts": asserts}, cur


def _tests_of(test_src: str, func: str) -> str:
    """Source of the test methods that use `func`."""
    refs = _test_refs(test_src)
    return "\n".join(textwrap.dedent(" " * fn.col_offset + (ast.get_source_segment(test_src, fn) or ""))
                     for cls in ast.parse(test_src).body if isinstance(cls, ast.ClassDef)
                     for fn in cls.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and func in refs.get(fn.name, ()))


def _verify_hint(original: str, cur: str, test_src: str) -> dict | None:
    """What a model needs to write its own fix of the changed top-level functions and what its fix is compared
    on; never our fix. Functions with no literal visible call have no probe inputs, so they are not checked."""
    orig_units = {u.qualname: u for u in _units(original)}
    items: list[dict] = []
    asserts: list[str] = []
    for u in _units(cur):
        old = orig_units.get(u.qualname)
        if old is None or "." in u.qualname or _segment(cur, u) == _segment(original, old):
            continue
        call = visible_call(test_src, u.qualname)
        if call is None:
            continue
        args, stmt = call
        items.append({"func": u.qualname, "buggy": textwrap.dedent(_segment(original, old)).strip("\n"),
                      "inputs": inputs_from_call(args), "tests": _tests_of(test_src, u.qualname) or stmt})
        asserts.append(stmt)
    return {"verify": items, "visible_asserts": asserts} if items else None


def _partial(ws: Path, mylib: Path, cur: str, report: TestReport) -> SolveResult:
    _write_mylib(mylib, cur)
    shutil.rmtree(ws / "__pycache__", ignore_errors=True)
    return SolveResult("solved", answer="DONE", tentative=True,
                       note=f"partial repair, still failing: {report.failed}")


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
        initially_failing = list(report.failed)
        cur = original
        try:
            while not report.ok:
                step = _repair_round(ctx, cur, report, refs)
                if step is None:
                    if provable_contradiction(test_src, original):
                        return SolveResult("infeasible", note="tests contradict each other")
                    if cur != original:  # best partial repair: a guess beats no answer
                        return _partial(ws, mylib, cur, report)
                    return SolveResult("unparsed", note=f"no repair found for {report.failed}")
                cur, report = step
        except _OutOfTime:
            if provable_contradiction(test_src, original):  # search is hopeless: no model calls
                return SolveResult("infeasible", note="tests contradict each other")
            if cur != original and not report.ok:
                return _partial(ws, mylib, cur, report)
            raise
        _write_mylib(mylib, cur)
        shutil.rmtree(ws / "__pycache__", ignore_errors=True)
        try:  # the deterministic choice is already on disk; ambiguity adds a hint and may
            # swap in a better-ranked green fix (tie preference)
            extra, preferred = _ambiguity(ctx, original, cur, test_src, initially_failing)
        except Exception:
            extra, preferred = None, cur
        if preferred != cur:
            _write_mylib(mylib, preferred)
            shutil.rmtree(ws / "__pycache__", ignore_errors=True)
        if extra:
            return SolveResult("solved", answer="DONE", note="repaired by mutation, ambiguous",
                               tentative=True, extra=extra)
        hint = _verify_hint(original, preferred, test_src) if config.verify_coding() else None
        if hint:   # on disk and answered already: a check that never comes changes nothing
            return SolveResult("solved", answer="DONE", note="repaired by mutation, unchecked",
                               tentative=True, extra=hint)
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

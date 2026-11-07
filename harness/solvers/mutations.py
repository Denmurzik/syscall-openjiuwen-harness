"""Single-point AST mutations of a function's source (for mutation-based repair)."""
import ast
import textwrap
from dataclasses import dataclass
from typing import Callable, Iterator

_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_AUG = (ast.Add, ast.Sub, ast.Mult)
_ORDER = (ast.Lt, ast.Gt, ast.LtE, ast.GtE)
_EQ = (ast.Eq, ast.NotEq)
_IN = (ast.In, ast.NotIn)
_IS = (ast.Is, ast.IsNot)

# Pairs that are plain single-token inversions of each other. ==/!=, in/not in and is/is not
# are deliberately absent: `x % 2 == 1` -> `!= 1` is a different fix than `== 0`.
_INVERSE = {
    (ast.Add, ast.Sub), (ast.Sub, ast.Add), (ast.Mult, ast.Div), (ast.Div, ast.Mult),
    (ast.Lt, ast.Gt), (ast.Gt, ast.Lt), (ast.LtE, ast.GtE), (ast.GtE, ast.LtE),
}
_CALL_SWAP = {"min": "max", "max": "min", "sorted": "reversed", "reversed": "sorted",
              "any": "all", "all": "any"}

# Priority groups: operators first, then calls, constants, slices.
P_OP, P_CALL, P_CONST, P_SLICE = 0, 1, 2, 3


@dataclass(frozen=True)
class Mutant:
    source: str
    kind: str
    inverse: bool
    prio: int


@dataclass(frozen=True)
class _Alt:
    kind: str
    inverse: bool
    prio: int
    apply: Callable[[ast.AST, "ast.AST | None"], None]


def _replace(parent: ast.AST, old: ast.AST, new: ast.AST) -> None:
    for field, val in ast.iter_fields(parent):
        if val is old:
            setattr(parent, field, new)
            return
        if isinstance(val, list):
            for i, v in enumerate(val):
                if v is old:
                    val[i] = new
                    return


def _is_neg_one(node: "ast.AST | None") -> bool:
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant) and node.operand.value == 1
            and type(node.operand.value) is int)


def _op_swaps(cur: type, group: tuple, setter, kind: str) -> list[_Alt]:
    alts = []
    for new in group:
        if new is cur:
            continue
        alts.append(_Alt(kind, (cur, new) in _INVERSE, P_OP,
                         lambda node, parent, new=new, setter=setter: setter(node, new)))
    return alts


def _not(expr: ast.expr) -> ast.UnaryOp:
    return ast.UnaryOp(op=ast.Not(), operand=expr)


def _flip_reverse(n: ast.Call, p) -> None:
    for k in n.keywords:
        if k.arg == "reverse":
            k.value.value = not k.value.value


def _alts(node: ast.AST, parent: "ast.AST | None") -> list[_Alt]:
    alts: list[_Alt] = []
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        alts += _op_swaps(type(node.op), _ARITH,
                          lambda n, new: setattr(n, "op", new()), "binop")
    elif isinstance(node, ast.AugAssign) and type(node.op) in _AUG:
        alts += _op_swaps(type(node.op), _AUG,
                          lambda n, new: setattr(n, "op", new()), "augassign")
    elif isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for group in (_ORDER, _EQ, _IN, _IS):
                if type(op) in group:
                    alts += _op_swaps(
                        type(op), group,
                        lambda n, new, i=i: n.ops.__setitem__(i, new()), "compare")
        if not (isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not)):
            alts.append(_Alt("add-not", False, P_OP, lambda n, p: _replace(p, n, _not(n))))
    elif isinstance(node, ast.BoolOp):
        new = ast.Or if isinstance(node.op, ast.And) else ast.And
        alts.append(_Alt("boolop", True, P_OP, lambda n, p, new=new: setattr(n, "op", new())))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        alts.append(_Alt("del-not", False, P_OP, lambda n, p: _replace(p, n, n.operand)))
    elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
        t = node.test
        if not isinstance(t, ast.Compare) and not (
                isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not)):
            alts.append(_Alt("add-not", False, P_OP,
                             lambda n, p: setattr(n, "test", _not(n.test))))
    elif isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            alts.append(_Alt("bool", True, P_CONST,
                             lambda n, p: setattr(n, "value", not n.value)))
        elif isinstance(v, int):
            in_cmp = isinstance(parent, ast.Compare)
            news: list[tuple[int, bool]] = []
            if v in (0, 1):
                news.append((1 - v, in_cmp))
            for d in (1, -1):
                if all(x != v + d for x, _ in news):
                    news.append((v + d, False))
            for new, inv in news:
                alts.append(_Alt("const", inv, P_CONST,
                                 lambda n, p, new=new: setattr(n, "value", new)))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _CALL_SWAP:
            alts.append(_Alt("call", True, P_CALL, lambda n, p: setattr(
                n.func, "id", _CALL_SWAP[n.func.id])))
        if any(kw.arg == "reverse" and isinstance(kw.value, ast.Constant)
               and isinstance(kw.value.value, bool) for kw in node.keywords):
            alts.append(_Alt("reverse", True, P_CALL, _flip_reverse))
        fname = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if fname in ("sorted", "sort") and not any(k.arg == "reverse" for k in node.keywords):
            alts.append(_Alt("reverse-add", False, P_CALL, lambda n, p: n.keywords.append(
                ast.keyword(arg="reverse", value=ast.Constant(value=True)))))
    elif isinstance(node, ast.Subscript):
        s = node.slice
        if isinstance(s, ast.Slice) and s.lower is None and s.upper is None and _is_neg_one(s.step):
            alts.append(_Alt("del-rev-slice", True, P_SLICE,
                             lambda n, p: _replace(p, n, n.value)))
    elif isinstance(node, ast.Return) and node.value is not None:
        v = node.value
        if not (isinstance(v, ast.Constant) and v.value is None) and not (
                isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Slice)
                and _is_neg_one(v.slice.step)):
            alts.append(_Alt("add-rev-slice", False, P_SLICE, lambda n, p: setattr(
                n, "value", ast.Subscript(
                    value=n.value,
                    slice=ast.Slice(lower=None, upper=None,
                                    step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))),
                    ctx=ast.Load()))))
    return alts


def _parents(tree: ast.AST) -> dict:
    out = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            out[id(c)] = p
    return out


def tagged_mutants(func_src: str) -> Iterator[Mutant]:
    """Yield single-point mutants, operator replacements first, without duplicates."""
    src = textwrap.dedent(func_src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    original = ast.unparse(tree)
    parents = _parents(tree)
    sites: list[tuple[int, int, _Alt]] = []
    for idx, node in enumerate(ast.walk(tree)):
        for k, alt in enumerate(_alts(node, parents.get(id(node)))):
            sites.append((idx, k, alt))
    sites.sort(key=lambda s: (s[2].prio, s[0], s[1]))
    seen = {original}
    for idx, k, alt in sites:
        fresh = ast.parse(src)
        fparents = _parents(fresh)
        node = list(ast.walk(fresh))[idx]
        parent = fparents.get(id(node))
        try:
            _alts(node, parent)[k].apply(node, parent)
            ast.fix_missing_locations(fresh)
            out = ast.unparse(fresh)
        except Exception:
            continue
        if out in seen:
            continue
        seen.add(out)
        yield Mutant(out, alt.kind, alt.inverse, alt.prio)


def mutants(func_src: str) -> Iterator[str]:
    for m in tagged_mutants(func_src):
        yield m.source

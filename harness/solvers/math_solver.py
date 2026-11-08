import ast
import re

from harness.result import SolveResult
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

_FLAGS = re.IGNORECASE
_THEN = r"(?:then\s+)?"

# Each rule: (regex matched against the whole sentence, kind).
_FRAC = {"half": 2, "third": 3, "quarter": 4, "fifth": 5, "tenth": 10}
_FRACS = "|".join(_FRAC)
_ART = r"(?:one|a|an)"
_INT = r"(\d{1,3}(?:,\d{3})+|\d+)"

_START = re.compile(rf"{_THEN}a warehouse starts with {_INT} box(?:es)?\.", _FLAGS)
_SPLIT = re.compile(
    rf"{_THEN}the stock is (?:split equally into|then divided evenly among|divided evenly among) {_INT} "
    r"warehouses(?: and it keeps one share|,? and this one retains a single share)"
    r" \(rounded down\)\.", _FLAGS)
_SCALE = re.compile(
    rf"{_THEN}(?:the stock is (doubled|tripled)(?: by (?:merging with a sister warehouse|a big merger))?"
    r"|an? (?:[a-z]+ ){0,4}merger(?: with an? [a-z]+ warehouse)? (?:then )?(doubles|triples) the stock)\.", _FLAGS)
_SHIP = re.compile(
    rf"{_THEN}(?:it ships {_INT} box(?:es)? per day for {_INT} days?"
    rf"|every day for {_INT} days?, {_INT} box(?:es)? leave the warehouse"
    rf"|for {_INT} days? in a row, {_INT} box(?:es)? (?:is|are) shipped out daily)\.", _FLAGS)
_RECEIVE = re.compile(
    rf"{_THEN}(?:it receives {_INT} (?:deliveries|delivery) of {_INT} box(?:es)? each"
    rf"|{_INT} (?:trucks arrive|truck arrives), each carrying {_INT} box(?:es)?"
    rf"|the warehouse takes in {_INT} shipments? of {_INT} box(?:es)? apiece)\.", _FLAGS)
_LOSS = re.compile(
    rf"{_THEN}(?:it loses {_INT} box(?:es)? to theft|thieves take {_INT} box(?:es)?"
    rf"|a leak ruins {_INT} box(?:es)?, which (?:is|are) thrown away|water damage destroys {_INT} box(?:es)?)\.", _FLAGS)
_MOVED = re.compile(
    rf"{_THEN}(?:{_ART} )?({_FRACS}) of the stock(?: \(rounded down\)|, rounded down,) "
    r"is (?:moved|transferred) to another site\.", _FLAGS)
_SELL = re.compile(
    rf"{_THEN}(?:it )?(?:then )?sells (?:{_ART} )?({_FRACS}) of the (?:boxes, rounding the number sold down"
    r"|stock, with the number of boxes sold rounded down)\.", _FLAGS)
_ADD_PCT = re.compile(
    rf"{_THEN}(?:the stock grows by {_INT} percent \(rounded down\)"
    rf"|a seasonal restock adds {_INT} percent of the current stock, rounded down)\.", _FLAGS)
_LOSE_PCT = re.compile(
    rf"{_THEN}(?:{_INT} percent of the stock \(rounded down\) spoils"
    rf"|spoilage removes {_INT} percent of the stock, rounded down)\.", _FLAGS)
_QUESTION = re.compile(
    rf"{_THEN}(?:how many boxes does it have at the end|what is the final stock level in boxes"
    r"|how many boxes remain in the end|how many boxes are left at the end)\?", _FLAGS)

_RULES = [
    (_START, "start"), (_SPLIT, "split"), (_SCALE, "scale"), (_SHIP, "ship"), (_RECEIVE, "receive"),
    (_LOSS, "loss"), (_MOVED, "moved"), (_SELL, "sell"), (_ADD_PCT, "add_pct"), (_LOSE_PCT, "lose_pct"),
    (_QUESTION, "question"),
]

_OPS = {
    "+": lambda x, n: x + n,
    "-": lambda x, n: x - n,
    "*": lambda x, n: x * n,
    "//": lambda x, n: x // n,
}


def _unparsed(note: str) -> SolveResult:
    return SolveResult("unparsed", note=note)


def _nums(m) -> list[int]:
    return [int(g.replace(",", "")) for g in m.groups() if g is not None and g.replace(",", "").isdigit()]


def parse(instruction: str) -> dict | SolveResult:
    start = None
    ops: list[list] = []
    steps: list[str] = []  # the same operations as expressions over x
    complex_op = False  # an operation that `ops` cannot express
    guard = False  # a loss: stock must not go below zero, else leave it to the model
    asked = False
    for sentence in split_sentences(strip_boilerplate(instruction)):
        sentence = " ".join(sentence.split())
        matches = [(m, kind) for rx, kind in _RULES if (m := rx.fullmatch(sentence))]
        if len(matches) != 1:
            return _unparsed(f"unrecognized sentence: {sentence!r}")
        m, kind = matches[0]
        if asked:
            return _unparsed(f"sentence after question: {sentence!r}")
        if kind == "start":
            if start is not None:
                return _unparsed("duplicate start")
            start = int(m.group(1).replace(",", ""))
            continue
        if kind == "question":
            asked = True
            continue
        if start is None:
            return _unparsed(f"operation before start: {sentence!r}")
        n = _nums(m)
        if kind == "split":
            op = ["//", n[0]]
        elif kind == "scale":
            word = (m.group(1) or m.group(2)).lower()
            op = ["*", 2 if word in ("doubled", "doubles") else 3]
        elif kind == "ship":
            op = ["-", n[0] * n[1]]
        elif kind == "receive":
            op = ["+", n[0] * n[1]]
        elif kind == "loss":
            op = ["-", n[0]]
            guard = True
        else:
            guard = True
            complex_op = True
            if kind in ("moved", "sell"):
                d = _FRAC[m.group(1).lower()]
                steps.append(f"x-x//{d}")
            elif kind == "add_pct":
                steps.append(f"x+x*{n[0]}//100")
            else:
                steps.append(f"x-x*{n[0]}//100")
            ops.append(["?", 0])
            continue
        ops.append(op)
        steps.append(f"x{op[0]}{op[1]}")
    if start is None:
        return _unparsed("no start value")
    if not asked:
        return _unparsed("no question")
    if complex_op or guard:
        spec = {"start": start, "steps": steps}
        if guard:
            spec["nonneg"] = True
        return spec
    return {"start": start, "ops": ops}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


_LIMIT = 10**18
_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.FloorDiv: lambda a, b: a // b}


class _Infeasible(Exception):
    pass


class _Unparsed(Exception):
    pass


def _eval(node: ast.AST, x: int) -> int:
    if isinstance(node, ast.Expression):
        return _eval(node.body, x)
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:
            raise _Unparsed("non-integer constant")
        return node.value
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise _Unparsed(f"unknown name {node.id!r}")
        return x
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, x)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, x), _eval(node.right, x)
        if isinstance(node.op, ast.FloorDiv) and b == 0:
            raise _Infeasible("division by zero")
        v = _BIN[type(node.op)](a, b)
        if abs(v) > _LIMIT:
            raise _Unparsed("value too large")
        return v
    raise _Unparsed(f"forbidden expression node {type(node).__name__}")


def execute_steps(spec: dict) -> SolveResult:
    """Safe evaluator for {"start": int, "steps": ["x-3*4", "x-x//4", ...]}."""
    try:
        x = spec["start"]
        steps = spec["steps"]
        if not _is_int(x) or abs(x) > _LIMIT or not isinstance(steps, list):
            return _unparsed("bad spec")
        went_negative = False
        for step in steps:
            if _is_int(step):  # a whole-placeholder step ("{N2}") arrives as an int: set x to it
                x = step
            elif isinstance(step, str) and len(step) <= 200:
                try:
                    tree = ast.parse(step.strip(), mode="eval")
                except (SyntaxError, ValueError, RecursionError):
                    return _unparsed(f"bad step syntax: {step!r}")
                x = _eval(tree, x)
            else:
                return _unparsed("bad step")
            if abs(x) > _LIMIT:
                return _unparsed("value too large")
            if spec.get("nonneg") and x < 0:
                went_negative = True
        if went_negative:  # kept only as a last-resort guess (the scheduler writes it if nothing better appears)
            return SolveResult("unparsed", note="stock below zero", extra={"best_guess": str(x)})
        return SolveResult("solved", str(x))
    except _Infeasible as e:
        return SolveResult("infeasible", note=str(e))
    except _Unparsed as e:
        return _unparsed(str(e))
    except Exception as e:
        return _unparsed(f"bad spec: {e!r}")


def execute(spec: dict) -> SolveResult:
    if isinstance(spec, dict) and "steps" in spec and "ops" not in spec:
        return execute_steps(spec)
    try:
        x = spec["start"]
        ops = spec["ops"]
        if not _is_int(x) or not isinstance(ops, list):
            return _unparsed("bad spec")
        for item in ops:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return _unparsed("bad op")
            op, n = item
            if op not in _OPS or not _is_int(n):
                return _unparsed(f"bad op: {item!r}")
        for op, n in ops:
            if op == "//" and n == 0:
                return SolveResult("infeasible", note="division by zero")
            x = _OPS[op](x, n)
        return SolveResult("solved", str(x))
    except Exception as e:  # malformed spec of any other shape
        return _unparsed(f"bad spec: {e!r}")


def solve(task: Task) -> SolveResult:
    try:
        spec = parse(task.instruction)
        if isinstance(spec, SolveResult):
            return spec
        return execute(spec)
    except Exception as e:
        return _unparsed(f"unexpected error: {e!r}")

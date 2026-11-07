import re

from harness.result import SolveResult
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

_FLAGS = re.IGNORECASE
_THEN = r"(?:then\s+)?"

# Each rule: (regex matched against the whole sentence, kind).
_START = re.compile(rf"{_THEN}a warehouse starts with (\d+) boxes\.", _FLAGS)
_SPLIT = re.compile(
    rf"{_THEN}the stock is split equally into (\d+) warehouses and it keeps one share \(rounded down\)\.", _FLAGS)
_DOUBLE = re.compile(rf"{_THEN}the stock is doubled by merging with a sister warehouse\.", _FLAGS)
_SHIP = re.compile(rf"{_THEN}it ships (\d+) boxes per day for (\d+) days\.", _FLAGS)
_RECEIVE = re.compile(rf"{_THEN}it receives (\d+) deliveries of (\d+) boxes each\.", _FLAGS)
_QUESTION = re.compile(rf"{_THEN}how many boxes does it have at the end\?", _FLAGS)

_RULES = [
    (_START, "start"), (_SPLIT, "split"), (_DOUBLE, "double"),
    (_SHIP, "ship"), (_RECEIVE, "receive"), (_QUESTION, "question"),
]

_OPS = {
    "+": lambda x, n: x + n,
    "-": lambda x, n: x - n,
    "*": lambda x, n: x * n,
    "//": lambda x, n: x // n,
}


def _unparsed(note: str) -> SolveResult:
    return SolveResult("unparsed", note=note)


def parse(instruction: str) -> dict | SolveResult:
    start = None
    ops: list[list] = []
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
            start = int(m.group(1))
        elif kind == "question":
            if asked:
                return _unparsed("repeated question")
            asked = True
        else:
            if start is None:
                return _unparsed(f"operation before start: {sentence!r}")
            if kind == "split":
                ops.append(["//", int(m.group(1))])
            elif kind == "double":
                ops.append(["*", 2])
            elif kind == "ship":
                ops.append(["-", int(m.group(1)) * int(m.group(2))])
            else:
                ops.append(["+", int(m.group(1)) * int(m.group(2))])
    if start is None:
        return _unparsed("no start value")
    if not asked:
        return _unparsed("no question")
    return {"start": start, "ops": ops}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def execute(spec: dict) -> SolveResult:
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

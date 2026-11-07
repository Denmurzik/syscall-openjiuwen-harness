import re
from pathlib import Path

from harness.result import SolveResult
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

_FLAGS = re.IGNORECASE
_MAX_PATH = 10
_SPEC_KEYS = {"entity", "path", "field"}

_INTRO = re.compile(r"the directory kb/ contains a small knowledge base of markdown files\.", _FLAGS)

# (regex over the question without trailing punctuation, field); group 1 is the entity expression.
_QUESTIONS = [
    (re.compile(r"in which year was (.+) born", _FLAGS), "Born"),
    (re.compile(r"in which year was (.+) founded", _FLAGS), "Founded"),
    (re.compile(r"in which city is the headquarters of (.+)", _FLAGS), "Headquarters"),
    (re.compile(r"where is (.+) headquartered", _FLAGS), "Headquarters"),
    (re.compile(r"who is the mentor of (.+)", _FLAGS), "Mentor"),
    (re.compile(r"who is (.+)['\u2019]s mentor", _FLAGS), "Mentor"),
    (re.compile(r"which company does (.+) work for", _FLAGS), "Employer"),
    (re.compile(r"who employs (.+)", _FLAGS), "Employer"),
    (re.compile(r"who is (.+)['\u2019]s employer", _FLAGS), "Employer"),
]

_POSSESSIVE = re.compile(r"(.+)['\u2019]s (employer|mentor)", _FLAGS)
_THE_OF = re.compile(r"the (employer|mentor) of (.+)", _FLAGS)


def _unparsed(note: str) -> SolveResult:
    return SolveResult("unparsed", note=note)


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def _is_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


# ---------------------------------------------------------------- knowledge base

def load_kb(kb_dir: Path) -> dict[str, dict[str, str]]:
    """Name from the `# Name` line -> {Key: value}. Files in name order; a repeated name keeps the first file."""
    files = sorted((p for p in Path(kb_dir).rglob("*.md") if p.is_file()), key=lambda p: (p.name, str(p)))
    kb: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        name = None
        fields: dict[str, str] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if name is None:
                if not line.startswith("# "):
                    break
                name = " ".join(line[2:].split())
                continue
            key, sep, value = line.partition(":")
            if sep and key.strip():
                fields.setdefault(key.strip(), value.strip())
        if name and _norm(name) not in seen:
            seen.add(_norm(name))
            kb[name] = fields
    return kb


# ---------------------------------------------------------------------- parsing

def _parse_entity(expr: str) -> tuple[str, list[str]] | None:
    """Return (name, path) for the entity grammar, or None when it is not understood or ambiguous."""
    expr = " ".join(expr.split())
    if not expr:
        return None
    poss = _POSSESSIVE.fullmatch(expr)
    the_of = _THE_OF.fullmatch(expr)
    if poss is not None and the_of is not None:
        return None  # "the mentor of X's employer": scope is ambiguous, do not guess
    if poss is not None:
        inner = _parse_entity(poss.group(1))
        if inner is None:
            return None
        return inner[0], inner[1] + [poss.group(2).capitalize()]
    if the_of is not None:
        inner = _parse_entity(the_of.group(2))
        if inner is None:
            return None
        return inner[0], inner[1] + [the_of.group(1).capitalize()]
    if "'s " in expr.lower() or "\u2019s " in expr.lower() or expr.lower().startswith("the "):
        return None
    return expr, []


def parse(instruction: str) -> dict | SolveResult:
    question = None
    for sentence in split_sentences(strip_boilerplate(instruction)):
        sentence = " ".join(sentence.split())
        if _INTRO.fullmatch(sentence):
            continue
        if question is not None:
            return _unparsed(f"extra sentence: {sentence!r}")
        question = sentence
    if question is None:
        return _unparsed("no question")
    body = question.rstrip("?.! ").strip()
    matches = []
    for rx, field in _QUESTIONS:
        m = rx.fullmatch(body)
        if m:
            ent = _parse_entity(m.group(1))
            if ent is not None:
                matches.append((ent, field))
    if not matches:
        return _unparsed(f"unrecognized question: {question!r}")
    if any(m != matches[0] for m in matches):
        return _unparsed(f"ambiguous question: {question!r}")
    (entity, path), field = matches[0]
    return {"entity": entity, "path": path, "field": field}


# -------------------------------------------------------------------- execution

def execute(spec: dict, workspace: Path) -> SolveResult:
    try:
        if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
            return _unparsed("bad spec")
        entity, path, field = spec["entity"], spec["path"], spec["field"]
        if not _is_str(entity) or not _is_str(field):
            return _unparsed("bad spec: entity/field")
        if not isinstance(path, list) or len(path) > _MAX_PATH or not all(_is_str(p) for p in path):
            return _unparsed("bad spec: path")

        kb_dir = Path(workspace) / "kb"
        if not kb_dir.is_dir():
            return SolveResult("infeasible", note="no kb directory")
        index = {_norm(name): fields for name, fields in load_kb(kb_dir).items()}

        def lookup(name: str, key: str) -> str | None:
            fields = index.get(_norm(name))
            if fields is None:
                return None
            for k, v in fields.items():
                if _norm(k) == _norm(key):
                    return v or None
            return None

        if _norm(entity) not in index:
            return SolveResult("infeasible", note=f"entity not found: {entity!r}")
        current = entity
        for key in path:
            nxt = lookup(current, key)
            if nxt is None:
                return SolveResult("infeasible", note=f"{current!r} has no {key!r}")
            if _norm(nxt) not in index:
                return SolveResult("infeasible", note=f"entity not found: {nxt!r}")
            current = nxt
        value = lookup(current, field)
        if value is None:
            return SolveResult("infeasible", note=f"{current!r} has no {field!r}")
        return SolveResult("solved", value)
    except Exception as e:
        return _unparsed(f"bad spec: {e!r}")


def solve(task: Task) -> SolveResult:
    try:
        spec = parse(task.instruction)
        if isinstance(spec, SolveResult):
            return spec
        return execute(spec, task.workspace)
    except Exception as e:
        return _unparsed(f"unexpected error: {e!r}")

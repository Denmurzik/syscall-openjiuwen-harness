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
    (re.compile(r"what year was (.+) born", _FLAGS), "Born"),
    (re.compile(r"when was (.+) founded", _FLAGS), "Founded"),
    (re.compile(r"where is the headquarters of (.+)", _FLAGS), "Headquarters"),
    (re.compile(r"in which city is (.+) headquartered", _FLAGS), "Headquarters"),
    (re.compile(r"who is the mentor of (.+)", _FLAGS), "Mentor"),
    (re.compile(r"who mentors (.+)", _FLAGS), "Mentor"),
    (re.compile(r"who is the employer of (.+)", _FLAGS), "Employer"),
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

def _parse_entity(expr: str, bare: bool = False) -> list[tuple[str, list[str]]]:
    """Readings of an entity expression as (name, path). Structural readings use the possessive /
    "the X of" grammar (a plain name when there is none); `bare=True` gives only the whole string as a name."""
    expr = " ".join(expr.split())
    if not expr:
        return []
    if bare:
        return [(expr, [])]
    out: list[tuple[str, list[str]]] = []
    poss = _POSSESSIVE.fullmatch(expr)
    if poss is not None:
        for name, path in _parse_entity(poss.group(1)):
            out.append((name, path + [poss.group(2).capitalize()]))
    the_of = _THE_OF.fullmatch(expr)
    if the_of is not None:
        for name, path in _parse_entity(the_of.group(2)):
            out.append((name, path + [the_of.group(1).capitalize()]))
    return out or [(expr, [])]


def _question(instruction: str) -> str | SolveResult:
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
    return question


def _specs(question: str, bare: bool) -> list[dict]:
    bodies = []
    for body in (question.rstrip("?.! ").strip(), question.rstrip("?! ").strip()):
        if body and body not in bodies:
            bodies.append(body)
    specs: list[dict] = []
    for body in bodies:
        for rx, field in _QUESTIONS:
            m = rx.fullmatch(body)
            if not m:
                continue
            for entity, path in _parse_entity(m.group(1), bare):
                spec = {"entity": entity, "path": path, "field": field}
                if spec not in specs:
                    specs.append(spec)
    return specs


def parse(instruction: str) -> dict | SolveResult:
    """One reading -> {"entity", "path", "field"}. Ambiguous scope ("the mentor of X's employer") ->
    {"alternatives": [spec, ...]}; `solve` runs each spec through the strict single-spec `execute`."""
    question = _question(instruction)
    if isinstance(question, SolveResult):
        return question
    specs = _specs(question, bare=False)
    if not specs:
        return _unparsed(f"unrecognized question: {question!r}")
    return specs[0] if len(specs) == 1 else {"alternatives": specs}


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
            return SolveResult("infeasible", note=f"start entity not found: {entity!r}")
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


def _run(specs: list[dict], workspace: Path) -> SolveResult:
    results = [execute(spec, workspace) for spec in specs]
    answers = {r.answer for r in results if r.status == "solved"}
    if len(answers) == 1:
        return SolveResult("solved", answers.pop())
    if len(answers) > 1:
        return _unparsed(f"ambiguous question, readings disagree: {sorted(answers)!r}")
    if any(r.status == "unparsed" for r in results):
        return _unparsed("bad spec")
    return SolveResult("infeasible", note="no reading has an answer")


def solve(task: Task) -> SolveResult:
    try:
        spec = parse(task.instruction)
        if isinstance(spec, SolveResult):
            return spec
        result = _run(spec.get("alternatives", [spec]), task.workspace)
        if result.status == "infeasible":
            # The name may itself contain "'s" / "the X of" / punctuation: retry with the whole string as a name.
            question = _question(task.instruction)
            if not isinstance(question, SolveResult):
                bare = [b for b in _specs(question, bare=True) if b != spec]
                if bare:
                    retry = _run(bare, task.workspace)
                    if retry.status == "solved":
                        return retry
        return result
    except Exception as e:
        return _unparsed(f"unexpected error: {e!r}")

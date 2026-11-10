"""Model fallbacks: program synthesis for answer tasks, function patches for coding tasks."""
import ast
import asyncio
import io
import re
import shutil
import tempfile
import textwrap
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from harness.llm import (LLMUnavailable, ModelFailing, ProxyDown,  # BudgetExhausted is deliberately not caught here
                         chat_or_alt)
from harness.result import SolveResult
from harness.sandbox import run_program
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.shape import DESCRIBE, NAME, expected_shape, mismatch
from harness.solvers.coding_solver import _write_mylib as write_mylib  # atomic writer reused
from harness.solvers.coding_solver import provable_contradiction
from harness.solvers.testrun import TestReport, run_in_dir
from harness.tasks import Task

from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON, PROGRAM_MAX_TOKENS  # noqa: E402,F401

PROGRAM_SYSTEM = ("Write a Python 3 program that prints only the final answer to stdout. "
                  "Never write answer.txt or any file: the task text mentions answer.txt, ignore that, "
                  "the answer is read from the program's output. "
                  "Files are in the current directory. "
                  "If the task cannot be answered from the given data, print IMPOSSIBLE. "
                  "The program must compute the answer itself from the values in the task, one statement per "
                  "sentence of the task, in the order of the text: never print a number you worked out in your head. "
                  "Output only code, no explanations, no markdown.")
SEARCH_SYSTEM = ("Write a short Python 3 program that prints only the final answer to stdout. "
                 "The knowledge base is ALREADY loaded into the dict `kb` (page title -> {field: value}, all values "
                 "are strings): do not open files, do not define parsers, use `kb` directly, a few lines. "
                 "A field that names a person or company is the title of that page: follow it with kb[...]. "
                 "If a page or field the question needs is missing, print IMPOSSIBLE. "
                 "Answer only with the field the question asks for; if that field does not exist for the entity, "
                 "print IMPOSSIBLE. A relation in the other direction (\"whose mentor is X\", \"employees of X\") or a count "
                 "is computed by scanning kb.items(): that is not a missing field. "
                 "Never write answer.txt or any file: the task text mentions answer.txt, ignore that, "
                 "the answer is read from the program's output. "
                 "Output only code, no comments, no markdown.")
# prepended by the harness to the model's search program; never crashes on a missing or empty kb/

KB_PREAMBLE = (
    "import os\n"
    "class KbMissing(KeyError):\n"
    "    pass\n"
    "class KbFieldMissing(KbMissing):\n"
    "    pass\n"
    "class _KbDict(dict):\n"
    "    def __missing__(self, key):\n"
    "        raise KbMissing(key)\n"
    "class _KbPage(dict):\n"
    "    def __missing__(self, key):\n"
    "        raise KbFieldMissing(key)\n"
    "kb = _KbDict()\n"
    "for _f in (sorted(os.listdir('kb')) if os.path.isdir('kb') else []):\n"
    "    _name, _d = None, _KbPage()\n"
    "    try:\n"
    "        _text = open(os.path.join('kb', _f), encoding='utf-8', errors='replace').read()\n"
    "    except OSError:\n"
    "        continue\n"
    "    for _ln in _text.splitlines():\n"
    "        if _ln.startswith('#') and _name is None:\n"
    "            _name = _ln.lstrip('#').strip()\n"
    "        elif ':' in _ln:\n"
    "            _k, _v = _ln.split(':', 1)\n"
    "            _d[_k.strip()] = _v.strip()\n"
    "    if _name:\n"
    "        kb[_name] = _d\n")


def kb_program(code: str) -> str:
    """The preamble plus the model's code, which runs under its own file name so that traceback lines are its own."""
    return KB_PREAMBLE + f"exec(compile({code!r}, '<program>', 'exec'), globals())\n"


PATCH_SYSTEM = ("You fix bugs in mylib.py so that the tests pass. Return only the corrected "
                "functions, no explanations, in this format, one block per function:\n"
                "FUNC <name>\n<complete new text of the function>\n"
                "For a method use FUNC <Class.method>. Do not return anything else.")

PREVIEW_TOTAL = 4096  # bytes
PREVIEW_FILE = 500    # bytes
PREVIEW_LINES = 5
FEEDBACK_LIMIT = 1024
TEST_OUTPUT_LIMIT = 2048
FILE_CAP = 12 * 1024

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", re.S)


def extract_code(text: str, join_all: bool = True) -> str:
    """Code from ``` fences: all blocks joined (patches) or only the first (programs)."""
    blocks = [m.group(1).strip("\n").rstrip() for m in _FENCE.finditer(text)]
    if blocks:
        return "\n".join(b for b in blocks if b) if join_all else blocks[0]
    t = text.strip("\n")
    if t.lstrip().startswith("```"):  # unterminated fence
        t = re.sub(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\n?", "", t)
    return t.rstrip()


# a leading "-" is a minus sign only when no digit/letter precedes it ("2024-03-15" is a date)
_NUM = re.compile(r"(?:(?<![\w])-)?\d+(?:\.\d+)?")


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_DECIMALS = re.compile(r"\bround(?:ed|ing)?\b[^.?!]*?\bto\s+(?:(?P<n>\d+|one|two|three|four|five|six)\s+(?:decimal(?:\s+places?|s)?|d\.?p\.?))\b", re.I)
_TAIL_DECIMALS = re.compile(r"(?<!given )\bto\s+(?P<n>\d+|one|two|three|four|five|six)\s+decimal(?:\s+places?|s)?\s*[?.!]?$", re.I)
_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# a sentence of its own that rounds the ANSWER ("Round (the answer) to 2 decimal places."), never the inputs
_ROUND_ANSWER = re.compile(
    r"^(?:please\s+)?(?:round|give|report|express|return|provide|present|write)\b"
    r"(?:\s+(?:the|your)\s+(?:final\s+)?(?:answer|result|value|number))?(?:\s+(?:rounded|correct))?"
    r"\s+to\s+(?P<n>\d+|one|two|three|four|five|six)\s+(?:decimal(?:\s+places?|s)?|d\.?p\.?)\s*[.!]?$", re.I)


def _requested_decimals(instruction: str | None) -> int | None:
    """Places asked for the answer: the rounding phrase of an asking sentence (one ending in '?'; with none,
    the last sentence), else a separate 'Round (the answer) to N decimal places.' sentence anywhere.
    Sentences about the inputs ('Prices are given to 2 decimal places') never count."""
    sentences = split_sentences(strip_boilerplate(instruction or ""))
    found = None
    asking = [s for s in sentences if s.rstrip().endswith("?")] or sentences[-1:]
    for s in asking:
        found = _DECIMALS.search(s) or _TAIL_DECIMALS.search(s)
        if found:
            break
    if not found:
        found = next((m for m in (_ROUND_ANSWER.match(s.strip()) for s in sentences) if m), None)
    if not found:
        return None
    word = found.group("n").lower()
    n = _WORDS.get(word) if word in _WORDS else int(word)
    return n if n is not None and n <= 6 else None


def _with_decimals(ans: str, instruction: str | None) -> str:
    """A lone number gets exactly the requested places, rounded as the phase-1 executor does (`_fmt`)."""
    n = _requested_decimals(instruction)
    if n is None or not _PLAIN_NUMBER.fullmatch(ans):
        return ans
    from harness.solvers.data_solver import _fmt
    try:
        return _fmt(Decimal(ans), "exact_str", n)
    except (InvalidOperation, ValueError):
        return ans


_PREFIX = re.compile(r"^(?:answer|result)\s*:\s*", re.I)


def normalize_answer(stdout: str, check: str | None, instruction: str | None = None) -> str | None:
    if check == "exact_int":
        nums = _NUM.findall(re.sub(r"(?<=\d),(?=\d{3})", "", stdout))
        if not nums:
            return None
        try:
            n = Decimal(nums[-1]).to_integral_value(rounding=ROUND_HALF_UP)
            out = format(n, "f")
        except (InvalidOperation, OverflowError, ValueError):
            return None
        return "0" if n == 0 else out
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    ans = _PREFIX.sub("", lines[-1]).strip()
    ans = ans.strip("\"'").strip()
    return _with_decimals(ans, instruction) if ans else None


def _is_text(raw: bytes) -> bool:
    if b"\0" in raw:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start >= len(raw) - 3  # cut inside a multibyte char
    return True


def _nbytes(t: str) -> int:
    return len(t.encode("utf-8"))


def describe_workspace(ws: Path) -> str:
    """File tree plus first lines of text files; at most PREVIEW_TOTAL bytes of UTF-8."""
    files = sorted(p for p in ws.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    names = [p.relative_to(ws).as_posix() for p in files]
    head = "Files:" if names else "Files: (none)"
    parts, used = [head], _nbytes(head)
    for n in names:  # whole lines only
        if used + _nbytes(n) + 1 > PREVIEW_TOTAL:
            return "\n".join(parts)
        parts.append(n)
        used += _nbytes(n) + 1
    for p, n in zip(files, names):
        try:
            with open(p, "rb") as f:
                raw = f.read(PREVIEW_FILE * 4)
        except OSError:
            continue
        if not _is_text(raw):
            continue
        text = raw.decode("utf-8", errors="ignore")
        first = "\n".join(text.splitlines()[:PREVIEW_LINES])
        first = first.encode("utf-8")[:PREVIEW_FILE].decode("utf-8", errors="ignore")
        block = f"--- {n} (first lines) ---\n{first}"
        if used + _nbytes(block) + 1 > PREVIEW_TOTAL:
            break
        parts.append(block)
        used += _nbytes(block) + 1
    return "\n".join(parts)


_NOT_CONST = object()


def is_bare_literal(code: str) -> bool:
    """True when the last print of the program shows a constant nobody computed: print(340), or a name that is
    only ever assigned constants. print(1200 - 40 * 5), print(result) and IMPOSSIBLE are not bare."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "print" and n.args]
    if not prints:
        return False
    consts: dict[str, object] = {}  # name -> its constant value while every assignment to it is a plain constant
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            ok = isinstance(n.value, ast.Constant)
            for t in n.targets:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        if ok and isinstance(t, ast.Name) and consts.get(nm.id, n.value.value) == n.value.value:
                            consts[nm.id] = n.value.value
                        else:
                            consts[nm.id] = _NOT_CONST
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.NamedExpr)):
            for nm in ast.walk(n.target):
                if isinstance(nm, ast.Name):
                    consts[nm.id] = _NOT_CONST
        elif isinstance(n, (ast.With, ast.FunctionDef, ast.Import, ast.ImportFrom)):
            for nm in ast.walk(n):
                if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Store):
                    consts[nm.id] = _NOT_CONST

    def value(e):
        """The constant an expression stands for, or _NOT_CONST."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.JoinedStr):
            if all(isinstance(v, ast.Constant) for v in e.values):
                return "".join(v.value for v in e.values)
            return _NOT_CONST
        if isinstance(e, ast.Name):
            return consts.get(e.id, _NOT_CONST)
        return _NOT_CONST

    def bare(call) -> bool:
        vals = [value(a) for a in call.args]
        return all(v is not _NOT_CONST and not (isinstance(v, str) and v.strip() == "IMPOSSIBLE") for v in vals)

    return all(bare(c) for c in prints)  # one computed print (or an IMPOSSIBLE verdict) makes it a real program


def _kb_examples(ws: Path) -> tuple[list[str], int]:
    """One example page per kind of file (the part of the name before the first "_") and the number of pages
    that the preamble loads into `kb` (files with a "# Title" line)."""
    kb_dir = ws / "kb"
    files = sorted(p for p in kb_dir.iterdir() if p.is_file()) if kb_dir.is_dir() else []
    out, seen, pages = [], set(), set()
    for f in files:
        name, d = None, {}
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            if ln.startswith("#") and name is None:
                name = ln.lstrip("#").strip()
            elif ":" in ln:
                k, v = ln.split(":", 1)
                d[k.strip()] = v.strip()
        if not name:
            continue
        pages.add(name)
        kind = f.name.split("_")[0]
        if kind in seen or len(out) >= 4:
            continue
        seen.add(kind)
        out.append(f"kb[{name!r}] == {d!r}")
    return out, len(pages)


def _is_kb_task(task: Task) -> bool:
    return task.task_type == "search" and (task.workspace / "kb").is_dir()


def _program_messages(task: Task) -> list[dict]:
    if _is_kb_task(task):
        ex, n = _kb_examples(task.workspace)
        user = (f"{strip_boilerplate(task.instruction)}\n\nkb has {n} pages, for example:\n" + "\n".join(ex))
        return [{"role": "system", "content": SEARCH_SYSTEM}, {"role": "user", "content": user}]
    user = f"{task.instruction}\n\n{describe_workspace(task.workspace)}"
    return [{"role": "system", "content": PROGRAM_SYSTEM}, {"role": "user", "content": user}]


def _kb_key_error(stderr: str) -> bool:
    """The traceback ends in KbMissing: the preamble's `kb` or one of its pages had no such key. A KeyError of
    the model's own dicts is something else and takes the normal retry."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return bool(lines) and lines[-1].startswith(("KbMissing", "KbFieldMissing"))


def _kb_field_missing(stderr: str) -> str | None:
    """The field a page was asked for and does not have (a page that is there: the name was found)."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines or not lines[-1].startswith("KbFieldMissing"):
        return None
    return lines[-1].partition(":")[2].strip().strip("'\"") or "?"


def _kb_fields(ws: Path) -> list[str]:
    fields: set[str] = set()
    for f in (sorted((ws / "kb").iterdir()) if (ws / "kb").is_dir() else []):
        try:
            for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in ln and not ln.startswith("#"):
                    fields.add(ln.split(":", 1)[0].strip())
        except OSError:
            continue
    return sorted(fields)[:20]


def _is_container_repr(answer: str) -> bool:
    """A printed dict, list, tuple, set or dict view (a whole page): never the one value a question asks for."""
    t = answer.strip()
    if t.startswith(("dict_items(", "dict_keys(", "dict_values(")):
        return True
    if not t.startswith(("{", "[", "(")):
        return False
    try:
        return isinstance(ast.literal_eval(t), (dict, list, tuple, set))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False


def _is_placeholder(text: str) -> bool:
    return text.strip().strip("'\"").strip().casefold().rstrip(".") in _PLACEHOLDERS


def _kb_placeholder(res, answer: str | None) -> str | None:
    """The program ran and printed None / unknown / n/a (raw last line or normalized answer: normalize_answer drops
    such a line for a number question, and strips "Answer:"). "clean": it is the only output line, the lookup found
    nothing. "mixed": other lines precede it (print(main()) after a real answer): the normal retry decides."""
    if res.returncode != 0 or res.timed_out:
        return None
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if not lines or not (_is_placeholder(lines[-1]) or (answer is not None and _is_placeholder(answer))):
        return None
    return "clean" if len(lines) == 1 else "mixed"


async def solve_with_program(task: Task, client, model: str = MODEL_FAST, shape_model: str = MODEL_ALT) -> SolveResult:
    """Up to two `model` programs; an answer of the wrong shape gets one more try on `shape_model`.
    A confirmation vote (scheduler) passes the same model twice: the vote is then one model's program."""
    messages = _program_messages(task)
    use_kb = _is_kb_task(task)
    note = ""
    attempt, attempts, shape_retried = 0, 2, False
    literal_retried = False
    field_retried = False
    guess = None  # the latest wrong-shape answer: a best guess for the scheduler's last pass
    while attempt < attempts:
        try:
            r = await chat_or_alt(client, model, messages, PROGRAM_MAX_TOKENS)
        except LLMUnavailable as e:
            return SolveResult("unparsed", note=f"llm unavailable: {e}",
                               extra=None if guess is None or field_retried else {"best_guess": guess, "shape_mismatch": True})
        if getattr(r, "truncated", False):  # a program cut at a line end still runs and prints a wrong answer
            note = "program cut by the budget limit"
            break
        code = extract_code(r.text, join_all=False)
        if not literal_retried and is_bare_literal(code):  # a number worked out in the head: one more try, no run
            literal_retried, attempts = True, attempts + 1
            note = "program printed a hard-coded literal"
            messages = messages + [{"role": "assistant", "content": r.text}, {"role": "user", "content":
                "Your program prints a constant that no code computed. Compute it in code from the values in the "
                "task and the files, one statement per sentence of the task. Print the final answer with print(); "
                "do not write it to a file. Output only code."}]
            attempt += 1
            continue
        # subprocess + copytree block for up to 20 s: keep them off the event loop
        res = await asyncio.to_thread(run_program, kb_program(code) if use_kb else code, task.workspace)
        field = _kb_field_missing(res.stderr) if use_kb and res.returncode != 0 and not res.timed_out else None
        if field is not None and not field_retried:
            # the page is there, the field is not: an invented field ("Mentee") or a field of the other kind of
            # page asked while scanning all pages. One more try with the fields that exist
            field_retried, attempts = True, attempts + 1
            note = f"kb page has no field {field!r}"
            messages = messages + [{"role": "assistant", "content": r.text}, {"role": "user", "content":
                f"Your program asked a page for the field {field!r}, which that page does not have. The only fields "
                f"are: {', '.join(_kb_fields(task.workspace))}; a page has only the fields of its kind (see the "
                "examples). For a relation in the other direction (who is mentored by X, who works for X) scan "
                "kb.items() and compare with .get(). If the question needs a fact that no field holds, print "
                "IMPOSSIBLE. Output only code."}]
            attempt += 1
            continue
        if use_kb and res.returncode != 0 and not res.timed_out and _kb_key_error(res.stderr):
            # clean "not found": no retry. A best_guess of a wrong-shape first attempt is dropped on purpose:
            # for an infeasible task a wrong-shape file on disk is worse than no file
            return SolveResult("unparsed", note="kb page or field not found")
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
            return SolveResult("infeasible", note="program printed IMPOSSIBLE")
        answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check, task.instruction)
        ph = _kb_placeholder(res, answer) if use_kb else None
        if ph == "clean":  # same as KbMissing, also when the shape check would only guess
            return SolveResult("unparsed", note="kb page or field not found")
        container = use_kb and (ph == "mixed" or (answer is not None and _is_container_repr(answer)))
        if container:
            answer = None  # a whole page or a trailing None is never the answer and never a best guess
        if answer is not None:
            bad = mismatch(task, answer)
            if bad is None:
                # the printed text is kept for the confirmation vote: it tells a computed integer from a rounded one
                return SolveResult("solved", answer=answer, note="model program", extra={"stdout": res.stdout[-FEEDBACK_LIMIT:]})
            guess = answer
            if shape_retried:
                return SolveResult("unparsed", note=f"answer shape: {bad}", extra=None if field_retried else
                                   {"best_guess": guess, "shape_mismatch": True})
            shape_retried, model, attempts = True, shape_model, attempt + 2  # exactly one more call
            note = f"answer shape: {bad}"
            feedback = (f"Your program printed {answer[:60]!r}, but the question asks for "
                        f"{DESCRIBE[expected_shape(task)]}. Re-read the question and fix the program.")
        else:
            why = "timed out" if res.timed_out else (
                f"exit code {res.returncode}" if res.returncode != 0 else "no usable answer printed")
            if container:
                why = "it printed a placeholder after other output" if ph == "mixed" else "it printed a whole page or list"
            note = f"program {why}"
            feedback = (f"Your program failed ({why}).\nstderr:\n{res.stderr[-FEEDBACK_LIMIT // 2:]}\n"
                        f"stdout:\n{res.stdout[-FEEDBACK_LIMIT // 2:]}")
        messages = messages + [{"role": "assistant", "content": r.text},
                               {"role": "user", "content": feedback + "\nFix it. Print the final answer with print(); "
                                           "do not write it to a file. Output only code."
                                           + (" Print only the one requested value." if container else "")}]
        attempt += 1
    return SolveResult("unparsed", note=note,
                       extra=None if guess is None or field_retried else {"best_guess": guess, "shape_mismatch": True})


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_BARE_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_BARE_NAME = re.compile(r"[^\W\d_][\w.,'&-]*(?: [^\W\d_][\w.,'&-]*){0,4}")
_PARTICLES = {"de", "da", "do", "dos", "das", "del", "della", "di", "du", "van", "von", "der", "den", "la", "le",
              "el", "al", "bin", "ibn", "of", "the", "and", "&", "y", "e", "ter", "ten", "dos"}
_PLACEHOLDERS = {"unknown", "none", "n/a", "na", "null", "nil", "impossible", "no answer", "not found",
                 "not available", "undefined", "unclear"}
_SENTENCE_WORDS = {"is", "are", "was", "were", "be", "it", "this", "that", "answer", "cannot", "can't", "cant",
                   "not", "no", "i", "we", "there", "has", "have", "had", "does", "do", "did"}


def _looks_like_name(line: str) -> bool:
    if not _BARE_NAME.fullmatch(line) or line.casefold().rstrip(".") in _PLACEHOLDERS:
        return False
    words = line.split()
    if line.endswith(".") and len(words) > 1 and not re.fullmatch(r"(?:[A-Z][\w'-]*\.? ?)+", line):
        return False
    if len(words) > 1 and line.endswith(".") and not words[-1][:1].isupper():
        return False
    for w in words:
        base = w.strip(".,")
        if base.casefold() in _SENTENCE_WORDS and base.casefold() not in {"the", "of"}:
            return False
        if not (base[:1].isupper() or base.casefold() in _PARTICLES):
            return False
    return not (len(words) > 1 and line.endswith(".") and words[0].casefold() in _SENTENCE_WORDS) and bool(words)


def bare_answer(text: str, task: Task) -> str | None:
    """The reply itself is the answer: after <think> and a fence, one line holding one value of the expected shape.
    Never IMPOSSIBLE, never a sentence, never code."""
    body = _THINK_RE.sub("", text)
    body = extract_code(body, join_all=False) if "```" in body else body
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    line = lines[0].strip("\"'").strip()
    shape = expected_shape(task)
    if _BARE_NUMBER.fullmatch(line):
        pass
    elif shape == NAME and _looks_like_name(line):
        pass
    else:
        return None
    answer = normalize_answer(line, task.check, task.instruction)
    if answer is None or answer == "IMPOSSIBLE" or mismatch(task, answer) is not None:
        return None
    return answer


async def solve_with_reason(task: Task, client) -> SolveResult:
    """One MODEL_REASON program, for the scheduler's last pass over what the cheap routes left open.
    An answer of the right shape settles it as solved; a last line IMPOSSIBLE gives the verdict infeasible."""
    try:
        r = await client.chat(MODEL_REASON, _program_messages(task), 3000, reasoning_effort="low")
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
    if getattr(r, "truncated", False):
        return SolveResult("unparsed", note="reason program cut by the budget limit")
    bare = bare_answer(r.text, task)  # the model answered without a program (a lone number)
    if bare is not None:
        return SolveResult("solved", answer=bare, note="model reason answer")
    use_kb = _is_kb_task(task)
    code = extract_code(r.text, join_all=False)
    res = await asyncio.to_thread(run_program, kb_program(code) if use_kb else code, task.workspace)
    if use_kb and res.returncode != 0 and not res.timed_out and _kb_key_error(res.stderr):
        return SolveResult("unparsed", note="kb page or field not found")
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
        return SolveResult("infeasible", note="reason program printed IMPOSSIBLE")
    answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check, task.instruction)
    ph = _kb_placeholder(res, answer) if use_kb else None
    if ph == "clean":
        return SolveResult("unparsed", note="kb page or field not found")
    if ph == "mixed":
        return SolveResult("unparsed", note="reason program printed a placeholder after other output")
    if answer is not None and use_kb and _is_container_repr(answer):
        return SolveResult("unparsed", note="reason program printed a whole page or list")
    if answer is None:
        return SolveResult("unparsed", note="reason program gave no usable answer")
    bad = mismatch(task, answer)
    if bad is not None:
        return SolveResult("unparsed", note=f"answer shape: {bad}",
                           extra={"best_guess": answer, "shape_mismatch": True})
    return SolveResult("solved", answer=answer, note="model reason program")


# ---------------------------------------------------------------- coding

_FUNC_HEAD = re.compile(r"^[ \t>#*`]*FUNC[ \t]+([A-Za-z_][\w.]*)[ \t*`:\r]*$", re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _locate(src: str) -> dict[str, tuple[int, int, int]]:
    """qualname -> (first line incl. decorators, end line exclusive, column), 0-based lines."""
    out: dict[str, tuple[int, int, int]] = {}

    def add(node, qual):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        out.setdefault(qual, (first - 1, node.end_lineno, node.col_offset))

    for node in ast.parse(src).body:
        if isinstance(node, _DEFS):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, _DEFS):
                    add(sub, f"{node.name}.{sub.name}")
    return out


def _indent(body: str, tree: ast.Module, col: int) -> str:
    """Indent code lines by `col`; continuation lines of multi-line strings stay as they are."""
    if not col:
        return body
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant)
                                               and isinstance(node.value, (str, bytes))):
            skip.update(range(node.lineno + 1, node.end_lineno + 1))
    ind = " " * col
    return "\n".join(ln if (i in skip or not ln.strip()) else ind + ln
                     for i, ln in enumerate(body.split("\n"), 1))


def _first_def(block: str) -> tuple[str, ast.Module] | None:
    """The first function definition in a block of model text: it may be wrapped in a fence and followed by
    explanations. Lines are cut from the end until what is left parses."""
    lines = block.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if ln.lstrip().startswith(("def ", "async def ", "@"))), None)
    if start is None:
        return None
    lines = lines[start:][:400]
    for end in range(len(lines), 0, -1):
        text = textwrap.dedent("\n".join(lines[:end]).strip("\n"))
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        if not tree.body or not isinstance(tree.body[0], _DEFS):
            return None
        text = "\n".join(text.split("\n")[:tree.body[0].end_lineno])
        return text, ast.parse(text)
    return None


def apply_patch(src: str, patch: str) -> str | None:
    """Replace whole functions named in `FUNC <name>` blocks; None when anything is off.
    The model may add a preamble, <think>, fences, **bold** headers and explanations after a function."""
    try:
        patch = _THINK.sub("", patch.replace("\r\n", "\n"))
        heads = list(_FUNC_HEAD.finditer(patch))
        if not heads:
            return None
        where = _locate(src)
        edits = []
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(patch)
            found = _first_def(patch[h.end():end])
            if found is None:
                return None
            body, tree = found
            name = h.group(1)
            if tree.body[0].name != name.rsplit(".", 1)[-1] or name not in where:
                return None
            start, stop, col = where[name]
            edits.append((start, stop, _indent(body.rstrip(), tree, col) + "\n"))
        spans = sorted((s, e) for s, e, _ in edits)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            return None
        lines = io.StringIO(src).readlines()  # splits on "\n" only, like ast line numbers
        for start, stop, text in sorted(edits, reverse=True):
            lines[start:stop] = [text]
        out = "".join(lines)
        ast.parse(out)
        return out
    except (SyntaxError, ValueError, RecursionError):
        return None


def _test_file(ws: Path) -> str:
    """Same discovery as phase 1: test_mylib.py, else the first test_*.py."""
    if (ws / "test_mylib.py").is_file():
        return "test_mylib.py"
    cands = sorted(p.name for p in ws.glob("test_*.py"))
    return cands[0] if cands else "test_mylib.py"


def _run_candidate(ws: Path, mylib_text: str) -> TestReport:
    tmp = Path(tempfile.mkdtemp(prefix="llm_fix_"))
    try:
        cand = tmp / "ws"
        shutil.copytree(ws, cand, ignore=shutil.ignore_patterns("__pycache__"))
        (cand / "mylib.py").write_text(mylib_text, encoding="utf-8", newline="\n")
        return run_in_dir(cand, _test_file(ws))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cap(text: str, limit: int = FILE_CAP) -> str:
    if len(text) <= limit:
        return text
    h = limit // 2
    return f"{text[:h]}\n...[{len(text) - limit} chars cut]...\n{text[-h:]}"


def _patch_messages(mylib: str, tests: str, output: str) -> list[dict]:
    user = (f"mylib.py:\n{_cap(mylib)}\n\ntest_mylib.py:\n{_cap(tests)}\n\n"
            f"Output of the failing tests:\n{output[-TEST_OUTPUT_LIMIT:]}\n\n"
            "Return only the corrected functions in the FUNC format.")
    return [{"role": "system", "content": PATCH_SYSTEM}, {"role": "user", "content": user}]


def _budget_is_free(client) -> bool:
    """More than half of the budget left: the reasoning step is affordable for any reward."""
    ledger = getattr(client, "ledger", None)
    try:
        return ledger is not None and ledger.remaining() > 0.5 * ledger.budget
    except Exception:
        return False  # fake or foreign clients: keep the old reward threshold


async def fix_coding(task: Task, client, resume: dict | None = None) -> SolveResult:
    """`resume`: the `ladder` of an earlier result that found the LLM unavailable. Models answer the same at
    temperature 0, so only the rungs that got no answer are bought again, from the partial fix reached so far."""
    resume = resume or {}
    ws = Path(task.workspace)
    mylib_path = ws / "mylib.py"
    try:
        base = resume.get("base") or mylib_path.read_text(encoding="utf-8")
        test_name = _test_file(ws)
        tests = (ws / test_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return SolveResult("unparsed", note=f"cannot read workspace: {e}")
    base_report = await asyncio.to_thread(_run_candidate, ws, base)
    if base_report.ok:
        return SolveResult("solved", answer="DONE", note="tests already pass")
    if provable_contradiction(tests, base):  # visible tests cannot all pass: spend nothing
        return SolveResult("infeasible", note="tests contradict each other")
    output = base_report.output
    attempts = [(MODEL_FAST, 400, {}), (MODEL_ALT, 600, {})]
    if task.reward >= 3 or _budget_is_free(client):
        attempts.append((MODEL_REASON, 3000, {"reasoning_effort": "low"}))
    note = "no attempt made"
    unavailable = None  # a rung that could not run: the task stays open for the scheduler's retry pass
    answered = list(resume.get("answered", ()))
    while True:
        failing = False
        for model, max_tokens, extra in [a for a in attempts if a[0] not in answered]:
            try:
                r = await client.chat(model, _patch_messages(base, tests, output), max_tokens, **extra)
            except ProxyDown as e:  # says nothing of this rung: the next one would only cost more. The retry pass asks it again
                unavailable = e
                break
            except ModelFailing as e:
                unavailable, failing = e, True
                continue
            except LLMUnavailable as e:
                unavailable = e
                continue
            answered.append(model)
            fixed = apply_patch(base, r.text)
            if fixed is None:
                note = f"{model}: patch did not apply"
                continue
            report = await asyncio.to_thread(_run_candidate, ws, fixed)
            if report.ok:
                write_mylib(mylib_path, fixed)
                shutil.rmtree(ws / "__pycache__", ignore_errors=True)
                return SolveResult("solved", answer="DONE", note=f"patched by {model}")
            note = f"{model}: tests still failing"
            output = report.output
            if len(report.failed) < len(base_report.failed) and not any(f.startswith("<") for f in report.failed):
                base, base_report = fixed, report  # keep partial progress for the next attempt
        if answered or not failing:
            break
        # no model of this ladder has answered yet: the task is not given up for a passing failure
        await client.wait_for_model([a[0] for a in attempts])
        unavailable = None
    if unavailable is None:
        return SolveResult("unparsed", note=note)
    return SolveResult("unparsed", note=f"llm unavailable: {unavailable}",
                       extra={"ladder": {"answered": answered, "base": base}})

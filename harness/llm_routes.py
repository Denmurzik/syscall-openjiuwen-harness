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
from harness.shape import DESCRIBE, expected_shape, mismatch
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
    last = max(prints, key=lambda n: (n.lineno, n.col_offset))
    consts: dict[str, bool] = {}  # name -> every assignment to it is a plain constant
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            is_const = isinstance(n.value, ast.Constant)
            for t in n.targets:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        consts[nm.id] = consts.get(nm.id, True) and is_const and isinstance(t, ast.Name)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.NamedExpr)):
            for nm in ast.walk(n.target):
                if isinstance(nm, ast.Name):
                    consts[nm.id] = False
        elif isinstance(n, (ast.With, ast.FunctionDef, ast.Import, ast.ImportFrom)):
            for nm in ast.walk(n):
                if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Store):
                    consts[nm.id] = False

    def constant(e) -> bool:
        if isinstance(e, ast.Constant):
            return not (isinstance(e.value, str) and e.value.strip() == "IMPOSSIBLE")
        if isinstance(e, ast.JoinedStr):
            return all(constant(v.value) if isinstance(v, ast.FormattedValue) else True for v in e.values)
        if isinstance(e, ast.Name):
            return consts.get(e.id) is True
        return False

    return all(constant(a) for a in last.args)


def _program_messages(task: Task) -> list[dict]:
    user = f"{task.instruction}\n\n{describe_workspace(task.workspace)}"
    return [{"role": "system", "content": PROGRAM_SYSTEM}, {"role": "user", "content": user}]


async def solve_with_program(task: Task, client) -> SolveResult:
    """Up to two MODEL_FAST programs; an answer of the wrong shape gets one more try on MODEL_ALT."""
    messages = _program_messages(task)
    note = ""
    model, attempt, attempts, shape_retried = MODEL_FAST, 0, 2, False
    literal_retried = False
    guess = None  # the latest wrong-shape answer: a best guess for the scheduler's last pass
    while attempt < attempts:
        try:
            r = await chat_or_alt(client, model, messages, PROGRAM_MAX_TOKENS)
        except LLMUnavailable as e:
            return SolveResult("unparsed", note=f"llm unavailable: {e}",
                               extra=None if guess is None else {"best_guess": guess, "shape_mismatch": True})
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
        res = await asyncio.to_thread(run_program, code, task.workspace)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
            return SolveResult("infeasible", note="program printed IMPOSSIBLE")
        answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check, task.instruction)
        if answer is not None:
            bad = mismatch(task, answer)
            if bad is None:
                return SolveResult("solved", answer=answer, note="model program")
            guess = answer
            if shape_retried:
                return SolveResult("unparsed", note=f"answer shape: {bad}", extra={"best_guess": guess, "shape_mismatch": True})
            shape_retried, model, attempts = True, MODEL_ALT, attempt + 2  # exactly one more call
            note = f"answer shape: {bad}"
            feedback = (f"Your program printed {answer[:60]!r}, but the question asks for "
                        f"{DESCRIBE[expected_shape(task)]}. Re-read the question and fix the program.")
        else:
            why = "timed out" if res.timed_out else (
                f"exit code {res.returncode}" if res.returncode != 0 else "no usable answer printed")
            note = f"program {why}"
            feedback = (f"Your program failed ({why}).\nstderr:\n{res.stderr[-FEEDBACK_LIMIT // 2:]}\n"
                        f"stdout:\n{res.stdout[-FEEDBACK_LIMIT // 2:]}")
        messages = messages + [{"role": "assistant", "content": r.text},
                               {"role": "user", "content": feedback + "\nFix it. Print the final answer with print(); "
                                           "do not write it to a file. Output only code."}]
        attempt += 1
    return SolveResult("unparsed", note=note, extra=None if guess is None else {"best_guess": guess, "shape_mismatch": True})


async def solve_with_reason(task: Task, client) -> SolveResult:
    """One MODEL_REASON program, for the scheduler's last pass over what the cheap routes left open.
    An answer of the right shape settles it as solved; a last line IMPOSSIBLE gives the verdict infeasible."""
    try:
        r = await client.chat(MODEL_REASON, _program_messages(task), 3000, reasoning_effort="low")
    except LLMUnavailable as e:
        return SolveResult("unparsed", note=f"llm unavailable: {e}")
    if getattr(r, "truncated", False):
        return SolveResult("unparsed", note="reason program cut by the budget limit")
    res = await asyncio.to_thread(run_program, extract_code(r.text, join_all=False), task.workspace)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
        return SolveResult("infeasible", note="reason program printed IMPOSSIBLE")
    answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check, task.instruction)
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

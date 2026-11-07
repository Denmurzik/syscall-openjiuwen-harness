"""Model fallbacks: program synthesis for answer tasks, function patches for coding tasks."""
import ast
import re
import shutil
import tempfile
from pathlib import Path

from harness.llm import BudgetExhausted, LLMUnavailable  # noqa: F401  (BudgetExhausted propagates)
from harness.result import SolveResult
from harness.sandbox import run_program
from harness.solvers.coding_solver import _write_mylib as write_mylib  # public alias
from harness.solvers.testrun import TestReport, run_in_dir
from harness.tasks import Task

MODEL_FAST = "qwen3.6-35b-a3b-noreason"
MODEL_MID = "qwen3.8-27b-noreason"
MODEL_REASON = "qwen3.8-27b"

PROGRAM_SYSTEM = ("Write a Python 3 program that prints only the final answer. "
                  "Files are in the current directory. "
                  "If the task cannot be answered from the given data, print IMPOSSIBLE. "
                  "Output only code, no explanations, no markdown.")
PATCH_SYSTEM = ("You fix bugs in mylib.py so that the tests pass. Return only the corrected "
                "functions, no explanations, in this format, one block per function:\n"
                "FUNC <name>\n<complete new text of the function>\n"
                "For a method use FUNC <Class.method>. Do not return anything else.")

PREVIEW_TOTAL = 4096
PREVIEW_FILE = 500
PREVIEW_LINES = 5
FEEDBACK_LIMIT = 1024
TEST_OUTPUT_LIMIT = 2048

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip("\n").rstrip()
    t = text.strip("\n")
    if t.lstrip().startswith("```"):  # unterminated fence
        t = re.sub(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\n?", "", t)
    return t.rstrip()


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(stdout: str, check: str | None) -> str | None:
    if check == "exact_int":
        nums = _NUM.findall(re.sub(r"(?<=\d),(?=\d{3})", "", stdout))
        if not nums:
            return None
        return str(int(round(float(nums[-1]))))
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    ans = re.sub(r"^answer\s*:\s*", "", lines[-1], flags=re.I).strip()
    ans = ans.strip("\"'").strip()
    return ans or None


def _is_text(raw: bytes) -> bool:
    if b"\0" in raw:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start >= len(raw) - 3  # cut inside a multibyte char
    return True


def describe_workspace(ws: Path) -> str:
    files = sorted(p for p in ws.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    tree = "\n".join(p.relative_to(ws).as_posix() for p in files)
    parts = [f"Files:\n{tree}" if tree else "Files: (none)"]
    used = len(parts[0])
    for p in files:
        try:
            with open(p, "rb") as f:
                raw = f.read(PREVIEW_FILE * 4)
        except OSError:
            continue
        if not _is_text(raw):
            continue
        text = raw.decode("utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:PREVIEW_LINES])
        head = head.encode("utf-8")[:PREVIEW_FILE].decode("utf-8", errors="ignore")
        block = f"--- {p.relative_to(ws).as_posix()} (first lines) ---\n{head}"
        if used + len(block) + 1 > PREVIEW_TOTAL:
            break
        parts.append(block)
        used += len(block) + 1
    return "\n".join(parts)[:PREVIEW_TOTAL]


def _program_messages(task: Task) -> list[dict]:
    user = f"{task.instruction}\n\n{describe_workspace(task.workspace)}"
    return [{"role": "system", "content": PROGRAM_SYSTEM}, {"role": "user", "content": user}]


async def solve_with_program(task: Task, client) -> SolveResult:
    messages = _program_messages(task)
    note = ""
    for _ in range(2):
        try:
            r = await client.chat(MODEL_FAST, messages, 400)
        except LLMUnavailable as e:
            return SolveResult("unparsed", note=f"llm unavailable: {e}")
        code = extract_code(r.text)
        res = run_program(code, task.workspace)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
            return SolveResult("infeasible", note="program printed IMPOSSIBLE")
        answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check)
        if answer is not None:
            return SolveResult("solved", answer=answer, note="model program")
        why = "timed out" if res.timed_out else (
            f"exit code {res.returncode}" if res.returncode != 0 else "no usable answer printed")
        note = f"program {why}"
        feedback = (f"Your program failed ({why}).\nstderr:\n{res.stderr[-FEEDBACK_LIMIT // 2:]}\n"
                    f"stdout:\n{res.stdout[-FEEDBACK_LIMIT // 2:]}")
        messages = messages + [{"role": "assistant", "content": r.text},
                               {"role": "user", "content": feedback + "\nFix it. Output only code."}]
    return SolveResult("unparsed", note=note)


# ---------------------------------------------------------------- coding

_FUNC_HEAD = re.compile(r"^FUNC[ \t]+([A-Za-z_][\w.]*)[ \t]*$", re.M)
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


def apply_patch(src: str, patch: str) -> str | None:
    """Replace whole functions named in `FUNC <name>` blocks; None when anything is off."""
    try:
        if "```" in patch:
            patch = extract_code(patch)
        heads = list(_FUNC_HEAD.finditer(patch))
        if not heads or patch[:heads[0].start()].strip():
            return None
        where = _locate(src)
        edits = []
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(patch)
            body = patch[h.end():end].strip("\n")
            name = h.group(1)
            tree = ast.parse(body)
            if len(tree.body) != 1 or not isinstance(tree.body[0], _DEFS) \
                    or tree.body[0].name != name.rsplit(".", 1)[-1] or name not in where:
                return None
            start, stop, col = where[name]
            ind = " " * col
            text = "".join(ind + ln if ln.strip() else ln
                           for ln in body.rstrip().splitlines(keepends=True)) + "\n"
            edits.append((start, stop, text))
        spans = sorted((s, e) for s, e, _ in edits)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            return None
        lines = src.splitlines(keepends=True)
        for start, stop, text in sorted(edits, reverse=True):
            lines[start:stop] = [text]
        out = "".join(lines)
        ast.parse(out)
        return out
    except (SyntaxError, ValueError, RecursionError):
        return None


def _run_candidate(ws: Path, mylib_text: str) -> TestReport:
    tmp = Path(tempfile.mkdtemp(prefix="llm_fix_"))
    try:
        cand = tmp / "ws"
        shutil.copytree(ws, cand, ignore=shutil.ignore_patterns("__pycache__"))
        (cand / "mylib.py").write_text(mylib_text, encoding="utf-8", newline="\n")
        return run_in_dir(cand)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _patch_messages(mylib: str, tests: str, output: str) -> list[dict]:
    user = (f"mylib.py:\n{mylib}\n\ntest_mylib.py:\n{tests}\n\n"
            f"Output of the failing tests:\n{output[-TEST_OUTPUT_LIMIT:]}\n\n"
            "Return only the corrected functions in the FUNC format.")
    return [{"role": "system", "content": PATCH_SYSTEM}, {"role": "user", "content": user}]


async def fix_coding(task: Task, client) -> SolveResult:
    ws = Path(task.workspace)
    mylib_path = ws / "mylib.py"
    try:
        base = mylib_path.read_text(encoding="utf-8")
        tests = (ws / "test_mylib.py").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return SolveResult("unparsed", note=f"cannot read workspace: {e}")
    base_report = _run_candidate(ws, base)
    if base_report.ok:
        return SolveResult("solved", answer="DONE", note="tests already pass")
    output = base_report.output
    attempts = [(MODEL_FAST, 400, {}), (MODEL_MID, 600, {})]
    if task.reward >= 3:
        attempts.append((MODEL_REASON, 3000, {"reasoning_effort": "low"}))
    note = "no attempt made"
    for model, max_tokens, extra in attempts:
        try:
            r = await client.chat(model, _patch_messages(base, tests, output), max_tokens, **extra)
        except LLMUnavailable as e:
            return SolveResult("unparsed", note=f"llm unavailable: {e}")
        fixed = apply_patch(base, r.text)
        if fixed is None:
            note = f"{model}: patch did not apply"
            continue
        report = _run_candidate(ws, fixed)
        if report.ok:
            write_mylib(mylib_path, fixed)
            shutil.rmtree(ws / "__pycache__", ignore_errors=True)
            return SolveResult("solved", answer="DONE", note=f"patched by {model}")
        note = f"{model}: tests still failing"
        output = report.output
        if len(report.failed) < len(base_report.failed) and not any(f.startswith("<") for f in report.failed):
            base, base_report = fixed, report  # keep partial progress for the next attempt
    return SolveResult("unparsed", note=note)

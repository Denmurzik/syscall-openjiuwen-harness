"""Model fallbacks: program synthesis for answer tasks, function patches for coding tasks."""
import ast
import asyncio
import difflib
import io
import logging
import re
import shutil
import tempfile
import textwrap
import time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from harness.llm import (LLMUnavailable, ModelFailing, ProxyDown,  # BudgetExhausted is deliberately not caught here
                         chat_or_alt)
from harness.oj_agent import call_tool
from harness.result import SolveResult
from harness.sandbox import run_program
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.shape import DESCRIBE, NAME, expected_shape, mismatch
from harness.solvers.coding_solver import _write_mylib as write_mylib  # atomic writer reused
from harness.solvers.coding_solver import missing_test_import, provable_contradiction
from harness.solvers.testrun import TestReport, run_in_dir
from harness.tasks import Task
from harness.trajectory import CALLS

from harness import config  # noqa: E402
from harness.config import MODEL_ALT, MODEL_FAST, MODEL_REASON, PROGRAM_MAX_TOKENS  # noqa: E402,F401

log = logging.getLogger("harness")

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
    "pages = _KbDict()\n"
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
    "        kb[_name] = _d\n"
    "        pages[_name] = _text\n")


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
RICH_JSON_MAX = 2_000_000   # bytes: a larger .json gets no structure line (it is parsed whole)
RICH_XLSX_MAX = 1_000_000   # bytes: a larger workbook shows its sheet names only (openpyxl parses every shared string)
RICH_SQL_STEPS = 200_000    # SQLite virtual-machine steps a statement of the preview may take
RICH_ZIP_MEMBERS = 5000     # an archive with more members is not opened (its central directory is read whole)
_RICH_CACHE: dict = {}      # (path, size, mtime_ns, text?, shown preview) -> the block added for it, or None
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


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _binary_preview(p: Path, raw: bytes) -> str | None:
    """HARNESS_RICH_PREVIEW: a few lines on what a file that is not text holds, None when it is not a kind read here
    or cannot be read. Nothing is written: the database is opened read-only and immutable (no journal file)."""
    try:
        if raw.startswith(b"SQLite format 3\x00"):
            import sqlite3
            con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro&immutable=1", uri=True, timeout=2)
            try:
                con.set_progress_handler(lambda: 1, RICH_SQL_STEPS)   # a statement that runs long is interrupted
                lines = ["(SQLite database: tables and first rows)"]
                tables = con.execute("select name, sql, type from sqlite_master where type in ('table', 'view') "
                                     "and name not like 'sqlite_%' order by name").fetchmany(8)
                for name, sql, kind in tables:
                    lines.append(_clip(sql or name, 300))
                    if kind != "table":   # the rows of a view may cost a whole query: its schema is enough
                        continue
                    quoted = '"' + name.replace('"', '""') + '"'
                    lines += ["  " + _clip(repr(row)) for row in con.execute(f"select * from {quoted} limit 2")]
            finally:
                con.close()
            return "\n".join(lines)
        if raw.startswith(b"PK\x03\x04"):
            import zipfile
            count = _zip_members(p)
            if count is None:   # damaged, or zip64: only named
                return None
            if count > RICH_ZIP_MEMBERS:
                return f"(zip archive, {count} members)"
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                if any(n.startswith("xl/") for n in names):
                    if p.stat().st_size > RICH_XLSX_MAX:
                        return _xlsx_sheet_names(z)
                    import openpyxl
                    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
                    try:
                        lines = ["(Excel workbook: sheets and first rows)"]
                        for sheet in wb.worksheets[:6]:
                            lines.append(f"sheet {sheet.title!r}:")
                            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                                if i >= 3:
                                    break
                                lines.append("  " + _clip(repr(row)))
                    finally:
                        wb.close()
                    return "\n".join(lines)
                lines = [f"(zip archive, {len(names)} members)"] + names[:12]
                first = next((n for n in names if not n.endswith("/")), None)
                if first is not None:
                    with z.open(first) as f:
                        head = f.read(PREVIEW_FILE)
                    if _is_text(head):
                        lines.append(f"first lines of {first}:")
                        lines += head.decode("utf-8", errors="ignore").splitlines()[:PREVIEW_LINES]
                return "\n".join(lines)
        if raw.startswith(b"\x1f\x8b"):
            import gzip
            with gzip.open(p, "rb") as f:
                head = f.read(PREVIEW_FILE)
            if _is_text(head):
                return "\n".join(["(gzip file, first lines after decompression)"]
                                 + head.decode("utf-8", errors="ignore").splitlines()[:PREVIEW_LINES])
    except Exception:   # a damaged or unusual file: the prompt names it, as before
        return None
    return None


def _zip_members(p: Path) -> int | None:
    """The member count of a zip archive from its end record (the last 64 KiB), None when it holds more than a
    plain record can say (zip64) or no record is found."""
    import struct
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65_557))
        tail = f.read()
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or len(tail) - i < 22:
        return None
    total = struct.unpack("<H", tail[i + 10:i + 12])[0]
    return None if total == 0xFFFF else total


def _xlsx_sheet_names(z) -> str | None:
    """The sheet names of a workbook too large to open, from xl/workbook.xml alone."""
    import html
    try:
        with z.open("xl/workbook.xml") as f:
            xml = f.read(200_000).decode("utf-8", errors="ignore")
    except (KeyError, OSError, RuntimeError):
        return None
    names = [html.unescape(m) for m in re.findall(r'<(?:\w+:)?sheet\b[^>]*?\bname="([^"]*)"', xml)][:12]
    if not names:
        return None
    return _clip("(Excel workbook, too large to show rows here; sheets: " + ", ".join(map(repr, names)) + ")", 480)


def _json_shape(p: Path, shown: str) -> str | None:
    """HARNESS_RICH_PREVIEW: one line on the top-level structure of a .json file longer than the preview shows."""
    try:
        import json
        if p.suffix.lower() != ".json" or p.stat().st_size > RICH_JSON_MAX:
            return None
        text = p.read_text(encoding="utf-8")
        if len(text.strip()) <= len(shown.strip()):
            return None
        data = json.loads(text)
    except Exception:   # unreadable or not JSON: the first lines stand alone, as before
        return None

    def kind(v):  # noqa: E306
        if isinstance(v, dict):
            return f"object with {len(v)} keys such as {', '.join(map(repr, list(v)[:3]))}" if v else "empty object"
        if isinstance(v, list):
            return f"list of {len(v)}"
        return type(v).__name__

    if isinstance(data, dict):
        return _clip("(top-level keys: " + "; ".join(f"{k!r}: {kind(v)}" for k, v in list(data.items())[:10]) + ")", 480)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _clip(f"(a list of {len(data)} objects with keys {', '.join(map(repr, list(data[0])[:12]))})", 480)
    return None


def _rich_block(p: Path, n: str, raw: bytes | None, first: str | None) -> str | None:
    """HARNESS_RICH_PREVIEW: the block added for one file (`raw`: the head of a file that is not text; `first`: the
    preview shown of a .json file), remembered per path, size and modification time: the prompt is built several
    times per task and the file does not change in between."""
    try:
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns, raw is None, first)
    except OSError:
        return None
    if key in _RICH_CACHE:
        return _RICH_CACHE[key]
    try:
        if raw is None:
            shape = _json_shape(p, first or "")
            block = None if shape is None else f"--- {n} ---\n{shape}"
        else:
            inside = _binary_preview(p, raw)
            block = None if inside is None else f"--- {n} ---\n{inside}"
    except Exception:
        block = None
    if len(_RICH_CACHE) >= 512:
        _RICH_CACHE.clear()
    _RICH_CACHE[key] = block
    return block


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
    later = []   # HARNESS_RICH_PREVIEW: what is added goes after every text preview, so it never takes the room of one
    for p, n in zip(files, names):
        try:
            with open(p, "rb") as f:
                raw = f.read(PREVIEW_FILE * 4)
        except OSError:
            continue
        if not _is_text(raw):
            if config.rich_preview():
                later.append((p, n, raw, None))
            continue
        text = raw.decode("utf-8", errors="ignore")
        first = "\n".join(text.splitlines()[:PREVIEW_LINES])
        first = first.encode("utf-8")[:PREVIEW_FILE].decode("utf-8", errors="ignore")
        block = f"--- {n} (first lines) ---\n{first}"
        if used + _nbytes(block) + 1 > PREVIEW_TOTAL:
            break
        parts.append(block)
        used += _nbytes(block) + 1
        if config.rich_preview() and p.suffix.lower() == ".json":
            later.append((p, n, None, first))
    for p, n, raw, first in later:
        block = _rich_block(p, n, raw, first)
        if block is None or used + _nbytes(block) + 1 > PREVIEW_TOTAL:
            continue
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


# the organizers' archive was packed on a Mac
_OS_JUNK = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep", ".keep", ".gitignore"}
_PAGE_SUFFIXES = {".md", ".markdown", ".txt", ""}


def _kb_junk(p: Path) -> bool:
    """An entry that says nothing about the base: OS and VCS litter or an empty dot-file. A README or an index is
    not junk: as a `# Title` page it is a page, with data (index.csv, a token in README.txt) the base is mixed."""
    n = p.name
    if n in _OS_JUNK or n.startswith("._"):
        return True
    try:
        return n.startswith(".") and p.stat().st_size == 0
    except OSError:
        return False


def _is_page(p: Path) -> bool:
    """A text file of a page kind whose first non-blank line is a `# Title`: what the preamble loads into `kb`.
    A hidden file is never a page (`.env`, `.secret`: the foreign bases hide data in them)."""
    if p.name.startswith(".") or not p.is_file() or p.suffix.lower() not in _PAGE_SUFFIXES:
        return False
    try:
        with open(p, "rb") as f:
            raw = f.read(PREVIEW_FILE * 4)
    except OSError:
        return False
    if b"\0" in raw:  # gz, sqlite: binary. Stray non-UTF-8 bytes in a page are read as the preamble reads them
        return False
    first = next((ln for ln in raw.decode("utf-8", errors="replace").lstrip("\ufeff").splitlines() if ln.strip()), "")
    return first.lstrip().startswith("#")


def _kb_is_plain(ws: Path) -> bool:
    """True when kb/ holds only pages the preamble loads: `# Title` text files (.md, .markdown, .txt or no suffix)
    right inside kb/, besides junk (`_kb_junk`). A subdirectory or a file of another kind (csv, json, log, gz,
    sqlite, a dot-file with data) is invisible to `kb`, and the model, shown only `kb`, answers IMPOSSIBLE; such a
    base goes to the generic program route with the file tree instead."""
    try:
        entries = list((ws / "kb").iterdir())
    except OSError:
        return True
    return all((p.is_file() and _kb_junk(p)) or _is_page(p) for p in entries)


def _is_kb_task(task: Task) -> bool:
    return task.task_type == "search" and (task.workspace / "kb").is_dir() and _kb_is_plain(task.workspace)


PROSE_SYSTEM = ("Answer a question from a knowledge base of short text pages (biographies, profiles). Reply with a "
                "short Python 3 program that prints only the final answer to stdout. "
                "Every page is ALREADY loaded into the dict `pages` (page title -> full text of the page): do not open "
                "files. The pages that matter most are shown below. "
                "A fact that a shown page states may be printed directly: copy the value exactly as the page writes it "
                "(a name, a year, a city), nothing else. A count, a comparison or anything over many pages is computed "
                "in code over pages.items() with `in`, str methods or re, using the wording you see in the shown pages. "
                "Follow chains yourself: when the answer needs a second page, read that page below. "
                "If the pages do not state the fact the question needs, print IMPOSSIBLE. "
                "Never write answer.txt or any file: the answer is read from the program's output. "
                "Output only code, no comments, no markdown.")
PROSE_CONTEXT_CHARS = 6000   # page text shown to the model (about 200 units of input)
PROSE_PAGE_CHARS = 1500      # of one page
_FIELD_LINE = re.compile(r"^[^\W\d_][\w /()-]{0,30}:[ \t]*\S")
_MD_LEAD = re.compile(r"^\s*(?:[-*+][ \t]+|#+[ \t]*)")   # a list bullet or a heading mark before a field
# a count or an aggregate over pages: computed in code, never a constant copied from a page. Only clear wordings:
# a count or a sum; a superlative that selects among entities ("the oldest person", "the most employees", "Who was
# born earliest?"); a comparative between named choices ("more employees, A or B?", "older than"). A single fact
# worded with the same words ("joined later in her career", "most of the time", "at least once") does not count
_PROSE_AGGREGATE_ASK = re.compile(
    r"\b(?:how many|number of|count|counts|total|sum|average|mean)\b"
    r"|\bthe (?:oldest|youngest|earliest|most|fewest|least|largest|smallest|biggest|highest|lowest|greatest)\b"
    r"|\b(?:oldest|youngest|earliest|latest|most|fewest|least)\s*\?"
    r"|\b(?:older|younger|earlier|later|more|fewer|less|larger|smaller|bigger|higher|lower)\b[^?.]*\b(?:than|or)\b",
    re.I)


def _kb_pages(ws: Path) -> dict[str, str]:
    """title -> full text of every page the preamble loads."""
    out: dict[str, str] = {}
    kb_dir = ws / "kb"
    try:
        files = sorted(p for p in kb_dir.iterdir() if p.is_file()) if kb_dir.is_dir() else []
    except OSError:
        files = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = next((ln.lstrip("#").strip() for ln in text.splitlines() if ln.startswith("#")), None)
        if title:
            out[title] = text
    return out


def _has_fields(text: str) -> bool:
    """The page holds at least one short `Key: value` line (the shape the `kb` dict is built from), also written
    as markdown (`- Born: 1970`, `**Born:** 1970`, `## Born: 1970`) or with no space after the colon."""
    lines = text.splitlines()
    title = next((i for i, ln in enumerate(lines) if ln.startswith("#")), None)
    for i, ln in enumerate(lines):
        if i == title:
            continue
        s = _MD_LEAD.sub("", ln, count=1).replace("**", "").replace("__", "")
        if _FIELD_LINE.match(s) and len(s.split()) <= 12:
            return True
    return False


def _kb_is_prose(ws: Path) -> bool:
    """Most pages are running text with no `Key: value` line: the `kb` dict of fields is empty for them, and a model
    shown only that dict answers IMPOSSIBLE (or 0 for a count). Such a base is read as text."""
    pages = _kb_pages(ws)
    return bool(pages) and sum(1 for t in pages.values() if _has_fields(t)) * 2 < len(pages)


def _prose_context(ws: Path, question: str) -> tuple[str, int]:
    """The page texts for the prompt, within PROSE_CONTEXT_CHARS: the pages the question names, then the pages those
    name and the pages naming them, then the rest in file order. Returns (text, number of pages in the base)."""
    pages = _kb_pages(ws)
    low = {t: t.casefold() for t in pages}
    q = question.casefold()
    first = [t for t in pages if low[t] in q]
    linked = [t for t in pages if t not in first and any(low[t] in pages[f].casefold() for f in first)]
    back = [t for t in pages if t not in first and t not in linked and any(low[f] in pages[t].casefold() for f in first)]
    seen = set(first) | set(linked) | set(back)
    parts, used = [], 0
    for t in first + linked + back + [t for t in pages if t not in seen]:
        full = pages[t].strip()
        body = full[:PROSE_PAGE_CHARS]
        if len(full) > len(body):   # a cut page that hides a fact must not read as a page without it
            body += (f"\n[... {len(full) - len(body)} more characters of this page are cut here; the full text is "
                     f"in pages[{t!r}]: read it in code before you print IMPOSSIBLE]")
        if used + len(body) > PROSE_CONTEXT_CHARS and parts:
            break
        parts.append(body)
        used += len(body) + 2
    return "\n\n".join(parts), len(pages)


def _quoted_from_kb(ws: Path, answer: str) -> bool:
    """The printed constant stands word for word on a page: a fact read from the text, not a number worked out
    in the head. Whole words only: `19` inside `1950` or `Robotics` inside `Dune Robotics` is not a match."""
    a = " ".join(answer.split()).casefold()
    if len(a) < 2:
        return False
    rx = re.compile(r"(?<![\w.])" + re.escape(a) + r"(?!\w)")
    return any(rx.search(" ".join(t.split()).casefold()) for t in _kb_pages(ws).values())


def _asks_aggregate(question: str) -> bool:
    """A count, a sum or a comparison over pages ("How many ...", "the oldest ..."): a constant printed for it was
    counted in the head from the shown pages, never read from one."""
    return bool(_PROSE_AGGREGATE_ASK.search(question))


# a relation chain over pages ("the employer of the mentor of X", "X's mentor's employer", "the employer of X's
# mentor"): a constant printed for it was followed across pages in the head (r16c search_8104: 1975, a whole word on
# another page, for 1973). One relation ("the mentor of X", "the company where X works", "the name of X's spouse",
# "the title of X's novel") is read from the shown pages
_PROSE_MULTIHOP_ASK = re.compile(
    r"\b[Tt]he (?!(?:name|year|date|city|country|place|title)\b)[a-z]+(?: [a-z]+)? of the [a-z]+(?: [a-z]+)? of\b"
    r"|\w['\u2019]s [a-z]+(?: [a-z]+)?['\u2019]s\b"
    r"|(?<!\bname )(?<!\btitle )\bof (?:the )?[A-Z][\w-]*(?: [A-Z][\w-]*)*['\u2019]s [a-z]+")


def _asks_multihop(question: str) -> bool:
    """A chain of two relations or more: like an aggregate, answered by a program over pages, never a printed constant."""
    return bool(_PROSE_MULTIHOP_ASK.search(question))


def _is_prose_task(task: Task) -> bool:
    return _is_kb_task(task) and _kb_is_prose(task.workspace)


MIXED_KB_HINT = (" The answer may be in any file listed below, of any format, also in subdirectories and "
                 "hidden files: read the files that matter with open() / os.walk('kb').")
FREE_OUTPUT_HINT = " The program's whole stdout is the answer: print exactly what the task asks, nothing else."


_NAMES_TESTS = re.compile(r"mylib|test_|unittest|pytest")


def is_free_coding(task: Task) -> bool:
    """A coding task with nothing to repair: no mylib.py, no test file, and the text names neither. Today it
    ends as `unparsed` with no call at all; the program route can answer it (the text asks for an output)."""
    if task.task_type != "coding" or task.workspace is None:
        return False
    ws = Path(task.workspace)
    if (ws / "mylib.py").exists() or (task.check or "") == "unittest" or _NAMES_TESTS.search(task.instruction or ""):
        return False
    try:  # a test file anywhere (tests/test_x.py, x_test.py) makes it a repair task of another layout
        return not any(p.is_file() for pat in ("test*.py", "*_test.py") for p in ws.rglob(pat))
    except OSError:
        return False


def _program_messages(task: Task) -> list[dict]:
    if _is_prose_task(task):
        question = strip_boilerplate(task.instruction)
        shown, n = _prose_context(task.workspace, question)
        user = f"{question}\n\n`pages` has {n} pages. Pages (the most relevant first):\n\n{shown}"
        return [{"role": "system", "content": PROSE_SYSTEM}, {"role": "user", "content": user}]
    if _is_kb_task(task):
        ex, n = _kb_examples(task.workspace)
        user = (f"{strip_boilerplate(task.instruction)}\n\nkb has {n} pages, for example:\n" + "\n".join(ex))
        return [{"role": "system", "content": SEARCH_SYSTEM}, {"role": "user", "content": user}]
    hint = ""
    if task.task_type == "search" and (task.workspace / "kb").is_dir():
        hint = MIXED_KB_HINT
    elif is_free_coding(task):
        hint = FREE_OUTPUT_HINT
    user = f"{task.instruction}{hint}\n\n{describe_workspace(task.workspace)}"
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
    prose = use_kb and _kb_is_prose(task.workspace)
    # a search answer is one value on any route: None / unknown / a whole dict is never it, nor a best guess
    guard = use_kb or task.task_type == "search"
    free_text = task.check != "exact_int" and is_free_coding(task)  # the whole output, not its last line
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
        quoted = False
        question = strip_boilerplate(task.instruction) if prose else ""
        if prose and is_bare_literal(code) and not _asks_aggregate(question) and not _asks_multihop(question):
            # a fact copied from a shown page: taken when a page really holds it
            probe = await asyncio.to_thread(run_program, kb_program(code), task.workspace)
            said = normalize_answer(probe.stdout, task.check, task.instruction) if probe.returncode == 0 else None
            quoted = said is not None and _quoted_from_kb(task.workspace, said)
        if not literal_retried and not quoted and is_bare_literal(code):  # a number worked out in the head: one more try, no run
            literal_retried, attempts = True, attempts + 1
            note = "program printed a hard-coded literal"
            messages = messages + [{"role": "assistant", "content": r.text}, {"role": "user", "content":
                "Your program prints a constant that no code computed. Compute it in code from the values in the "
                "task and the files, one statement per sentence of the task. Print the final answer with print(); "
                "do not write it to a file. Output only code."}]
            attempt += 1
            continue
        # subprocess + copytree block for up to 20 s: keep them off the event loop
        res = await call_tool("run_program", run_program, kb_program(code) if use_kb else code, task.workspace)
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
        if answer is not None and free_text and len(lines) > 1:  # a list or a table: every line is the answer
            answer = "\n".join(ln.rstrip() for ln in res.stdout.strip().splitlines())
        ph = _kb_placeholder(res, answer) if guard else None
        if ph == "clean":  # same as KbMissing, also when the shape check would only guess
            return SolveResult("unparsed", note="kb page or field not found" if use_kb
                               else "program printed a placeholder")
        container = guard and (ph == "mixed" or (answer is not None and _is_container_repr(answer)))
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


# notes of solve_with_program for a FAST verdict "this cannot be answered" that comes from the model's own program
_IMPOSSIBLE_NOTES = ("program printed IMPOSSIBLE", "kb page or field not found")


def first_impossible(res: SolveResult) -> bool:
    """`solve_with_program` ended in the model's own verdict that the task cannot be answered: its program printed
    IMPOSSIBLE, or asked the knowledge base for a page or field that is not there. No guess came with it."""
    return (res.status in ("infeasible", "unparsed") and res.note in _IMPOSSIBLE_NOTES
            and (res.extra or {}).get("best_guess") is None)


# a printed refusal in other words than IMPOSSIBLE ("Cannot be determined", "IMPOSSIBLE: no data")
_REFUSAL = re.compile(r"\b(?:impossible|cannot|can't|can not|not enough|insufficient|undetermin\w*|"
                      r"not (?:be )?determin\w*|no (?:such|data|information|answer))\b", re.I)


async def second_opinion(task: Task, client, first: SolveResult) -> SolveResult:
    """HARNESS_SECOND_IMPOSSIBLE (task 86): after `first_impossible`, one MODEL_ALT program on the same prompt. A
    well-formed answer (it ran, printed a value of the expected shape computed in code: not a hard-coded literal, not
    IMPOSSIBLE or a refusal in other words, a placeholder or a whole page) comes back as a best guess: an answer to an
    infeasible task costs nothing and a wrong one is worth no answer, while the final REASON pass may still take the
    task. The guess only fills an empty slot (`fill_only`): it never replaces a guess stored before. Anything else
    returns `first` unchanged."""
    try:
        r = await client.chat(MODEL_ALT, _program_messages(task), PROGRAM_MAX_TOKENS)
    except LLMUnavailable:
        return first
    if getattr(r, "truncated", False):
        return first
    use_kb = _is_kb_task(task)
    code = extract_code(r.text, join_all=False)
    if is_bare_literal(code):   # a value worked out in the head: FAST's route never takes one as it is either
        return first
    res = await call_tool("run_program", run_program, kb_program(code) if use_kb else code, task.workspace)
    if res.returncode != 0 or res.timed_out:
        return first
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if not lines or lines[-1] == "IMPOSSIBLE":
        return first
    answer = normalize_answer(res.stdout, task.check, task.instruction)
    if answer is None or answer == "IMPOSSIBLE" or _is_placeholder(answer) or _kb_placeholder(res, answer) \
            or _is_container_repr(answer) or mismatch(task, answer) is not None or _REFUSAL.search(lines[-1]) \
            or _REFUSAL.search(answer):
        return first
    return SolveResult("unparsed", note=f"{first.note}; {MODEL_ALT} answered",
                       extra={"best_guess": answer, "fill_only": True})


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
        # `literal`: no code computed it (the prose agreement gives it no vote on a count or a chain; task 79)
        return SolveResult("solved", answer=bare, note="model reason answer", extra={"literal": True})
    use_kb = _is_kb_task(task)
    code = extract_code(r.text, join_all=False)
    res = await call_tool("run_program", run_program, kb_program(code) if use_kb else code, task.workspace)
    if use_kb and res.returncode != 0 and not res.timed_out and _kb_key_error(res.stderr):
        return SolveResult("unparsed", note="kb page or field not found")
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if res.returncode == 0 and not res.timed_out and lines and lines[-1] == "IMPOSSIBLE":
        return SolveResult("infeasible", note="reason program printed IMPOSSIBLE")
    answer = None if res.returncode != 0 or res.timed_out else normalize_answer(res.stdout, task.check, task.instruction)
    guard = use_kb or task.task_type == "search"
    ph = _kb_placeholder(res, answer) if guard else None
    if ph == "clean":
        return SolveResult("unparsed", note="kb page or field not found" if use_kb
                           else "reason program printed a placeholder")
    if ph == "mixed":
        return SolveResult("unparsed", note="reason program printed a placeholder after other output")
    if answer is not None and guard and _is_container_repr(answer):
        return SolveResult("unparsed", note="reason program printed a whole page or list")
    if answer is None:
        return SolveResult("unparsed", note="reason program gave no usable answer")
    bad = mismatch(task, answer)
    if bad is not None:
        return SolveResult("unparsed", note=f"answer shape: {bad}",
                           extra={"best_guess": answer, "shape_mismatch": True})
    return SolveResult("solved", answer=answer, note="model reason program",
                       extra={"literal": True} if is_bare_literal(code) else None)


# ---------------------------------------------------------------- prose agreement (round 17, task 79)

def _prose_vote(res: SolveResult | None) -> str | None:
    """The answer an attempt gives: solved, of the right shape (both routes check it). A crash, a timeout,
    IMPOSSIBLE, a placeholder or a wrong shape is no answer, not a vote."""
    if res is None or res.status != "solved" or res.answer is None:
        return None
    a = str(res.answer).strip()
    return a if a and a != "IMPOSSIBLE" else None


def _program_failed(res: SolveResult) -> bool:
    """FAST's attempt gave no answer that another model could not change: a program that failed (exit code, timeout,
    wrong shape, no output, a placeholder, a route that raised), a page or field the program looked up and did not
    find (on prose that is a wrong title or key in the model's code, not proof: ea064d3 in Docker, search_8104 ended
    on KbMissing after two FAST calls), or IMPOSSIBLE. Not the proxy (the retry pass runs the unit again) and not
    the budget cutting the call."""
    note = res.note or ""
    if res.status == "infeasible":
        return True
    return (res.status == "unparsed" and not note.startswith(LLM_UNAVAILABLE)
            and "cut by the budget limit" not in note)


LLM_UNAVAILABLE = "llm unavailable"   # note prefix of a result whose model call never got an answer


async def _prose_attempt(route, task: Task) -> tuple[SolveResult | None, set]:
    """One attempt (`route()`): its result, None when it crashed, and the weights that answered its calls."""
    from harness.confirm import family
    from harness.llm import BudgetExhausted
    calls = CALLS.get()
    start = len(calls) if calls is not None else 0
    try:
        res = await route()
    except BudgetExhausted:
        raise
    except Exception as e:  # the attempt is lost, the answers so far are not
        log.warning("prose attempt on %s failed: %r", task.task_id, e)
        res = None
    used = {family(c.get("model")) for c in (calls[start:] if calls is not None else []) if not c.get("error")}
    return res, used


_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})


def _vote_key(answer: str) -> str:
    """A text answer as the soft check reads it: no case, no extra spaces, no quotes around it, no closing
    punctuation ("Arktika Labs." = "arktika labs"), one apostrophe (U+2019 = "'")."""
    s = " ".join(str(answer).split()).translate(_APOSTROPHES).strip("\"' ")
    return s.rstrip(".,;:!").strip().casefold()


def _same_vote(a: str, b: str, check: str | None) -> bool:
    """Two attempts give the same answer: numbers as numbers (exact_int: 18 = 18.0), text by `_vote_key`."""
    from harness.confirm import same_answer
    if same_answer(a, b):
        return True
    return check != "exact_int" and _vote_key(a) == _vote_key(b) != ""


# a last word whose dot belongs to it: a known abbreviation (Jr., Inc., Co., St.) or two initials or more (U.S.)
_ABBREV_END = re.compile(r"(?:^|[\s(])(?:Jr|Sr|Inc|Co|Ltd|Corp|St|Dr|Bros|Mfg|Plc|Esq|Ave)\.$|(?:^|\s)(?:[A-Za-z]\.){2,}$")


def _is_clean(answer: str) -> bool:
    """The text is its own soft-check form, case aside: no extra spaces, quotes around it, closing punctuation or
    curly apostrophe; a whole number written as a plain integer ("1999", not "1999.0"). Other numbers count as clean
    (their places may be the ones the task asks for). The closing dot of an abbreviation is part of the answer:
    "John Smith Jr." is clean and "John Smith Jr" is not, while "Arktika Labs." is not."""
    s = str(answer)
    try:
        d = Decimal(s.strip())
        if d.is_finite():
            whole = d.to_integral_value()
            return d != whole or s == format(whole, "f")
    except (InvalidOperation, ValueError):
        pass
    core = " ".join(s.split()).translate(_APOSTROPHES).strip("\"' ").rstrip(".,;:!").strip()
    if core and _ABBREV_END.search(core + "."):
        core += "."
    return s == core


def _cleanest(*results: SolveResult) -> SolveResult:
    """Of agreeing attempts (in order of preference), the first whose answer is clean (`_is_clean`), else the first:
    answers equal only after normalisation are written in the form the checker reads as they are."""
    return next((r for r in results if _is_clean(r.answer)), results[0])


def _agreed(res: SolveResult, note: str, reasoned: bool) -> SolveResult:
    extra = {**(res.extra or {}), "reasoned": True} if reasoned else res.extra
    return SolveResult(res.status, answer=res.answer, note=note, extra=extra)


async def prose_agreement(task: Task, client, first: SolveResult, may_call, keep=None) -> SolveResult:
    """A prose answer stands when two independent attempts give it (round 17, task 79). `first` is the FAST program
    route's result (`solve_with_program`). A MODEL_ALT program on the same prompt follows; equal answers (`_same_vote`:
    numbers as numbers, text as the soft check reads it) settle it. Different, or one gave
    nothing: one MODEL_REASON program (`solve_with_reason`) is the third, and an answer two attempts share wins;
    without one, REASON's answer, else the first, else ALT's (a wrong answer scores the same as none). An ALT
    attempt answered by FAST's weights (ALT failing, `chat_or_alt`) is no vote. `may_call(reason)` is the
    scheduler's gate (ledger, spend share, time) before the ALT (False) and the REASON (True) attempt: when it
    says no, the answers so far stand. `extra["reasoned"]`: REASON gave its verdict here, the final pass skips the
    task. A first attempt without an answer (`_program_failed`: a failed program, a page or field not found,
    IMPOSSIBLE, a route that raised) goes on to ALT and REASON; only the proxy and the budget cut end it here. FAST
    and ALT both printing IMPOSSIBLE agree: no REASON call.
    REASON's answer is no vote when no code computed it (a lone number, a printed constant) and the question is a
    count, a comparison or a chain (`_asks_aggregate`, `_asks_multihop`): the r16c search_8104 failure. `keep(res)`
    (the scheduler's settle) writes ALT's answer before the REASON attempt when FAST had none (FAST's own answer is
    on disk already), so a deadline in that call keeps it."""
    from harness.confirm import family
    same = lambda a, b: _same_vote(a, b, task.check)  # noqa: E731
    fast = _prose_vote(first)
    if fast is None and not _program_failed(first):
        return first
    head = f"{first.note}; agree:"
    if not may_call(False):
        return _agreed(first, f"{head} no budget for a second attempt", False)
    alt_res, alt_used = await _prose_attempt(
        lambda: solve_with_program(task, client, model=MODEL_ALT, shape_model=MODEL_ALT), task)
    alt = _prose_vote(alt_res)
    alt_say = repr(alt)
    if alt is not None and family(MODEL_FAST) in alt_used:
        alt, alt_say = None, f"{alt!r} (answered by {MODEL_FAST}, no vote)"
    if fast is not None and alt is not None and same(fast, alt):
        return _agreed(_cleanest(first, alt_res), f"{head} fast and alt {fast!r}", False)
    if (first.status == "infeasible" and alt_res is not None and alt_res.status == "infeasible"
            and family(MODEL_FAST) not in alt_used):
        return _agreed(first, f"{head} fast and alt IMPOSSIBLE", False)   # two agreeing attempts: no REASON call
    head = f"{head} fast {fast!r}, alt {alt_say}"
    if fast is None and alt is not None and keep is not None:
        keep(alt_res)
    if not may_call(True):
        if fast is None and alt is not None:
            return _agreed(alt_res, f"{head}; no budget for {MODEL_REASON}: alt's answer", False)
        return _agreed(first, f"{head}; no budget for {MODEL_REASON}: first kept", False)
    rsn_res, _ = await _prose_attempt(lambda: solve_with_reason(task, client), task)
    rsn = _prose_vote(rsn_res)
    rsn_say = repr(rsn)
    dropped = None   # REASON's constant for a count or a chain: no vote, only a guess when nothing else answers
    if rsn is not None and (rsn_res.extra or {}).get("literal"):
        question = strip_boilerplate(task.instruction)
        if _asks_aggregate(question) or _asks_multihop(question):
            dropped, rsn, rsn_say = rsn, None, f"{rsn!r} (a constant for a count or a chain, no vote)"
    rnote = (rsn_res.note or "") if rsn_res is not None else ""
    reasoned = rsn_res is not None and not rnote.startswith("llm unavailable") and "cut by the budget" not in rnote
    head = f"{head}, reason {rsn_say}"
    if rsn is not None and fast is not None and same(rsn, fast):
        return _agreed(_cleanest(first, rsn_res), f"{head}: fast and reason agree", reasoned)
    if rsn is not None and alt is not None and same(rsn, alt):
        return _agreed(_cleanest(rsn_res, alt_res), f"{head}: alt and reason agree", reasoned)
    if rsn is not None:
        return _agreed(rsn_res, f"{head}: no majority, reason's answer", reasoned)
    if fast is not None:
        return _agreed(first, f"{head}: no majority, first kept", reasoned)
    if alt is not None:
        return _agreed(alt_res, f"{head}: alt's answer, the only one", reasoned)
    if dropped is not None:
        # a wrong answer scores the same as none: REASON's constant goes to disk as a guess (put_guess), the task
        # stays open, and REASON's reply is no verdict, so the final pass may still ask it
        return SolveResult("unparsed", note=f"{head}: no answer; reason's constant kept as a guess",
                           extra={"best_guess": dropped})
    # nothing answered: REASON's IMPOSSIBLE is its verdict (the final pass skips the task); else the first result,
    # whose best guess, if it has one, the scheduler stores
    last = (rsn_res if rsn_res is not None and rsn_res.status == "infeasible"
            and (first.extra or {}).get("best_guess") is None else first)
    return _agreed(last, f"{head}: no answer", reasoned)


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

    try:
        body = ast.parse(src).body
    except SyntaxError:
        return _locate_text(src)
    for node in body:
        if isinstance(node, _DEFS):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, _DEFS):
                    add(sub, f"{node.name}.{sub.name}")
    return out


_TOP_DEF = re.compile(r"(?:async[ \t]+)?def[ \t]+(\w+)")


def _locate_text(src: str) -> dict[str, tuple[int, int, int]]:
    """Top-level functions of a file that does not parse (a missing colon, a bad indent): by text. A function
    runs from its `def` line, or from the `@` decorator lines right above it (as in `_locate`), to the next line
    that starts at column 0 (blank lines and comments before it stay outside). A wrong span cannot be written: the patched file must parse (apply_patch) and pass the visible
    tests (fix_coding)."""
    lines = src.split("\n")
    tops = [i for i, ln in enumerate(lines) if ln[:1] not in ("", " ", "\t", "#", ")", "]", "}")]
    out: dict[str, tuple[int, int, int]] = {}
    for n, i in enumerate(tops):
        m = _TOP_DEF.match(lines[i])
        if not m:
            continue
        stop = tops[n + 1] if n + 1 < len(tops) else len(lines)
        while stop > i + 1 and (not lines[stop - 1].strip() or lines[stop - 1].startswith("#")):
            stop -= 1
        first = n
        while first > 0 and lines[tops[first - 1]].startswith("@"):  # a reply repeating them replaces them
            first -= 1
        out.setdefault(m.group(1), (tops[first], stop, 0))
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


# directories a test file below the root is never looked for in (besides hidden ones: `.git`, `.venv`)
_DEEP_SKIP = frozenset({"__pycache__", "node_modules", "site-packages", "venv", "env", "build", "dist",
                        # data and samples, not the task's tests (`fixtures/test_input.py`, `examples/test_demo.py`)
                        "examples", "example", "docs", "doc", "fixtures", "data", "samples"})
DEEP_MAX_FILES = 8   # test files below the root run per candidate; more are left out (each run takes a process)


def _is_test_module(ws: Path, p: Path) -> bool:
    """A file below the root is a test file when it imports unittest, pytest or a module of the workspace and
    defines tests: a top-level `test*` function, a class with `test*` methods (`Test*` or a TestCase), or top-level
    asserts of a script. A data module named like a test (`test_input.py` of cases) is not one."""
    try:
        tree = ast.parse(p.read_bytes())
    except (OSError, SyntaxError, ValueError, RecursionError):
        return False
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    defines = any(
        isinstance(n, funcs) and n.name.startswith("test")
        or isinstance(n, ast.ClassDef) and any(isinstance(m, funcs) and m.name.startswith("test") for m in n.body)
        and (n.name.startswith("Test") or any(getattr(b, "attr", getattr(b, "id", "")).endswith("TestCase")
                                              for b in n.bases))
        or isinstance(n, ast.Assert) for n in tree.body)
    if not defines:
        return False
    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
        for m in names:
            top = m.split(".")[0]
            if top in ("unittest", "pytest") or _module_file(ws, m) is not None or _module_file(ws, top) is not None:
                return True
    return False


def _deep_tests(ws: Path) -> list[str]:
    """HARNESS_DEEP_TESTS: the test files below the workspace root (`tests/test_x.py`, `test/...`,
    `tests/unit/x_test.py`) as relative posix paths, the shallowest first, when the root holds none. [] when the
    root has a test file (its layout keeps round 18's discovery) or the switch is off."""
    if not config.deep_tests():
        return []
    try:
        if any(p.is_file() for pat in ("test_*.py", "*_test.py") for p in ws.glob(pat)):
            return []
        found = set()
        for pat in ("test_*.py", "*_test.py"):
            for p in ws.rglob(pat):
                rel = p.relative_to(ws)
                if len(rel.parts) < 2 or not p.is_file():
                    continue
                if any(part.startswith(".") or part in _DEEP_SKIP for part in rel.parts[:-1]):
                    continue
                if _is_test_module(ws, p):
                    found.add(rel.as_posix())
    except (OSError, ValueError):
        return []
    return sorted(found, key=lambda r: (r.count("/"), r))[:DEEP_MAX_FILES]


def _test_file(ws: Path) -> str:
    """Same discovery as phase 1: test_mylib.py, else the first test_*.py (then the first *_test.py). A root without
    one: the first test file below it (`_deep_tests`)."""
    if (ws / "test_mylib.py").is_file():
        return "test_mylib.py"
    cands = sorted(p.name for p in ws.glob("test_*.py")) or sorted(p.name for p in ws.glob("*_test.py"))
    if not cands:
        cands = _deep_tests(ws)
    return cands[0] if cands else "test_mylib.py"


def _read_tests(ws: Path) -> tuple[str, str]:
    """(name, text) of the tests the ladder shows: the test file (`_test_file`); several files below the root as one
    text, each under a `# <path>` line, named by their paths."""
    deep = _deep_tests(ws)
    if len(deep) > 1:
        texts = [f"# {n}\n{(ws / n).read_text(encoding='utf-8')}" for n in deep]
        return ", ".join(deep), "\n\n".join(texts)
    name = _test_file(ws)
    return name, (ws / name).read_text(encoding="utf-8")


def _module_file(ws: Path, module: str) -> str | None:
    """The workspace file of a dotted module (`pkg.core` -> `pkg/core.py` or `pkg/core/__init__.py`), from the root or
    from `src/`; never a test file nor a helper of the tests (`tests/helpers.py`)."""
    parts = module.split(".")
    if not all(re.fullmatch(r"[A-Za-z_]\w*", q) for q in parts) or parts[0] in ("tests", "test"):
        return None
    for base in (ws, ws / "src"):
        for p in (base.joinpath(*parts[:-1], parts[-1] + ".py"), base.joinpath(*parts, "__init__.py")):
            if p.is_file() and not p.name.startswith("test_") and not p.name.endswith("_test.py"):
                return p.relative_to(ws).as_posix()
    return None


def _defined(tree: ast.Module) -> set:
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _package_target(ws: Path) -> str | None:
    """HARNESS_DEEP_TESTS: the file under repair when the tests import a module below the root (`pkg/core.py` for
    `from pkg.core import f`, `import pkg.core as c` or `from pkg import core`; below `src/` too) or a package whose
    `__init__.py` takes its names from its own modules (`from .core import f`). Of the files reached, the one that
    defines most of the names the tests use (imported names, attributes); a tie goes to the first reached that is not
    an `__init__.py`. None when no reached file defines any of them."""
    if not config.deep_tests():
        return None
    reached: list[str] = []
    used: set = set()

    def reach(rel):
        if rel is not None and rel not in reached:
            reached.append(rel)

    try:
        for name in (_deep_tests(ws) or [_test_file(ws)]):
            try:
                tree = ast.parse((ws / name).read_bytes())
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):   # `import pkg.core as c`, `import pkg` (pkg/__init__.py)
                    for a in node.names:
                        reach(_module_file(ws, a.name))
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    for a in node.names:
                        sub = _module_file(ws, f"{node.module}.{a.name}") if a.name != "*" else None
                        if sub is not None:
                            reach(sub)
                        else:
                            used.add(a.name)
                            reach(_module_file(ws, node.module))
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    used.add(node.func.id)   # `from pkg import *`: the names come only from the calls
        # a package's __init__.py that takes its names from its own modules leads to them
        for rel in list(reached):
            if not rel.endswith("/__init__.py"):
                continue
            pkg = Path(rel).parent.parts
            if pkg and pkg[0] == "src":
                pkg = pkg[1:]
            for node in ast.walk(ast.parse((ws / rel).read_bytes())):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    for m in ([node.module] if node.module else [a.name for a in node.names]):
                        reach(_module_file(ws, ".".join(pkg + tuple(m.split(".")))))
        scored = []
        for i, rel in enumerate(reached):
            n = len(_defined(ast.parse((ws / rel).read_bytes())) & used)
            if n:
                scored.append((-n, rel.endswith("__init__.py"), i, rel))
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    return min(scored)[3] if scored else None


def _target_file(ws: Path, tests: str) -> str:
    """The file under repair: mylib.py; in a workspace without one, the first module of the workspace that the
    test file imports (`inventory.py` for `from inventory import ...`), else the only other .py file."""
    if (ws / "mylib.py").is_file():
        return "mylib.py"
    try:
        tree = ast.parse(tests)
    except (SyntaxError, ValueError):
        tree = None
    for node in (ast.walk(tree) if tree is not None else ()):
        mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
        for m in mods:
            name = m.split(".")[0] + ".py"
            if (ws / name).is_file():
                return name
    deep = _package_target(ws)
    if deep is not None:
        return deep
    try:
        others = sorted(p.name for p in ws.glob("*.py") if not p.name.startswith("test_")
                        and not p.name.endswith("_test.py"))
    except OSError:
        others = []
    return others[0] if len(others) == 1 else "mylib.py"


def _run_all(directory: Path, names: list[str]) -> TestReport:
    """Every test file of `names` run in `directory`: green when all are; the failures of all, the output of the red."""
    reports = [(n, run_in_dir(directory, n)) for n in names]
    if all(r.ok for _, r in reports):
        return TestReport(True, [], "\n".join(r.output for _, r in reports))
    failed = [f for _, r in reports for f in r.failed]
    return TestReport(False, failed, "\n".join(f"== {n}\n{r.output}" for n, r in reports if not r.ok))


def _run_candidate(ws: Path, mylib_text: str, target: str = "mylib.py") -> TestReport:
    tmp = Path(tempfile.mkdtemp(prefix="llm_fix_"))
    try:
        cand = tmp / "ws"
        shutil.copytree(ws, cand, ignore=shutil.ignore_patterns("__pycache__"))
        (cand / target).write_text(mylib_text, encoding="utf-8", newline="\n")
        deep = _deep_tests(ws)
        if len(deep) > 1:   # several test files below the root: all of them
            return _run_all(cand, deep)
        return run_in_dir(cand, _test_file(ws))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cap(text: str, limit: int = FILE_CAP) -> str:
    if len(text) <= limit:
        return text
    h = limit // 2
    return f"{text[:h]}\n...[{len(text) - limit} chars cut]...\n{text[-h:]}"


def _patch_messages(mylib: str, tests: str, output: str, target: str = "mylib.py",
                    test_name: str = "test_mylib.py") -> list[dict]:
    """With mylib.py the text is the round-10 text. A file under another name (`_target_file`) is shown under its
    own name, in the system line as well."""
    if target == "mylib.py":
        test_name = "test_mylib.py"
    user = (f"{target}:\n{_cap(mylib)}\n\n{test_name}:\n{_cap(tests)}\n\n"
            f"Output of the failing tests:\n{output[-TEST_OUTPUT_LIMIT:]}\n\n"
            "Return only the corrected functions in the FUNC format.")
    return [{"role": "system", "content": PATCH_SYSTEM.replace("mylib.py", target)},
            {"role": "user", "content": user}]


TIMEOUT_HINT = ("(the test run was stopped by the time limit and printed nothing more: a function never returns or "
                "is far too slow. Find it by reading the code, make it finish, and fast)")


def _shown_output(report: TestReport) -> str:
    """What the model is told of a test run. A run killed by the time limit prints nothing that says so: without
    the note the model sees no failure of the function that hangs and repairs only the others."""
    if "<timeout>" in report.failed:
        return report.output + "\n" + TIMEOUT_HINT
    return report.output


def _budget_is_free(client) -> bool:
    """More than half of the budget left: the reasoning step is affordable for any reward."""
    ledger = getattr(client, "ledger", None)
    try:
        return ledger is not None and ledger.remaining() > 0.5 * ledger.budget
    except Exception:
        return False  # fake or foreign clients: keep the old reward threshold


def _restored(src: str, original: str, name: str) -> str | None:
    """`src` with the definition `name` (a top-level function or a method, as in `_locate`) given back its text in
    `original`. None when `original` has no such definition, the texts are the same, the columns differ or the
    result does not parse."""
    here, there = _locate(src), _locate(original)
    if name not in here or name not in there or here[name][2] != there[name][2]:
        return None
    lines, old_lines = io.StringIO(src).readlines(), io.StringIO(original).readlines()
    (s0, e0, _), (s1, e1, _) = here[name], there[name]
    old = "".join(old_lines[s1:e1])
    if "".join(lines[s0:e0]) == old:
        return None
    if not old.endswith("\n"):
        old += "\n"
    lines[s0:e0] = [old]
    out = "".join(lines)
    try:
        ast.parse(out)
    except (SyntaxError, ValueError, RecursionError):
        return None
    return out


LINE_SIMILAR = 0.5   # a line of a rewritten block is given back the most similar original line of it from this ratio


def _simple(line: str) -> ast.stmt | None:
    """The one simple statement a stripped line holds (no `if`/`for` header, no block); None otherwise."""
    try:
        body = ast.parse(line.strip()).body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(body) != 1 or isinstance(body[0], (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
                                              ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    return body[0]


def _own_call(line: str, names: set) -> bool:
    """The line is a bare call of a function or method the file defines (`other.deposit(amount)`, `log(x)`)."""
    st = _simple(line)
    if not isinstance(st, ast.Expr) or not isinstance(st.value, ast.Call):
        return False
    f = st.value.func
    return (f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None) in names


_CONTAINER_METHODS = frozenset(dir(list)) | frozenset(dir(dict)) | frozenset(dir(set)) | frozenset(dir(str))


def _callee(f: ast.expr) -> str | None:
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _inlined(src: str, original: str, name: str, new_line: str, old_line: str) -> bool:
    """The model's `new_line` stands in for the own call `old_line` by inlining it, and giving the call back adds
    nothing twice. Not when the model's line calls the callee itself (an argument fix: `notify(log, order_id, who)`);
    for a method call, not when the receiver is no plain name (`self.tags.remove(t)` is a set's method), not when a
    receiver other than `self` gets a list/dict/set/str method name, and not when a line of the callee's body that
    acts on `self.` already stands, on the receiver, elsewhere in the model's definition (a faithful inline:
    `other.history.append(...)` written out next to `other.balance += amount`)."""
    try:
        f = ast.parse(old_line.strip()).body[0].value.func
        callee = _callee(f)
        if any(isinstance(n, ast.Call) and _callee(n.func) == callee for n in ast.walk(ast.parse(new_line.strip()))):
            return False
        if not isinstance(f, ast.Attribute):
            return True
        if not isinstance(f.value, ast.Name):
            return False
        recv = f.value.id
        if recv != "self" and callee in _CONTAINER_METHODS:
            return False
        s0, e0, _ = _locate(src)[name]
        mine = {ln.strip() for ln in io.StringIO(src).readlines()[s0:e0]} - {new_line.strip()}
        old_lines = io.StringIO(original).readlines()
        for q, (s, e, _) in _locate(original).items():
            if q.rsplit(".", 1)[-1] != callee:
                continue
            for ln in old_lines[s + 1:e]:
                st = ln.strip()
                if "self." in st and st.replace("self.", recv + ".") in mine:
                    return False
        return True
    except (SyntaxError, ValueError, RecursionError, AttributeError, IndexError, KeyError):
        return False


def _line_restores(src: str, original: str, name: str) -> list[tuple[tuple, str]]:
    """(key, `src` with one line of the definition `name` given back an original line) for each line of a block the
    model rewrote (a `replace` of difflib) that is a simple statement standing where the original called a function
    or method of its own file: the most similar such original line of that block, of the same indent and a
    similarity of at least LINE_SIMILAR (`other.balance += amount` -> `other.deposit(amount)`: an inlined call drops
    what the callee does besides). Never a condition or any other line: a mix of the model's block and the old
    condition (`if left <= 0:` -> `if left < 0:`, deep19 coding_9707) is neither text and can hide a bug the visible
    tests do not catch. In order of the lines; only the ones that parse."""
    here, there = _locate(src), _locate(original)
    if name not in here or name not in there or here[name][2] != there[name][2]:
        return []
    own = {q.rsplit(".", 1)[-1] for q in there}
    lines, old_lines = io.StringIO(src).readlines(), io.StringIO(original).readlines()
    (s0, e0, _), (s1, e1, _) = here[name], there[name]
    new, old = lines[s0:e0], old_lines[s1:e1]
    indent = lambda ln: ln[:len(ln) - len(ln.lstrip())]   # noqa: E731
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        for j in range(j1, j2):
            if _simple(new[j]) is None:
                continue
            cands = [(difflib.SequenceMatcher(None, old[k].strip(), new[j].strip()).ratio(), -k) for k in range(i1, i2)
                     if indent(old[k]) == indent(new[j]) and _own_call(old[k], own)]
            if not cands or max(cands)[0] < LINE_SIMILAR:
                continue
            k = -max(cands)[1]
            if not _inlined(src, original, name, new[j], old[k]):
                continue
            text = old[k] if old[k].endswith("\n") else old[k] + "\n"
            cand = "".join(lines[:s0 + j] + [text] + lines[s0 + j + 1:])
            try:
                ast.parse(cand)
            except (SyntaxError, ValueError, RecursionError):
                continue
            out.append(((name, new[j], text), cand))
    return out


async def _cut_back(ws: Path, original: str | None, fixed: str, targs: tuple = ()) -> tuple[str, list[str]]:
    """HARNESS_MINIMAL_PATCH (round 20, task 88): `fixed` is green on the visible tests; `original` is the task's
    file before phase 1 and the ladder. With HARNESS_MINIMAL_DEFS (off by default) every function or method whose text
    differs takes its original text back first, one at a time in file order, when the visible tests stay green
    (passes repeat while one restore is kept). Then the line pass (`_line_restores`): a line where the model inlined a
    call of the file's own function takes the call back when the tests stay green (far8 coding_8001:
    `other.deposit(amount)` for `other.balance += amount`, which lost b's history entry). Definitions the model added
    stay. Returns (source, restored names); never raises: out of time
    (`MINIMAL_PATCH_S`) or on any failure, what is done so far."""
    done: list[str] = []
    if not config.minimal_patch() or not original or original == fixed:
        return fixed, done
    deadline = time.monotonic() + config.MINIMAL_PATCH_S
    try:
        names = [n for n, _ in sorted(_locate(fixed).items(), key=lambda kv: kv[1][0])]
        tried: set = set()   # (name, source) pairs already run: a red restore is not run again on the same file
        progress = config.minimal_defs()   # HARNESS_MINIMAL_DEFS: the whole-definition pass, off by default
        while progress:
            progress = False
            for name in names:
                if name in done:
                    continue
                cand = _restored(fixed, original, name)
                if cand is None or (name, fixed) in tried:
                    continue
                if time.monotonic() > deadline:
                    return fixed, done
                tried.add((name, fixed))
                if (await call_tool("run_tests", _run_candidate, ws, cand, *targs)).ok:
                    fixed = cand
                    done.append(name)
                    progress = True
        # a definition the tests need changed stays the model's: one line of a block it rewrote at a time takes
        # the most similar original line back (far8 coding_8001: `transfer` needs no `withdraw` entry, but
        # `other.deposit(amount)` keeps the other side's history where the model wrote `other.balance += amount`)
        progress = True
        while progress:
            progress = False
            for name in names:
                if name in done:
                    continue
                for key, cand in _line_restores(fixed, original, name):
                    if (key, fixed) in tried:
                        continue
                    if time.monotonic() > deadline:
                        return fixed, done
                    tried.add((key, fixed))
                    if (await call_tool("run_tests", _run_candidate, ws, cand, *targs)).ok:
                        fixed = cand
                        done.append(f"{name}: {key[2].strip()}")
                        progress = True
                        break
                if progress:
                    break
    except Exception:
        log.exception("cut-back of a green patch failed; the patch stays as it is")
    return fixed, done


def _write_green(ws: Path, mylib_path: Path, text: str) -> None:
    write_mylib(mylib_path, text)
    shutil.rmtree(ws / "__pycache__", ignore_errors=True)
    shutil.rmtree(mylib_path.parent / "__pycache__", ignore_errors=True)   # a package's own


async def _write_and_cut(ws: Path, mylib_path: Path, original: str | None, fixed: str,
                         targs: tuple = ()) -> tuple[str, list[str]]:
    """The green patch goes to disk first (a deadline cancel inside the cut-back keeps it), then the cut-back; the
    file is written again only when the cut-back changed the text."""
    _write_green(ws, mylib_path, fixed)
    cut_text, cut = await _cut_back(ws, original, fixed, targs)
    if cut_text != fixed:
        _write_green(ws, mylib_path, cut_text)
    return cut_text, cut


def _cut_note(note: str, names: list[str]) -> str:
    return f"{note}, cut back to the original text: {names}" if names else note


def _other_sources(ws: Path, target: str) -> bool:
    """The workspace holds Python source besides the file under repair and the tests (another module the code under
    test may import)."""
    from harness.rescue import _protected
    try:
        return any(p.is_file() and p.relative_to(ws).as_posix() != target
                   and not _protected(p.relative_to(ws).as_posix()) and "__pycache__" not in p.parts
                   and p.name not in ("setup.py", "__init__.py")
                   for p in ws.rglob("*.py"))
    except (OSError, ValueError):
        return False


async def fix_coding(task: Task, client, resume: dict | None = None, restart: str | None = None) -> SolveResult:
    """`resume`: the `ladder` of an earlier result that found the LLM unavailable. Models answer the same at
    temperature 0, so only the rungs that got no answer are bought again, from the partial fix reached so far.
    `restart`: the file before phase-1 mutations (`extra["restart"]` of a partial repair); asked for first."""
    resume = resume or {}
    ws = Path(task.workspace)
    mylib_path = ws / "mylib.py"
    try:
        test_name, tests = _read_tests(ws)
        target = _target_file(ws, tests)   # mylib.py whenever the workspace has one
        targs = () if target == "mylib.py" else (target,)
        mylib_path = ws / target
        base = resume.get("base") or mylib_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        if is_free_coding(task):  # nothing to repair: the text asks for a program's output
            return await solve_with_program(task, client)
        if config.coding_rescue():   # expert 10: another build of task, one whole-file answer
            from harness.rescue import rescue_coding
            return await rescue_coding(task, client, f"cannot read workspace: {e}")
        return SolveResult("unparsed", note=f"cannot read workspace: {e}")
    base_report = await call_tool("run_tests", _run_candidate, ws, base, *targs)
    if base_report.ok:
        return SolveResult("solved", answer="DONE", note="tests already pass")
    # several test files below the root are one text here: no proof is read from it
    if len(_deep_tests(ws)) < 2 and provable_contradiction(tests, base, base_report.failed):  # visible tests cannot all pass: spend nothing
        return SolveResult("infeasible", note="tests contradict each other")
    missing = missing_test_import(tests, ws, base_report.output, task.instruction)
    if missing is not None:   # the tests cannot even load: nothing done to mylib.py turns them green
        return SolveResult("infeasible", note=f"the tests import a module that does not exist: {missing}")
    output = _shown_output(base_report)
    restart = restart or resume.get("restart")
    original = resume.get("original") or restart or base   # what a green patch is cut back to (task 88)
    restarted = bool(resume.get("restarted"))   # the restart call has answered already: never bought twice
    if restart and restart != base and not resume.get("answered") and not restarted:
        # phase 1 left a partial repair: mutations no one checked. One MODEL_FAST call for the whole repair from
        # the file before them; a green result is taken, anything else leaves the ladder below as it was
        try:
            first = await call_tool("run_tests", _run_candidate, ws, restart, *targs)
            if not first.ok and not any(f.startswith("<") for f in first.failed):
                r = await client.chat(MODEL_FAST, _patch_messages(restart, tests, _shown_output(first), target, test_name), 400)
                restarted = True
                fixed = apply_patch(restart, r.text)
                if fixed is not None and (await call_tool("run_tests", _run_candidate, ws, fixed, *targs)).ok:
                    fixed, cut = await _write_and_cut(ws, mylib_path, original, fixed, targs)
                    return SolveResult("solved", answer="DONE",
                                       note=_cut_note(f"patched by {MODEL_FAST} from the file before mutations", cut))
        except ModelFailing:
            pass  # only MODEL_FAST is out: the ladder below goes on to the other models
        except LLMUnavailable as e:
            # the same proxy would cost the ladder a second timeout: stop here and keep the restart, so that the
            # retry pass asks it again
            return SolveResult("unparsed", note=f"llm unavailable: {e}",
                               extra={"ladder": {"answered": [], "base": base, "restart": restart,
                                                 "original": original},
                                      "restart": restart})
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
                r = await client.chat(model, _patch_messages(base, tests, output, target, test_name), max_tokens, **extra)
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
            report = await call_tool("run_tests", _run_candidate, ws, fixed, *targs)
            if report.ok:
                fixed, cut = await _write_and_cut(ws, mylib_path, original, fixed, targs)
                return SolveResult("solved", answer="DONE", note=_cut_note(f"patched by {model}", cut))
            note = f"{model}: tests still failing"
            # a candidate that hangs where the base did not: its empty run says nothing of the base asked next
            hung = "<timeout>" in report.failed and "<timeout>" not in base_report.failed
            output = _shown_output(base_report if hung else report)
            if len(report.failed) < len(base_report.failed) and not any(f.startswith("<") for f in report.failed):
                base, base_report = fixed, report  # keep partial progress for the next attempt
            if "<timeout>" in base_report.failed and TIMEOUT_HINT not in output:
                output += "\n" + TIMEOUT_HINT   # the base asked next still hangs, whatever the candidate printed
        if answered or not failing:
            break
        # no model of this ladder has answered yet: the task is not given up for a passing failure
        await client.wait_for_model([a[0] for a in attempts])
        unavailable = None
    if unavailable is None:
        if config.coding_rescue() and note != "no attempt made" and _other_sources(ws, target):
            # expert 10: every rung answered and the tests are still red; the defect may sit in another module
            from harness.rescue import rescue_coding
            return await rescue_coding(task, client, note)
        return SolveResult("unparsed", note=note)
    ladder = {"answered": answered, "base": base}
    if original != base:   # the partial fix of a rung is the base now: the cut-back still goes to the original
        ladder["original"] = original
    if restarted:
        ladder["restarted"] = True
    return SolveResult("unparsed", note=f"llm unavailable: {unavailable}", extra={"ladder": ladder})

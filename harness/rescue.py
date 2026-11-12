"""Last resort for a coding task the ladder cannot serve (expert 10, patch b, HARNESS_CODING_RESCUE).

The ladder repairs one Python file against one test file. A task of another build ends without an answer and without
a call: the file to write does not exist yet (`solve.py` from a description), there is no test file (the examples in
the docstrings are the tests), the code is not Python, or the defect sits in a module the tests do not import. Here
the model sees the task text and the files of the workspace once and returns whole files. They are stored when the
checks that can be run here are green (the Python test files, a `python -m doctest` the task names) or when nothing
can be run at all (another language): a wrong answer costs nothing, no answer scores nothing. A red check stores
nothing. Test files are never written."""
import asyncio
import re
import shutil
import tempfile
from pathlib import Path

from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import LLMUnavailable, ProxyDown
from harness.proc import run_python
from harness.result import SolveResult
from harness.solvers.testrun import run_in_dir

RESCUE_SYSTEM = (
    "You finish a small programming task in a working directory where you cannot run commands. The task text and the "
    "files of the directory follow. Reply with the complete new text of every file you create or change and nothing "
    "else, one block per file, in exactly this format:\nFILE <relative path>\n<complete text of the file>\nEND FILE\n"
    "Never create or change test files (test_*.py, *_test.py, conftest.py, test.js, anything under tests/) and never "
    "change the examples inside docstrings. Hidden tests run the same code on other inputs: every function must do "
    "what its name, docstring, comments, the README and the task text say for every valid input. Keep the file "
    "names, function names and signatures the tests and the task use. No explanations, no markdown fences.")
MAX_TOKENS = 1800
SOURCE_CHARS = 4500    # more code shown whole than this: no call (a whole-file reply would not fit MAX_TOKENS)
FILE_CHARS = 8000      # a longer file is shown cut and may not be rewritten
TOTAL_CHARS = 24000
MAX_FILES = 24
MAX_WRITES = 6
CHECK_TIMEOUT = 30
_TEXT_SUFFIXES = frozenset({".py", ".js", ".mjs", ".ts", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
                            ".cfg", ".ini", ".csv", ".go", ".java", ".rb", ".c", ".h", ".cpp", ".rs", ".sql", ".html"})
_BLOCK = re.compile(r"^FILE[ \t]+(\S[^\n]*?)[ \t]*\n(.*?)(?:\nEND FILE[ \t]*$|(?=\nFILE[ \t]+\S)|\Z)", re.M | re.S)
# a reply that closes its blocks: only END FILE ends one (a line "FILE FORMAT v2" inside a string stays in the body)
_BLOCK_CLOSED = re.compile(r"^FILE[ \t]+(\S[^\n]*?)[ \t]*\n(.*?)^END FILE[ \t]*$", re.M | re.S)
_END = re.compile(r"^END FILE[ \t]*$", re.M)
_DOCTEST = re.compile(r"-m\s+doctest\s+(?:-v\s+)?([\w./-]+\.py)")
_SKIP_DIRS = frozenset({"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache"})
# an existing file is rewritten only when it is code: data and configuration the tests read stay as they are
_CODE_SUFFIXES = frozenset({".py", ".js", ".mjs", ".ts", ".go", ".java", ".rb", ".c", ".h", ".cpp", ".rs", ".sh"})
# what configures or feeds a test run, here or in the hidden one: never written
_RUN_CONFIG = frozenset({"pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "sitecustomize.py",
                         "usercustomize.py", "noxfile.py", "package.json", "run_tests.py", "run_tests.sh",
                         "runtests.py", ".coveragerc", ".mocharc.json", ".mocharc.js", "jest.config.js",
                         "jest.config.mjs", "vitest.config.js", "vitest.config.ts"})
_DATA_DIRS = ("testdata", "test_data", "fixtures", "__fixtures__", "__snapshots__", "expected")


def _protected(rel: str) -> bool:
    """A file of the tests or of their configuration: never written."""
    parts = rel.lower().split("/")
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    return (stem.startswith("test_") or stem.endswith("_test") or stem in ("test", "tests", "conftest")
            or ".test." in name or ".spec." in name or name in _RUN_CONFIG or name.endswith(".pth")
            or any(p in ("tests", "test", "__tests__", "spec") + _DATA_DIRS for p in parts[:-1]))


def _named_by_tests(ws: Path, rel: str) -> bool:
    """A file name some test file of the workspace mentions (expected output, fixture data)."""
    name = Path(rel).name
    try:
        for p in ws.rglob("*"):
            if (p.is_file() and p.suffix.lower() in _CODE_SUFFIXES and _protected(p.relative_to(ws).as_posix())
                    and not (set(p.relative_to(ws).parts[:-1]) & _SKIP_DIRS) and p.stat().st_size <= 200_000
                    and name in p.read_text(encoding="utf-8", errors="ignore")):
                return True
    except OSError:
        return True
    return False


def _files(ws: Path) -> list[tuple[str, str]]:
    """(relative path, text) of the workspace's text files, code before data and the smallest first so that a large
    one cannot crowd out the rest; at most MAX_FILES."""
    out = []
    try:
        paths = sorted(p for p in ws.rglob("*") if p.is_file() and not (set(p.relative_to(ws).parts[:-1]) & _SKIP_DIRS))
    except OSError:
        return []
    for p in paths:
        rel = p.relative_to(ws).as_posix()
        if p.suffix.lower() not in _TEXT_SUFFIXES or rel == "answer.txt":
            continue
        try:
            if p.stat().st_size > 200_000:
                continue
            out.append((rel, p.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    # code and tests first: thirty small data files must not crowd out the module to repair
    return sorted(out, key=lambda x: (Path(x[0]).suffix.lower() not in _CODE_SUFFIXES, len(x[1]), x[0]))[:MAX_FILES]


def messages(task, files: list[tuple[str, str]]) -> tuple[list[dict], set]:
    """The prompt, and the files shown whole (only those may be rewritten; a new file may always be created)."""
    blocks, whole, used = [], set(), 0
    for rel, text in sorted(files, key=lambda x: x[0]):
        shown = text
        if len(text) > FILE_CHARS or used + len(text) > TOTAL_CHARS:
            room = max(0, min(FILE_CHARS, TOTAL_CHARS - used))
            shown = text[:room] + "\n...[cut]..."
        else:
            whole.add(rel)
        used += len(shown)
        tag = " (tests, read-only)" if _protected(rel) else ""
        blocks.append(f"--- {rel}{tag} ---\n{shown}")
    user = f"Task:\n{task.instruction}\n\nFiles of the directory:\n" + ("\n".join(blocks) if blocks else "(none)")
    return [{"role": "system", "content": RESCUE_SYSTEM}, {"role": "user", "content": user}], whole


def parse(reply: str, ws: Path, whole: set) -> dict:
    """relative path -> new text for the blocks that may be stored: a path inside the workspace, not a test file, a
    text type, and either a new file or one that was shown whole. {} when the reply holds no such block."""
    out: dict = {}
    text = re.sub(r"<think>.*?</think>", "", (reply or "").replace("\r\n", "\n"), flags=re.S)
    for m in (_BLOCK_CLOSED if _END.search(text) else _BLOCK).finditer(text):
        rel = m.group(1).strip().strip("`'\"").replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        body = m.group(2)
        lines = body.split("\n")
        if lines and lines[0].lstrip().startswith("```"):     # a fenced block in spite of the format
            lines = lines[1:]
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and lines[-1].strip() == "```":
                lines.pop()
        body = "\n".join(lines).rstrip("\n") + "\n"
        p = Path(rel)
        if (not rel or p.is_absolute() or ".." in p.parts or rel.startswith("/") or ":" in rel or _protected(rel)
                or p.suffix.lower() not in _TEXT_SUFFIXES or rel == "answer.txt" or not body.strip()):
            continue
        if (ws / rel).exists() and (rel not in whole or p.suffix.lower() not in _CODE_SUFFIXES):
            continue
        if p.suffix.lower() not in _CODE_SUFFIXES and _named_by_tests(ws, rel):   # the tests' own data
            continue
        out[rel] = body
        if len(out) >= MAX_WRITES:
            break
    return out


def _checks(task, ws: Path) -> tuple[list[str], str | None]:
    """What can be run here: the Python test files (the root's, else those below it) and a doctest file the task
    names."""
    from harness.llm_routes import _deep_tests
    try:
        tests = sorted(p.name for pat in ("test_*.py", "*_test.py") for p in ws.glob(pat) if p.is_file())
    except OSError:
        tests = []
    tests = tests or _deep_tests(ws)
    m = _DOCTEST.search(task.instruction or "")
    doc = m.group(1).lstrip("./") if m else None
    if doc is not None and (".." in Path(doc).parts or Path(doc).is_absolute()):
        doc = None
    return tests, doc


def _examples(text: str) -> list[tuple[str, str]] | None:
    """(input, expected output) of every doctest example in the docstrings of `text`, whitespace folded; None when
    the text does not parse or an example is malformed."""
    import ast
    import doctest
    try:
        tree = ast.parse(text)
        out = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    out += [(" ".join(e.source.split()), " ".join(e.want.split()))
                            for e in doctest.DocTestParser().get_examples(doc)]
        return out
    except (SyntaxError, ValueError, RecursionError):
        return None


def _prompts(text: str) -> list[str]:
    return [" ".join(ln.split()) for ln in text.splitlines() if ln.lstrip().startswith(">>>")]


def _examples_kept(ws: Path, doc: str, new: dict) -> bool:
    """The doctest file keeps every example of the original, inputs and expected outputs alike, and has at least one:
    `python -m doctest` is green on a file without examples. An original that does not parse (the defect may be a
    syntax error) keeps at least every `>>>` line."""
    from collections import Counter
    try:
        old = (ws / doc).read_text(encoding="utf-8") if (ws / doc).is_file() else ""
    except (OSError, UnicodeDecodeError):
        return False
    text = new.get(doc, old)
    before, after = _examples(old), _examples(text)
    if not after:
        return False
    if before is None:
        return not (Counter(_prompts(old)) - Counter(_prompts(text)))
    return not (Counter(before) - Counter(after))


def _verify(task, ws: Path, new: dict) -> bool | None:
    """True: every check that can be run is green with the new files; False: one is red; None: nothing can be run."""
    tests, doc = _checks(task, ws)
    if not tests and doc is None:
        return None
    if doc is not None and not _examples_kept(ws, doc, new):
        return False
    tmp = Path(tempfile.mkdtemp(prefix="rescue_"))
    try:
        cand = tmp / "ws"
        shutil.copytree(ws, cand, ignore=shutil.ignore_patterns("__pycache__"))
        for rel, text in new.items():
            (cand / rel).parent.mkdir(parents=True, exist_ok=True)
            (cand / rel).write_text(text, encoding="utf-8", newline="\n")
        for name in tests:
            if not run_in_dir(cand, name, timeout=CHECK_TIMEOUT).ok:
                return False
        if doc is not None:
            if not (cand / doc).is_file():
                return False
            res = run_python(["-m", "doctest", doc], cwd=cand, timeout=CHECK_TIMEOUT)
            if res.timed_out or res.returncode != 0:
                return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _store(ws: Path, new: dict) -> None:
    for rel, text in new.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.rescue.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(p)
    shutil.rmtree(ws / "__pycache__", ignore_errors=True)


async def rescue_coding(task, client, why: str) -> SolveResult:
    """One call of MODEL_FAST, then one of MODEL_ALT when the first reply cannot be stored. `why`: the note of the
    result this replaces; it is kept when nothing is stored."""
    ws = Path(task.workspace)
    files = await asyncio.to_thread(_files, ws)
    if not files:
        return SolveResult("unparsed", note=why)
    if await asyncio.to_thread(_verify, task, ws, {}) is True:   # nothing to repair: the file stays as it is
        return SolveResult("solved", answer="DONE", note=f"{why}; rescue: checks already pass, nothing written"[:300])
    msgs, whole = messages(task, files)
    code = sum(len(text) for rel, text in files
               if rel in whole and not _protected(rel) and Path(rel).suffix.lower() in _CODE_SUFFIXES)
    if code > SOURCE_CHARS:   # a whole-file reply of that size would not fit MAX_TOKENS: two calls for nothing
        return SolveResult("unparsed", note=f"{why}; rescue: {code} chars of source, too long to rewrite whole"[:300])
    note = why
    ledger = getattr(client, "ledger", None)
    cost = sum(len(m["content"]) for m in msgs) / 30 + MAX_TOKENS   # prompt tokens / 10 + the whole reply
    for model in (MODEL_FAST, MODEL_ALT):
        try:
            affordable = ledger is None or ledger.can_afford(cost)
        except Exception:   # a foreign client
            affordable = True
        if not affordable:   # the reply would be cut to what is left, and a cut reply is never stored
            note = f"{why}; rescue: the budget left does not fit a whole-file reply"[:300]
            break
        try:
            r = await client.chat(model, msgs, MAX_TOKENS)
        except ProxyDown as e:   # no model answers: the second would only wait for its own timeout
            note = f"{why}; rescue: llm unavailable: {e}"[:300]
            break
        except LLMUnavailable as e:
            note = f"{why}; rescue: llm unavailable: {e}"[:300]
            continue
        if getattr(r, "truncated", False) or (r.completion_tokens or 0) >= MAX_TOKENS:
            note = f"{why}; rescue: {model} hit the output limit"[:300]
            continue
        new = parse(r.text, ws, whole)
        if not new:
            note = f"{why}; rescue: {model} returned no file"[:300]
            continue
        verdict = await asyncio.to_thread(_verify, task, ws, new)
        if verdict is False:
            note = f"{why}; rescue: {model}: checks still red"[:300]
            continue
        await asyncio.to_thread(_store, ws, new)
        how = "checks green" if verdict else "nothing to run here, stored unchecked"
        return SolveResult("solved", answer="DONE", note=f"rescue by {model}: wrote {sorted(new)} ({how})"[:300])
    return SolveResult("unparsed", note=note)

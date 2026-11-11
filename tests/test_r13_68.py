"""Task 68 (expert 6 prototype): a search base that is not plain markdown pages, and a coding task with nothing to repair,
go to the generic program route. Plain bases and repair tasks keep their routes."""
import asyncio
from pathlib import Path

from harness.llm import ChatResult
from harness.llm_routes import (SEARCH_SYSTEM, PROGRAM_SYSTEM, _is_kb_task, _kb_is_plain, _program_messages,
                                fix_coding, is_free_coding, solve_with_program)
from harness.tasks import Task


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=10, cost=11)


def _mk(tmp_path, ttype, text, check, files):
    d = tmp_path / "t"
    for n, c in files.items():
        p = d / "workspace" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c, encoding="utf-8")
    (d / "workspace").mkdir(parents=True, exist_ok=True)
    return Task("t", ttype, text, 2.0, check, d, d / "workspace")


PAGE = "# Ann Lee" + chr(10) + chr(10) + "Born: 1980" + chr(10)


def test_plain_kb_keeps_the_kb_route(tmp_path):
    t = _mk(tmp_path, "search", "In which year was Ann Lee born?", "exact_str", {"kb/person_ann_lee.md": PAGE})
    assert _kb_is_plain(t.workspace) and _is_kb_task(t)
    assert _program_messages(t)[0]["content"] == SEARCH_SYSTEM


def test_mixed_kb_goes_to_the_generic_route_and_reads_other_files(tmp_path):
    t = _mk(tmp_path, "search", "What is the price of the product with sku S-1?", "exact_str",
            {"kb/person_ann_lee.md": PAGE, "kb/data/products.csv": "sku,price" + chr(10) + "S-1,89.5" + chr(10)})
    assert not _kb_is_plain(t.workspace) and not _is_kb_task(t)
    msgs = _program_messages(t)
    assert msgs[0]["content"] == PROGRAM_SYSTEM and "kb/data/products.csv" in msgs[1]["content"]
    code = "import csv" + chr(10) + "print(next(r['price'] for r in csv.DictReader(open('kb/data/products.csv')) if r['sku']=='S-1'))"
    r = asyncio.run(solve_with_program(t, FakeClient([code])))
    assert (r.status, r.answer) == ("solved", "89.5")


def test_hidden_file_makes_the_base_mixed(tmp_path):
    t = _mk(tmp_path, "search", "What is the token?", "exact_str", {"kb/a.md": PAGE, "kb/.env": "TOKEN=x1"})
    assert not _is_kb_task(t)


def test_free_coding_runs_a_program_and_keeps_every_line(tmp_path):
    t = _mk(tmp_path, "coding", "The file text.txt contains 2 lines. Print each line reversed, one per line. "
            "Write a program that computes it.", "exact_str", {"text.txt": "ab" + chr(10) + "cd" + chr(10)})
    assert is_free_coding(t)
    code = "for l in open('text.txt').read().split():" + chr(10) + "    print(l[::-1])"
    c = FakeClient([code])
    r = asyncio.run(fix_coding(t, c))
    assert r.status == "solved" and r.answer == "ba" + chr(10) + "dc" and len(c.calls) == 1


def test_repair_task_without_mylib_stays_unparsed_without_a_call(tmp_path):
    t = _mk(tmp_path, "coding", "Run `python3 test_mylib.py`; fix the bugs in mylib.py.", "unittest", {})
    c = FakeClient([])
    r = asyncio.run(fix_coding(t, c))
    assert not is_free_coding(t) and r.status == "unparsed" and c.calls == []


def test_coding_with_a_test_file_is_not_free(tmp_path):
    t = _mk(tmp_path, "coding", "Make the tests pass.", "exact_str", {"test_x.py": "import unittest"})
    assert not is_free_coding(t)


def test_patch_applies_to_a_file_with_a_syntax_error():
    from harness.llm_routes import apply_patch
    nl = chr(10)
    src = "def add(a, b)" + nl + "    return a + b" + nl + nl + "def mul(a, b):" + nl + "    return a * b" + nl
    out = apply_patch(src, "FUNC add" + nl + "def add(a, b):" + nl + "    return a + b" + nl)
    assert out == src.replace("def add(a, b)", "def add(a, b):")
    # a patch that leaves the file unparseable is still refused
    assert apply_patch(src, "FUNC mul" + nl + "def mul(a, b):" + nl + "    return a * b" + nl) is None


def test_os_junk_does_not_make_the_base_mixed(tmp_path):
    t = _mk(tmp_path, "search", "In which year was Ann Lee born?", "exact_str",
            {"kb/person_ann_lee.md": PAGE, "kb/.DS_Store": "x", "kb/._person_ann_lee.md": "x"})
    assert _is_kb_task(t)


# ---- round 13 review additions

def test_free_coding_single_line_keeps_the_normalized_answer(tmp_path):
    t = _mk(tmp_path, "coding", "Print the word count of words.txt. Write a program that computes it.", "exact_str",
            {"words.txt": "a b c" + chr(10)})
    r = asyncio.run(fix_coding(t, FakeClient(["print('Answer:', len(open('words.txt').read().split()))"])))
    assert (r.status, r.answer) == ("solved", "3")


def test_free_coding_exact_int_takes_the_last_number(tmp_path):
    t = _mk(tmp_path, "coding", "Print the sum of numbers.txt. Write a program that computes it.", "exact_int",
            {"numbers.txt": "1" + chr(10) + "2" + chr(10)})
    code = "s = sum(int(x) for x in open('numbers.txt'))" + chr(10) + "print('lines', 2)" + chr(10) + "print(s)"
    r = asyncio.run(fix_coding(t, FakeClient([code])))
    assert (r.status, r.answer) == ("solved", "3")


def test_free_coding_without_output_writes_nothing_and_leaves_the_workspace(tmp_path):
    t = _mk(tmp_path, "coding", "Print the sum of numbers.txt. Write a program that computes it.", "exact_str",
            {"numbers.txt": "1" + chr(10)})
    code = "open('out.txt', 'w').write('1')" + chr(10) + "open('numbers.txt', 'w').write('x')"
    c = FakeClient([code, code])
    r = asyncio.run(fix_coding(t, c))
    assert r.status == "unparsed" and r.answer is None and not (r.extra or {}).get("best_guess")
    assert sorted(p.name for p in t.workspace.iterdir()) == ["numbers.txt"]
    assert (t.workspace / "numbers.txt").read_text(encoding="utf-8") == "1" + chr(10)
    assert not (t.dir / "answer.txt").exists() and len(c.calls) == 2


def test_free_coding_crash_gives_no_answer(tmp_path):
    t = _mk(tmp_path, "coding", "Print the sum of numbers.txt. Write a program that computes it.", "exact_str",
            {"numbers.txt": "1" + chr(10)})
    r = asyncio.run(fix_coding(t, FakeClient(["raise SystemExit(3)", "print(1/0)"])))
    assert r.status == "unparsed" and r.answer is None and not (r.extra or {}).get("best_guess")


def test_tests_in_a_subdirectory_or_named_in_the_text_are_not_free(tmp_path):
    t = _mk(tmp_path, "coding", "Make the program correct.", "exact_str", {"tests/test_app.py": "import unittest"})
    assert not is_free_coding(t)
    t = _mk(tmp_path, "coding", "Make the program correct.", "exact_str", {"app_test.py": "import unittest"})
    assert not is_free_coding(t)
    t = _mk(tmp_path, "coding", "Fix solution.py so that pytest passes.", "exact_str", {"solution.py": "x = 1"})
    assert not is_free_coding(t)
    t = _mk(tmp_path, "coding", "Fix the module.", "unittest", {"solution.py": "x = 1"})
    assert not is_free_coding(t)
    assert not is_free_coding(_mk(tmp_path, "math", "What is 2+2?", "exact_int", {}))


def test_free_coding_without_a_workspace_dir(tmp_path):
    d = tmp_path / "t0"
    d.mkdir()
    t = Task("t0", "coding", "There is no input file. Print the number of primes below 100. "
             "Write a program that computes it.", 1.0, "exact_int", d, d / "workspace")
    code = "print(sum(all(n % k for k in range(2, n)) for n in range(2, 100)))"
    r = asyncio.run(fix_coding(t, FakeClient([code])))
    assert (r.status, r.answer) == ("solved", "25")


_BROKEN = ("def add(a, b)" + chr(10) + "    return a - b" + chr(10) + chr(10)
           + "def mul(a, b):" + chr(10) + "    return a * b" + chr(10))
_TESTS = ("import unittest" + chr(10) + "from mylib import add, mul" + chr(10)
          + "class T(unittest.TestCase):" + chr(10)
          + "    def test_add(self):" + chr(10) + "        self.assertEqual(add(2, 3), 5)" + chr(10)
          + "    def test_mul(self):" + chr(10) + "        self.assertEqual(mul(2, 3), 6)" + chr(10)
          + "if __name__ == '__main__':" + chr(10) + "    unittest.main()" + chr(10))


def test_unparsable_mylib_patch_that_parses_but_fails_the_tests_is_refused(tmp_path):
    t = _mk(tmp_path, "coding", "Run `python3 test_mylib.py`; fix mylib.py.", "unittest",
            {"mylib.py": _BROKEN, "test_mylib.py": _TESTS})
    wrong = "FUNC add" + chr(10) + "def add(a, b):" + chr(10) + "    return a - b" + chr(10)
    c = FakeClient([wrong, wrong])
    r = asyncio.run(fix_coding(t, c))
    assert r.status == "unparsed" and "tests still failing" in r.note and len(c.calls) == 2
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == _BROKEN


def test_unparsable_mylib_patch_that_passes_the_tests_is_written(tmp_path):
    t = _mk(tmp_path, "coding", "Run `python3 test_mylib.py`; fix mylib.py.", "unittest",
            {"mylib.py": _BROKEN, "test_mylib.py": _TESTS})
    right = "FUNC add" + chr(10) + "def add(a, b):" + chr(10) + "    return a + b" + chr(10)
    r = asyncio.run(fix_coding(t, FakeClient([right])))
    assert (r.status, r.answer) == ("solved", "DONE")
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") == _BROKEN.replace(
        "def add(a, b)" + chr(10) + "    return a - b", "def add(a, b):" + chr(10) + "    return a + b")


def test_unparsable_test_file_stays_infeasible_in_phase_1(tmp_path):
    from harness.solvers import coding_solver
    t = _mk(tmp_path, "coding", "Run `python3 test_mylib.py`; fix mylib.py.", "unittest",
            {"mylib.py": _BROKEN, "test_mylib.py": "def broken(:" + chr(10)})
    r = coding_solver.solve(t, time_limit=30)
    assert r.status == "infeasible" and "does not parse" in r.note


# ---- fix round 1

def _plain(tmp_path, files):
    t = _mk(tmp_path, "search", "In which year was Ann Lee born?", "exact_str", files)
    return _kb_is_plain(t.workspace)


def test_plain_pages_of_any_page_suffix_keep_the_kb_route(tmp_path):
    bom = chr(0xfeff)
    assert _plain(tmp_path / "a", {"kb/person_ann.txt": PAGE, "kb/person_bob.markdown": PAGE, "kb/person_cy": PAGE})
    assert _plain(tmp_path / "b", {"kb/a.md": chr(10) + "  " + chr(10) + PAGE, "kb/b.MD": bom + PAGE})
    assert _plain(tmp_path / "c", {"kb/a.md": PAGE, "kb/.gitkeep": "", "kb/.DS_Store": "x", "kb/README.txt": "about",
                                   "kb/index.json": "{}", "kb/._a.md": "x", "kb/.empty": ""})


def test_other_files_make_the_base_mixed(tmp_path):
    assert not _plain(tmp_path / "a", {"kb/a.md": PAGE, "kb/data.csv": "# id,price" + chr(10) + "1,2" + chr(10)})
    assert not _plain(tmp_path / "b", {"kb/a.md": PAGE, "kb/sub/b.md": PAGE})
    assert not _plain(tmp_path / "c", {"kb/a.md": PAGE, "kb/notes.txt": "TOKEN: x1" + chr(10)})
    assert not _plain(tmp_path / "d", {"kb/a.md": PAGE, "kb/.env": "# secrets" + chr(10) + "TOKEN=x1"})
    d = tmp_path / "e" / "t" / "workspace" / "kb"
    d.mkdir(parents=True)
    (d / "a.md").write_text(PAGE, encoding="utf-8")
    (d / "b.md").write_bytes(b"\x1f\x8b\x08\x00binary\x00")
    assert not _kb_is_plain(d.parent)


def _mixed(tmp_path, q="Who leads project Harbor?"):
    return _mk(tmp_path, "search", q, "exact_str",
               {"kb/person_ann_lee.md": PAGE, "kb/data/projects.csv": "name,lead" + chr(10) + "Harbor,Ann Lee" + chr(10)})


def test_general_search_route_takes_a_placeholder_as_no_answer(tmp_path):
    t = _mixed(tmp_path)
    c = FakeClient(["import csv" + chr(10) + "print({}.get('x'))"])
    r = asyncio.run(solve_with_program(t, c))
    assert r.status == "unparsed" and r.answer is None and not r.extra and len(c.calls) == 1


def test_general_search_route_retries_a_container(tmp_path):
    t = _mixed(tmp_path)
    rows = "import csv" + chr(10) + "rows = list(csv.DictReader(open('kb/data/projects.csv')))" + chr(10)
    c = FakeClient([rows + "print(rows[0])", rows + "print(rows[0]['lead'])"])
    r = asyncio.run(solve_with_program(t, c))
    assert (r.status, r.answer) == ("solved", "Ann Lee") and len(c.calls) == 2
    c = FakeClient([rows + "print(rows[0])", rows + "print(rows)"])
    r = asyncio.run(solve_with_program(t, c))
    assert r.status == "unparsed" and r.answer is None and not (r.extra or {}).get("best_guess")


def test_reason_route_on_a_mixed_base_refuses_a_placeholder(tmp_path):
    from harness.llm_routes import solve_with_reason
    t = _mixed(tmp_path)
    r = asyncio.run(solve_with_reason(t, FakeClient(["```python" + chr(10) + "x = {}" + chr(10) + "print(x.get('a'))"
                                                       + chr(10) + "```"])))
    assert r.status == "unparsed" and "placeholder" in r.note


def test_text_locate_takes_the_decorators_with_the_function():
    from harness.llm_routes import apply_patch, _locate
    nl = chr(10)
    src = ("import functools" + nl + nl + "@functools.lru_cache(" + nl + "    maxsize=None)" + nl + "@staticmethod" + nl
           + "def f(a)" + nl + "    return a" + nl + nl + "def g(a):" + nl + "    return a" + nl)
    assert _locate(src)["f"] == (2, 7, 0)
    out = apply_patch(src, "FUNC f" + nl + "@functools.lru_cache(maxsize=None)" + nl + "def f(a):" + nl + "    return a + 1" + nl)
    assert out is not None and out.count("lru_cache") == 1 and "@staticmethod" not in out
    assert out.endswith("def g(a):" + nl + "    return a" + nl)


def test_a_page_with_stray_non_utf8_bytes_stays_plain(tmp_path):
    d = tmp_path / "t" / "workspace" / "kb"
    d.mkdir(parents=True)
    (d / "a.md").write_text(PAGE, encoding="utf-8")
    (d / "b.md").write_bytes(b"# Arktika Labs\n\nFounded: 1993\n\nCaf\xe9 \xc6\xff\xfe\x80")
    assert _kb_is_plain(d.parent)

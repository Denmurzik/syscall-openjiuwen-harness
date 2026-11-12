"""Round 21 fixes of patch b (HARNESS_CODING_RESCUE): no rewrite of what is green, no write of test data or
configuration, kept doctest examples, no call that cannot fit."""
import asyncio

from harness import llm_routes as L
from harness import rescue
from harness.llm import ChatResult
from harness.tasks import Task


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append((model, messages))
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=10, cost=11)

    _used = 0

    def _budget(self):
        return None


def _task(tmp_path, files, instruction="Some tests fail. Fix the bugs (do not edit the tests).", name="t", reward=2.0):
    d = tmp_path / name
    (d / "workspace").mkdir(parents=True)
    for rel, text in files.items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return Task(name, "coding", instruction, reward, "unittest", d, d / "workspace")


GOOD = 'def area(w, h):\n    """\n    >>> area(3, 4)\n    12\n    """\n    return w * h\n'
BAD = GOOD.replace("w * h", "w + h")


def test_an_already_green_doctest_costs_no_call_and_no_write(tmp_path):
    t = _task(tmp_path, {"geometry.py": GOOD}, "Make sure `python3 -m doctest geometry.py` finishes without failures.")
    c = Fake([])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and r.answer == "DONE" and "checks already pass" in r.note and not c.calls
    assert (t.workspace / "geometry.py").read_text(encoding="utf-8") == GOOD


DOC_TEXT = "Make `python3 -m doctest geometry.py` finish without failures."


def test_stripped_or_changed_examples_are_red(tmp_path):
    strip = 'FILE geometry.py\ndef area(w, h):\n    """Area."""\n    return w + h\nEND FILE\n'
    want = 'FILE geometry.py\n' + BAD.replace("    12\n", "    7\n") + 'END FILE\n'
    t = _task(tmp_path, {"geometry.py": BAD}, DOC_TEXT)
    c = Fake([strip, want])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "checks still red" in r.note and len(c.calls) == 2
    assert (t.workspace / "geometry.py").read_text(encoding="utf-8") == BAD
    t = _task(tmp_path, {"geometry.py": BAD}, DOC_TEXT, name="u")     # reindented, but every example kept: green
    fix = 'FILE geometry.py\ndef area(w, h):\n    """\n    >>> area(3,  4)\n    12\n    """\n    return w * h\nEND FILE\n'
    r = asyncio.run(L.fix_coding(t, Fake([fix])))
    assert r.status == "solved" and "checks green" in r.note


def test_a_new_doctest_file_needs_an_example(tmp_path):
    ws = tmp_path
    t = Task("t", "coding", "Write geo.py; `python -m doctest geo.py` must pass.", 2.0, "unittest", ws, ws)
    assert rescue._verify(t, ws, {"geo.py": "def f():\n    return 1\n"}) is False
    assert rescue._verify(t, ws, {"geo.py": 'def f():\n    """\n    >>> f()\n    1\n    """\n    return 1\n'}) is True


PRICING = {"pricing.py": "import json\n\n\ndef total(xs):\n    return sum(xs)\n",
           "helper.py": "X = 1\n",
           "expected.json": '{"total": 7}\n',
           "test_pricing.py": ("import json, unittest\nfrom pricing import total\n\n\nclass T(unittest.TestCase):\n"
                               "    def test_a(self):\n        exp = json.load(open('expected.json'))['total']\n"
                               "        self.assertEqual(total([1, 2, 3]), exp)\n")}


def test_the_data_the_tests_read_is_never_written(tmp_path):
    t = _task(tmp_path, PRICING, reward=1.0)
    useless = "FUNC total\ndef total(xs):\n    return sum(xs)\n"
    data = 'FILE expected.json\n{"total": 6}\nEND FILE\n'
    c = Fake([useless, useless, data, data])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "returned no file" in r.note
    assert (t.workspace / "expected.json").read_text(encoding="utf-8") == '{"total": 7}\n'
    ws = t.workspace    # a new file the tests name is their data too; a new file nobody names may be created
    assert rescue.parse("FILE expected_v2.json\n{}\nEND FILE\n", ws, set()) == {"expected_v2.json": "{}\n"}
    (ws / "test_more.py").write_text("open('expected_v2.json')\n", encoding="utf-8")
    assert rescue.parse("FILE expected_v2.json\n{}\nEND FILE\n", ws, set()) == {}


def test_run_configuration_and_existing_non_code_files_are_never_written(tmp_path):
    ws = tmp_path
    (ws / "README.md").write_text("# r\n", encoding="utf-8")
    (ws / "lib.py").write_text("x = 1\n", encoding="utf-8")
    reply = ("FILE pytest.ini\n[pytest]\naddopts = -k 'not hidden'\nEND FILE\nFILE setup.cfg\n[tool:pytest]\nEND FILE\n"
             "FILE pyproject.toml\n[tool.pytest.ini_options]\nEND FILE\nFILE sitecustomize.py\nimport os\nEND FILE\n"
             "FILE run_tests.py\nprint(1)\nEND FILE\nFILE testdata/case.json\n{}\nEND FILE\n"
             "FILE tox.ini\n[tox]\nEND FILE\nFILE package.json\n{}\nEND FILE\nFILE fixtures/a.py\nx = 1\nEND FILE\n"
             "FILE README.md\n# changed\nEND FILE\nFILE lib.py\nx = 2\nEND FILE\n")
    assert rescue.parse(reply, ws, {"README.md", "lib.py"}) == {"lib.py": "x = 2\n"}


def test_an_original_that_does_not_parse_keeps_its_prompt_lines(tmp_path):
    ws = tmp_path
    (ws / "g.py").write_text('def f(:\n    """\n    >>> f()\n    1\n    """\n', encoding="utf-8")
    ok = 'def f():\n    """\n    >>> f()\n    1\n    """\n    return 1\n'
    assert rescue._examples_kept(ws, "g.py", {"g.py": ok})
    assert not rescue._examples_kept(ws, "g.py", {"g.py": ok.replace(">>> f()", ">>> f() or 1")})


def test_too_much_source_to_rewrite_whole_costs_no_call(tmp_path):
    long_src = "".join(f"def f{i}(x):\n    '''Return x plus {i}.'''\n    return x + {i}\n\n\n" for i in range(100))
    t = _task(tmp_path, {"stats.js": "function f() {}\n", "lib.py": long_src, "notes.md": "x" * 6000},
              "`node test.js` fails. Fix stats.js.")
    c = Fake([])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "too long to rewrite whole" in r.note and not c.calls
    t = _task(tmp_path, {"stats.js": "function f() {}\n", "notes.md": "x" * 6000, "test.js": "y" * 6000},
              "`node test.js` fails. Fix stats.js.", name="u")     # data and tests do not count
    c = Fake(["FILE stats.js\nfunction f() { return 1; }\nEND FILE\n"])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and len(c.calls) == 1


def test_a_proxy_that_is_down_is_not_asked_twice(tmp_path):
    from harness.llm import ProxyDown

    class Down(Fake):
        async def chat(self, model, messages, max_tokens, **extra):
            self.calls.append((model, messages))
            raise ProxyDown("timeout")

    t = _task(tmp_path, {"stats.js": "function f() {}\n"}, "`node test.js` fails. Fix stats.js.")
    c = Down([])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "llm unavailable" in r.note and len(c.calls) == 1


def test_a_vendored_or_hidden_tree_is_not_another_source(tmp_path):
    ws = tmp_path
    for rel in (".venv/lib/site-packages/six.py", "node_modules/x/y.py", ".hidden/z.py", "venv/a.py",
                "pkg/__pycache__/m.py", "mylib.py", "test_mylib.py", "conftest.py", "setup.py"):
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text("x = 1\n", encoding="utf-8")
    assert not L._other_sources(ws, "mylib.py")
    (ws / "pkg/stats.py").write_text("x = 1\n", encoding="utf-8")
    assert L._other_sources(ws, "mylib.py")


def test_a_file_line_inside_a_closed_block_stays_in_the_body(tmp_path):
    body = 'def header():\n    return """\nFILE FORMAT v2\n"""\n\n\ndef f():\n    return 1\n'
    reply = f"FILE report.py\n{body}END FILE\nFILE b.py\ny = 2\nEND FILE\n"
    assert rescue.parse(reply, tmp_path, set()) == {"report.py": body, "b.py": "y = 2\n"}
    # an empty block does not swallow the next one
    assert rescue.parse("FILE a.py\nEND FILE\nFILE b.py\ny = 2\nEND FILE\n", tmp_path, set()) == {"b.py": "y = 2\n"}


def test_many_small_data_files_do_not_crowd_out_the_code(tmp_path):
    for i in range(30):
        (tmp_path / f"d{i:02d}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mylib.py").write_text("def f(x):\n    return x\n" * 5, encoding="utf-8")
    names = [rel for rel, _ in rescue._files(tmp_path)]
    assert names[0] == "mylib.py" and len(names) == rescue.MAX_FILES

"""Expert 10, patch b (HARNESS_CODING_RESCUE): a coding task the ladder cannot serve gets one whole-file answer."""
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


def _task(tmp_path, files, instruction="Some tests fail. Fix the bugs (do not edit the tests).", name="t"):
    d = tmp_path / name
    (d / "workspace").mkdir(parents=True)
    for rel, text in files.items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return Task(name, "coding", instruction, 2.0, "unittest", d, d / "workspace")


SOLVE_TESTS = ('import subprocess, sys, unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n'
               '        r = subprocess.run([sys.executable, "solve.py"], input="2 3\\n", capture_output=True, text=True)\n'
               '        self.assertEqual(r.stdout.strip(), "5")\n\n\nif __name__ == "__main__":\n    unittest.main()\n')
SOLVE_GOOD = "FILE solve.py\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\nEND FILE\n"
SOLVE_BAD = "FILE solve.py\nprint(0)\nEND FILE\n"


def test_a_program_written_from_scratch_is_stored_when_its_tests_are_green(tmp_path):
    t = _task(tmp_path, {"test_solve.py": SOLVE_TESTS}, "Write solve.py: read two integers, print their sum.")
    c = Fake([SOLVE_GOOD])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and r.answer == "DONE" and "checks green" in r.note and len(c.calls) == 1
    assert "print(a + b)" in (t.workspace / "solve.py").read_text(encoding="utf-8")
    assert "Write solve.py" in c.calls[0][1][1]["content"] and "test_solve.py (tests, read-only)" in c.calls[0][1][1]["content"]


def test_a_red_answer_is_not_stored_and_the_second_model_is_asked(tmp_path):
    t = _task(tmp_path, {"test_solve.py": SOLVE_TESTS}, "Write solve.py: read two integers, print their sum.")
    c = Fake([SOLVE_BAD, SOLVE_BAD])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and len(c.calls) == 2 and not (t.workspace / "solve.py").exists()
    assert r.note.startswith("cannot read workspace") and "checks still red" in r.note
    c = Fake([SOLVE_BAD, SOLVE_GOOD])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and [m for m, _ in c.calls] == [L.MODEL_FAST, L.MODEL_ALT]


DOC = ('def area(w, h):\n    """\n    >>> area(3, 4)\n    12\n    """\n    return w + h\n')
DOC_FIX = 'FILE geometry.py\ndef area(w, h):\n    """\n    >>> area(3, 4)\n    12\n    """\n    return w * h\nEND FILE\n'


def test_a_doctest_the_task_names_is_the_check(tmp_path):
    text = "Make `python3 -m doctest geometry.py` finish without failures."
    t = _task(tmp_path, {"geometry.py": DOC}, text)
    r = asyncio.run(L.fix_coding(t, Fake([DOC_FIX])))
    assert r.status == "solved" and "checks green" in r.note
    t = _task(tmp_path, {"geometry.py": DOC}, text, name="u")
    r = asyncio.run(L.fix_coding(t, Fake([DOC_FIX.replace("w * h", "w - h")] * 2)))
    assert r.status == "unparsed" and "return w + h" in (t.workspace / "geometry.py").read_text(encoding="utf-8")


JS = "function sum(xs) {\n  let t = 0;\n  for (let i = 1; i < xs.length; i++) t += xs[i];\n  return t;\n}\nmodule.exports = { sum };\n"
JS_TEST = 'const assert = require("assert");\nconst { sum } = require("./stats.js");\nassert.strictEqual(sum([1, 2]), 3);\n'


def test_another_language_is_stored_unchecked_and_its_test_file_is_never_written(tmp_path):
    t = _task(tmp_path, {"stats.js": JS, "test.js": JS_TEST}, "`node test.js` fails. Fix stats.js.")
    reply = ("FILE stats.js\n```js\n" + JS.replace("let i = 1", "let i = 0") + "```\nEND FILE\n"
             "FILE test.js\nconsole.log('ok');\nEND FILE\n")
    r = asyncio.run(L.fix_coding(t, Fake([reply])))
    assert r.status == "solved" and "stored unchecked" in r.note and "test.js" not in r.note
    assert "let i = 0" in (t.workspace / "stats.js").read_text(encoding="utf-8")
    assert "```" not in (t.workspace / "stats.js").read_text(encoding="utf-8")
    assert (t.workspace / "test.js").read_text(encoding="utf-8") == JS_TEST


MULTI = {"report.py": "from stats import mean\n\n\ndef summary(xs):\n    return mean(xs)\n",
         "stats.py": "def mean(xs):\n    return sum(xs) // len(xs)\n",
         "test_report.py": ("import unittest\nfrom report import summary\n\n\nclass T(unittest.TestCase):\n"
                            "    def test_a(self):\n        self.assertEqual(summary([1, 2]), 1.5)\n")}


def test_after_a_failed_ladder_another_module_can_be_repaired(tmp_path):
    t = _task(tmp_path, MULTI)
    t = Task(t.task_id, t.task_type, t.instruction, 1.0, t.check, t.dir, t.workspace)   # reward 1: two rungs
    useless = "FUNC summary\ndef summary(xs):\n    return mean(xs)\n"
    fix = "FILE stats.py\ndef mean(xs):\n    return sum(xs) / len(xs)\nEND FILE\n"

    class Client(Fake):
        _used = 0

        def _budget(self):
            return None

    c = Client([useless, useless, useless, fix])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "solved" and "rescue" in r.note and "stats.py" in r.note
    assert "/ len(xs)" in (t.workspace / "stats.py").read_text(encoding="utf-8")


def test_a_single_file_task_is_not_rescued_after_the_ladder(tmp_path):
    files = {"mylib.py": "def f(x):\n    return x + 1\n",
             "test_mylib.py": ("import unittest\nfrom mylib import f\n\n\nclass T(unittest.TestCase):\n"
                               "    def test_a(self):\n        self.assertEqual(f(2), 4)\n")}
    t = _task(tmp_path, files)
    c = Fake(["FUNC f\ndef f(x):\n    return x + 1\n"] * 3)
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and "rescue" not in r.note and len(c.calls) in (2, 3)


def test_the_switch_off_is_55354db(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CODING_RESCUE", "0")
    t = _task(tmp_path, {"test_solve.py": SOLVE_TESTS}, "Write solve.py.")
    c = Fake([SOLVE_GOOD])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and r.note.startswith("cannot read workspace") and not c.calls


def test_parse_refuses_paths_outside_the_workspace_and_cut_files(tmp_path):
    ws = tmp_path
    (ws / "big.py").write_text("x = 1\n", encoding="utf-8")
    reply = ("FILE ../evil.py\nx = 1\nEND FILE\nFILE /abs.py\nx = 1\nEND FILE\nFILE C:/abs.py\nx = 1\nEND FILE\n"
             "FILE tests/helper.py\nx = 1\nEND FILE\nFILE conftest.py\nx = 1\nEND FILE\nFILE a.exe\nx\nEND FILE\n"
             "FILE answer.txt\nDONE\nEND FILE\nFILE big.py\nx = 2\nEND FILE\nFILE ./pkg/new.py\nx = 3\nEND FILE\n")
    assert rescue.parse(reply, ws, whole=set()) == {"pkg/new.py": "x = 3\n"}
    assert rescue.parse(reply, ws, whole={"big.py"}) == {"big.py": "x = 2\n", "pkg/new.py": "x = 3\n"}
    assert rescue.parse("no blocks here", ws, set()) == {} and rescue.parse(None, ws, set()) == {}
    # a block without END FILE ends at the next FILE line or at the end of the reply
    assert rescue.parse("FILE a.py\nx = 1\nFILE b.py\ny = 2\n", ws, set()) == {"a.py": "x = 1\n", "b.py": "y = 2\n"}


def test_an_empty_workspace_costs_no_call(tmp_path):
    t = _task(tmp_path, {})
    c = Fake([])
    r = asyncio.run(L.fix_coding(t, c))
    assert r.status == "unparsed" and not c.calls

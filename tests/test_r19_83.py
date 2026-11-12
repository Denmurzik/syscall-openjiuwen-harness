"""Round 19, task 83: which exceptions the visible tests name for a function (round 18 review, R5)."""
import asyncio

import pytest

from harness import review
from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import ChatResult
from harness.tasks import Task

# ours raises ValueError below 0 (math.sqrt does, the text never says the word); the rewrites also above 1000
ROOT = ("import math\n\n\ndef safe_root(x):\n    \"\"\"Square root of x; x below 0 or above 1000 is out of range.\"\"\"\n"
        "    return math.sqrt(x)\n")
FIX = ("def safe_root(x):\n    \"\"\"Square root of x; x below 0 or above 1000 is out of range.\"\"\"\n"
       "    if x > 1000:\n        raise ValueError('out of range')\n    return math.sqrt(x)\n")
FIX2 = FIX.replace("    return math.sqrt(x)\n", "    return x ** 0.5 if x >= 0 else math.sqrt(x)\n")
BLOCK_TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
               "    def test_r(self):\n        self.assertEqual(m.safe_root(4), 2.0)\n\n"
               "    def test_neg(self):\n        with self.assertRaises(ValueError):\n"
               "            m.safe_root(-4)\n")

MEAN = ("def mean(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n\n\n"
        "def mean_abs(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sum(abs(x) for x in xs) / len(xs)\n")
MEAN_FIX = "def mean(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sum(xs) / len(xs)\n"
MEAN_FIX2 = MEAN_FIX.replace("sum(xs) / len(xs)", "sum(xs) / float(len(xs))")
MEAN_TESTS = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
              "    def test_m(self):\n        self.assertEqual(m.mean([2, 4]), 3)\n\n"
              "    def test_abs(self):\n        self.assertRaises(ValueError, m.mean_abs, [])\n")


def _reply(*funcs):
    return "\n".join(f"FUNC {f.split('(')[0].split()[-1]}\n{f}" for f in funcs)


def _task(tmp_path, lib, tests):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8")
    (d / "answer.txt").write_text("DONE", encoding="utf-8")
    return Task("t", "coding", "fix", 2.0, "unittest", d, d / "workspace")


class Fake:
    def __init__(self, fast, alt):
        self.replies, self.calls = {MODEL_FAST: fast, MODEL_ALT: alt}, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        return ChatResult(text=self.replies.get(model, "NONE"), prompt_tokens=10, completion_tokens=1, cost=2)


def _lib(task):
    return (task.workspace / "mylib.py").read_text(encoding="utf-8")


def _go(task, client):
    return asyncio.run(review.review(task, client))


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv("HARNESS_REVIEW_NAMED", raising=False)


def test_raise_named_by_a_with_block_on_another_line_is_stored(tmp_path):
    task = _task(tmp_path, ROOT, BLOCK_TESTS)
    assert _go(task, Fake(_reply(FIX), _reply(FIX2)))[0] == ["safe_root"]
    ns = {}
    exec(_lib(task), ns)
    assert ns["safe_root"](9) == 3.0
    with pytest.raises(ValueError):
        ns["safe_root"](2000)


def test_with_block_switch_off_is_round_18(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_NAMED", "0")
    task = _task(tmp_path, ROOT, BLOCK_TESTS)
    assert _go(task, Fake(_reply(FIX), _reply(FIX2)))[0] == [] and _lib(task) == ROOT


def test_a_longer_name_does_not_name_the_exception(tmp_path):
    task = _task(tmp_path, MEAN, MEAN_TESTS)
    assert _go(task, Fake(_reply(MEAN_FIX), _reply(MEAN_FIX2)))[0] == [] and _lib(task) == MEAN


def test_a_longer_name_switch_off_is_round_18(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_NAMED", "0")
    task = _task(tmp_path, MEAN, MEAN_TESTS)
    assert _go(task, Fake(_reply(MEAN_FIX), _reply(MEAN_FIX2)))[0] == ["mean"]


def test_naming_reads_lines_blocks_and_call_forms():
    tests = ("import pytest\nimport mylib as m\n\n\ndef test_a():\n    with pytest.raises(KeyError):\n"
             "        x = 1\n        m.lookup({}, 'k')\n\n\ndef test_b():\n    with pytest.raises(IndexError):\n"
             "        m.other([])\n\n\ndef test_c():\n    assert m.lookups(1) == 2\n\n\nclass T:\n"
             "    def test_d(self):\n        self.assertRaises(\n            ZeroDivisionError,\n"
             "            m.lookup, {}, 0)\n")
    text = "\n".join(review._naming(tests, "lookup"))
    assert "KeyError" in text and "ZeroDivisionError" in text and "IndexError" not in text
    assert "lookups" not in text
    assert review._naming(tests, "Store.lookup") == review._naming(tests, "lookup")    # a method by its own name
    assert review._naming("def f(:\n", "f") == ["def f(:"]      # no tree: the lines only


# fix round 1, I2: a block names the exception only for the call that raises, the last statement's outermost call
PARSE_MEAN = ("def parse(text):\n    \"\"\"Numbers of a comma-separated text; ValueError for anything else.\"\"\"\n"
              "    return [float(p) for p in text.split(',')]\n\n\n"
              "def mean(xs):\n    \"\"\"Mean of xs; 0.0 for an empty list.\"\"\"\n    if not xs:\n        return 0.0\n"
              "    return sum(xs) / len(xs)\n")
PARSE_TESTS = ("import pytest\nfrom mylib import parse, mean\n\n\ndef test_mean():\n    assert mean([1.0, 3.0]) == 2.0\n"
               "\n\ndef test_bad():\n    with pytest.raises(ValueError):\n        mean(parse('1,x'))\n")


def test_a_wrapping_call_does_not_name_the_inner_raise(tmp_path):
    fix = ("def mean(xs):\n    \"\"\"Mean of xs; 0.0 for an empty list.\"\"\"\n    if not xs:\n"
           "        raise ValueError('empty')\n    return sum(xs) / len(xs)\n")
    fix2 = fix.replace("sum(xs) / len(xs)", "sum(xs) / float(len(xs))")
    task = _task(tmp_path, PARSE_MEAN, PARSE_TESTS)
    assert _go(task, Fake(_reply(fix), _reply(fix2)))[0] == [] and _lib(task) == PARSE_MEAN
    assert not any("ValueError" in t for t in review._naming(PARSE_TESTS, "mean"))


def test_a_block_names_only_its_last_call():
    tests = ("import pytest\nimport mylib as m\n\n\ndef test_a():\n    with pytest.raises(KeyError):\n"
             "        x = m.normalize({})\n        m.validate(x)\n\n\ndef test_b():\n    with pytest.raises(IndexError):\n"
             "        y = m.first([])\n")
    assert not any("KeyError" in t for t in review._naming(tests, "normalize"))
    assert any("KeyError" in t for t in review._naming(tests, "validate"))
    assert any("IndexError" in t for t in review._naming(tests, "first"))      # an assignment's value counts

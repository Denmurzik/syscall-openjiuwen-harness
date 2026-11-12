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

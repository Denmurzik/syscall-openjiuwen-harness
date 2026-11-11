"""Coding ladder: the first call asks for the whole repair from the file before phase-1 mutations."""
import asyncio

from harness.llm import ChatResult, LLMUnavailable, ModelFailing, ProxyDown
from harness.llm_routes import fix_coding
from harness.solvers import coding_solver
from harness.tasks import Task

HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
MAIN = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
MIN_BUG = "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"
LAST_BUG = "def last_n(xs, n):\n    return xs[:n]\n"
TWO = (HEAD + "    def test_minimum(self):\n        self.assertEqual(m.minimum([3, 9, 2]), 2)\n\n"
       "    def test_last_n(self):\n        self.assertEqual(m.last_n([5, 6, 7, 8], 3), [6, 7, 8])\n" + MAIN)
GOOD_PATCH = ("FUNC minimum\ndef minimum(xs):\n    return min(xs)\n"
              "FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n")


def _coding(tmp_path, lib, tests, reward=2.0):
    d = tmp_path / "c"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(lib, encoding="utf-8", newline="\n")
    (d / "workspace" / "test_mylib.py").write_text(tests, encoding="utf-8", newline="\n")
    return Task("c", "coding", "Fix the bugs in mylib.py", reward, "unittest", d, d / "workspace")


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "user": messages[-1]["content"]})
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(text=r, prompt_tokens=10, completion_tokens=10, cost=11)


def _partial(tmp_path):
    lib = MIN_BUG + "\n\n" + LAST_BUG
    t = _coding(tmp_path, lib, TWO)
    res = coding_solver.solve(t, time_limit=60)
    return lib, t, res


def test_partial_repair_carries_the_file_before_mutations(tmp_path):
    lib, t, res = _partial(tmp_path)
    assert res.status == "solved" and res.tentative and "partial" in res.note
    assert res.extra == {"restart": lib}
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8") != lib   # the mutation is on disk


def test_ladder_asks_for_the_whole_repair_from_the_restart_file_first(tmp_path):
    lib, t, res = _partial(tmp_path)
    c = FakeClient([GOOD_PATCH])
    out = asyncio.run(fix_coding(t, c, restart=res.extra["restart"]))
    assert out.status == "solved" and "before mutations" in out.note and not out.tentative
    assert len(c.calls) == 1 and "x > best" in c.calls[0]["user"]   # the model saw the untouched minimum
    assert "return min(xs)" in (t.workspace / "mylib.py").read_text(encoding="utf-8")


def test_failed_restart_falls_back_to_the_ladder_from_the_partial_file(tmp_path):
    lib, t, res = _partial(tmp_path)
    partial = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    c = FakeClient(["nonsense", "FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, restart=res.extra["restart"]))
    assert out.status == "solved" and "before mutations" not in out.note
    assert len(c.calls) == 2 and "x < best" in c.calls[1]["user"]    # the second call got the partial repair
    assert (t.workspace / "mylib.py").read_text(encoding="utf-8").startswith(partial.split("\n\n\n")[0])


def test_half_green_restart_result_is_not_accepted(tmp_path):
    lib, t, res = _partial(tmp_path)
    only_min = "FUNC minimum\ndef minimum(xs):\n    return min(xs)\n"   # last_n still red
    c = FakeClient([only_min, "FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, restart=res.extra["restart"]))
    assert out.status == "solved" and "before mutations" not in out.note and len(c.calls) == 2


def _unavailable(tmp_path, exc):
    lib, t, res = _partial(tmp_path)
    c = FakeClient([exc, "FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, restart=res.extra["restart"]))
    return lib, t, c, out


def test_unavailable_model_on_the_restart_call_stops_and_keeps_the_restart(tmp_path):
    for exc in (LLMUnavailable("down"), ProxyDown("dead")):
        lib, t, c, out = _unavailable(tmp_path / type(exc).__name__, exc)
        assert out.status == "unparsed" and out.note.startswith("llm unavailable")
        assert len(c.calls) == 1                                   # the ladder did not go on
        assert out.extra["restart"] == lib and out.extra["ladder"]["restart"] == lib
        assert out.extra["ladder"]["answered"] == []


def test_failing_fast_model_on_the_restart_call_goes_on_to_the_ladder(tmp_path):
    lib, t, c, out = _unavailable(tmp_path, ModelFailing("flaky"))
    assert out.status == "solved" and len(c.calls) == 2 and "before mutations" not in out.note


def test_retry_pass_redoes_the_restart_call(tmp_path):
    lib, t, c, out = _unavailable(tmp_path, ProxyDown("dead"))
    c2 = FakeClient([GOOD_PATCH])
    again = asyncio.run(fix_coding(t, c2, out.extra["ladder"]))
    assert again.status == "solved" and "before mutations" in again.note and len(c2.calls) == 1


def test_restart_that_already_passes_makes_no_extra_call(tmp_path):
    lib, t, res = _partial(tmp_path)
    c = FakeClient(["FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    green = "def minimum(xs):\n    return min(xs)\n\n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"
    out = asyncio.run(fix_coding(t, c, restart=green))   # nothing to ask from a green file: the ladder goes on
    assert out.status == "solved" and "before mutations" not in out.note and len(c.calls) == 1


def test_broken_restart_is_skipped(tmp_path):
    lib, t, res = _partial(tmp_path)
    c = FakeClient(["FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, restart="def minimum(:\n"))
    assert out.status == "solved" and "before mutations" not in out.note and len(c.calls) == 1


def test_restart_equal_to_base_is_skipped(tmp_path):
    lib, t, res = _partial(tmp_path)
    partial = (t.workspace / "mylib.py").read_text(encoding="utf-8")
    c = FakeClient(["FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, restart=partial))
    assert "before mutations" not in out.note and len(c.calls) == 1


def test_answered_rungs_skip_the_restart(tmp_path):
    lib, t, res = _partial(tmp_path)
    c = FakeClient(["FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c, {"answered": ["other-model"]}, restart=lib))
    assert "before mutations" not in out.note and len(c.calls) == 1


def test_no_restart_means_the_old_ladder(tmp_path):
    lib, t, res = _partial(tmp_path)
    c = FakeClient(["FUNC last_n\ndef last_n(xs, n):\n    return xs[len(xs) - n:]\n"])
    out = asyncio.run(fix_coding(t, c))
    assert out.status == "solved" and len(c.calls) == 1

"""Expert pass 8: task shapes far from the examples (plain test functions, another module name, `import *`, prose
knowledge bases), the hop check of batched 2a specs, and the harness as an openJiuwen agent."""
import asyncio
import json

from harness import json_route, llm_routes as L
from harness.llm import ChatResult
from harness.result import SolveResult
from harness.solvers import probe
from harness.solvers.testrun import run_tests
from harness.tasks import Task

INS = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
       "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the "
       "word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost=6)


def _ws(tmp_path, files, name="t"):
    ws = tmp_path / name / "workspace"
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return ws


FIELDS = {"kb/person_ann_lee.md": "# Ann Lee\n\nBorn: 1970\nEmployer: Dune Robotics\nMentor: Bo Park\n",
          "kb/person_bo_park.md": "# Bo Park\n\nBorn: 1984\nEmployer: Dune Robotics\nMentor: Ann Lee\n",
          "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Graz\nFounded: 1999\n"}


# ------------------------------------------------------------------ batched 2a specs: hop count
def test_hop_count_mismatch():
    text = "Which year saw the foundation of the employer of the mentor of {E1}?"
    good = {"entity": "{E1}", "path": ["Mentor", "Employer"], "field": "Founded"}
    assert not json_route.hop_count_mismatch(text, good)
    assert json_route.hop_count_mismatch(text, {**good, "path": ["Mentor", "Mentor", "Employer"]})
    assert json_route.hop_count_mismatch(text, {**good, "path": ["Employer"]})
    assert not json_route.hop_count_mismatch(text, {"impossible": True})
    # a relation worded without the usual words is not counted at all
    assert not json_route.hop_count_mismatch("Which company pays the salary of the mentor of {E1}?",
                                             {"entity": "{E1}", "path": ["Mentor"], "field": "Employer"})


def test_a_batched_spec_with_a_wrong_hop_count_gets_its_own_call(tmp_path, monkeypatch):
    def task(i, q):
        ws = _ws(tmp_path, FIELDS, f"s{i}")
        return Task(f"s{i}", "search", f"The directory kb/ contains a small knowledge base of markdown files. {q}", 2.0,
                    "exact_str", tmp_path / f"s{i}", ws)
    from harness.templates import group
    t1, t2 = task(1, "Which year saw the foundation of the employer of the mentor of Ann Lee?"), task(2, "In which year was the mentor of Bo Park born?")
    items = [(tpl, members) for (_, tpl), members in group([t1, t2]).items()]
    bad = {"specs": [{"i": 1, "spec": {"entity": "{E1}", "hops": ["Mentor", "Mentor", "Employer", "Founded"]}},
                     {"i": 2, "spec": {"entity": "{E1}", "hops": ["Mentor", "Born"]}}]}
    single = {"entity": "{E1}", "hops": ["Mentor", "Employer", "Founded"]}
    client = Fake([json.dumps(bad), json.dumps(single)])
    out = asyncio.run(json_route.solve_batch("search", items, client))
    assert len(client.calls) == 2                      # the batch, then the dropped text alone
    assert out["s1"].status == "solved" and out["s1"].answer == "1999"
    assert out["s2"].status == "solved" and out["s2"].answer == "1970"

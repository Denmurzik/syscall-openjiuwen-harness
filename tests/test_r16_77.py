"""Round 16, task 77 (expert pass 8, patch c): the hop check of batched 2a specs."""
import asyncio
import json

from harness import json_route
from harness.llm import ChatResult
from harness.tasks import Task


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


# ------------------------------------------------------------------ fix round 1: the chain told twice is not counted
STEPS = ("- The kb/ directory contains a small knowledge base of markdown files.\n- We need to find the mentor of {E1}.\n"
         "- Then find the mentor of that person.\n- Then find the employer of that second mentor.\n"
         "- Finally, determine the city where that employer's headquarters is located.\n"
         "In which city is the headquarters of the employer of the mentor of the mentor of {E1}? Use the file tools or "
         "bash (ls, grep, cat) to find it.")
LEAD_IN = ("Let's trace the mentorship chain for {E1}. Who is the mentor of the mentor of the mentor of the mentor of "
           "{E1}? (The directory kb/ contains a small knowledge base of markdown files.) Use the file tools or bash "
           "(ls, grep, cat) to find it.")
Q21 = ("The directory kb/ contains a small knowledge base of markdown files. Which year saw the foundation of the "
       "employer of the mentor of {E1}? Use the file tools or bash (ls, grep, cat) to find it.")


def test_a_step_style_text_keeps_its_correct_batched_spec():
    # runs/r5/llm search_0073: the steps and the question both name the chain (Mentor 5 words, Employer 3 words)
    spec, _ = json_route._from_hops({"entity": "{E1}", "hops": ["Mentor", "Mentor", "Employer", "Headquarters"]})
    assert not json_route.hop_count_mismatch(STEPS, spec)


def test_a_lead_in_with_mentorship_keeps_its_correct_batched_spec():
    # runs/r5/llm search_0137: "mentorship" is not a hop
    spec, _ = json_route._from_hops({"entity": "{E1}", "hops": ["Mentor", "Mentor", "Mentor", "Mentor"]})
    assert not json_route.hop_count_mismatch(LEAD_IN, spec)
    assert json_route.hop_count_mismatch(LEAD_IN, {**spec, "path": ["Mentor", "Mentor", "Mentor", "Mentor", "Mentor"]})


def test_search_0021_wrong_batched_spec_is_still_rejected():
    good = {"entity": "{E1}", "path": ["Mentor", "Employer"], "field": "Founded"}
    assert not json_route.hop_count_mismatch(Q21, good)
    assert json_route.hop_count_mismatch(Q21, {**good, "path": ["Mentor", "Mentor", "Employer"]})

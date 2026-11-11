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


# ------------------------------------------------------------------ prose knowledge bases
BIO = {"kb/ann_lee.md": "# Ann Lee\n\nAnn Lee was born in Oslo in 1970. After graduating from Tartu College, Ann moved to Dune Robotics.\n",
       "kb/bo_park.md": "# Bo Park\n\nBo Park was born in Lyon in 1984. Bo works at Dune Robotics as an accountant.\n",
       "kb/dune_robotics.md": "# Dune Robotics\n\nDune Robotics was set up in 1999 and has its head office in Graz.\n"}
FIELDS = {"kb/person_ann_lee.md": "# Ann Lee\n\nBorn: 1970\nEmployer: Dune Robotics\nMentor: Bo Park\n",
          "kb/person_bo_park.md": "# Bo Park\n\nBorn: 1984\nEmployer: Dune Robotics\nMentor: Ann Lee\n",
          "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Graz\nFounded: 1999\n"}
Q = "The directory kb/ contains a small knowledge base of markdown files. Which college did Ann Lee graduate from? Use the file tools."


def _search(tmp_path, files, q=Q, check="exact_str"):
    ws = _ws(tmp_path, files)
    return Task("s", "search", q, 2.0, check, tmp_path / "t", ws)


def test_field_pages_are_not_prose(tmp_path):
    task = _search(tmp_path, FIELDS)
    assert L._is_kb_task(task) and not L._is_prose_task(task)
    assert L._program_messages(task)[0]["content"] == L.SEARCH_SYSTEM


def test_prose_pages_are_shown_as_text_the_named_page_first(tmp_path):
    task = _search(tmp_path, BIO)
    assert L._is_prose_task(task)
    m = L._program_messages(task)
    assert m[0]["content"] == L.PROSE_SYSTEM
    user = m[1]["content"]
    assert "`pages` has 3 pages" in user and user.index("# Ann Lee") < user.index("# Dune Robotics") < user.index("# Bo Park")


def test_prose_context_keeps_to_its_limit(tmp_path):
    files = {f"kb/p{i:03d}.md": f"# Person {i:03d}\n\n" + "Some sentence about this person. " * 20 for i in range(60)}
    files["kb/ann_lee.md"] = BIO["kb/ann_lee.md"]
    task = _search(tmp_path, files)
    shown, n = L._prose_context(task.workspace, Q)
    assert n == 61 and shown.startswith("# Ann Lee") and len(shown) <= L.PROSE_CONTEXT_CHARS + 200


def test_a_fact_quoted_from_a_prose_page_is_an_answer(tmp_path):
    task = _search(tmp_path, BIO)
    client = Fake(["print('Tartu College')"])
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.status == "solved" and res.answer == "Tartu College" and len(client.calls) == 1


def test_a_constant_no_page_holds_is_still_a_literal(tmp_path):
    task = _search(tmp_path, BIO)
    client = Fake(["print('Harvard')", "print(pages['Ann Lee'].split('graduating from ')[1].split(',')[0])"])
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.status == "solved" and res.answer == "Tartu College" and len(client.calls) == 2


def test_a_count_over_prose_pages_is_computed(tmp_path):
    q = Q.replace("Which college did Ann Lee graduate from?", "How many of the people work at Dune Robotics?")
    task = _search(tmp_path, BIO, q, "exact_int")
    client = Fake(["print(sum('Dune Robotics' in t and not k.startswith('Dune') for k, t in pages.items()))"])
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.status == "solved" and res.answer == "2"


def test_a_missing_prose_page_is_not_found(tmp_path):
    task = _search(tmp_path, BIO)
    res = asyncio.run(L.solve_with_program(task, Fake(["print(pages['Xavier Nobody'][:4])"])))
    assert res.status == "unparsed" and res.note == "kb page or field not found"

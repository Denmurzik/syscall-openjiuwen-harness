"""Round 16, task 76 (expert pass 8, patch b): search over a knowledge base written as prose."""
import asyncio

import pytest

from harness import llm_routes as L
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


# ------------------------------------------------------------------ fix round 1
def _people(n=30):
    return {f"kb/p{i:02d}.md": f"# Person {i:02d}\n\nPerson {i:02d} was born in {1950 + i} in Oslo. Person {i:02d} works "
                               f"at {'Dune Robotics' if i % 3 else 'Kite Labs'}.\n" for i in range(n)}


def test_a_count_printed_as_a_constant_is_never_taken(tmp_path):
    # 20 of the 30 work at Dune Robotics; "19" stands inside "1950": no longer quoted, and a count is never quoted
    q = Q.replace("Which college did Ann Lee graduate from?", "How many people work at Dune Robotics?")
    task = _search(tmp_path, _people(), q, "exact_int")
    client = Fake(["print(19)", "print(sum('Dune Robotics' in t for t in pages.values()))"])
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.status == "solved" and res.answer == "20" and len(client.calls) == 2
    assert "constant" in client.calls[1][-1]["content"]


def test_a_right_count_printed_as_a_constant_is_retried_too(tmp_path):
    q = Q.replace("Which college did Ann Lee graduate from?", "How many people work at Dune Robotics?")
    task = _search(tmp_path, _people(), q, "exact_int")
    client = Fake(["print(20)", "print(sum('Dune Robotics' in t for t in pages.values()))"])
    res = asyncio.run(L.solve_with_program(task, client))
    assert res.answer == "20" and len(client.calls) == 2


def test_quoted_constants_match_whole_words_only(tmp_path):
    ws = _ws(tmp_path, _people(10))   # Person 00..09, born 1950..1959
    for a in ("19", "95", "950", "Robo", "Osl"):
        assert not L._quoted_from_kb(ws, a), a
    for a in ("1950", "Oslo", "Dune Robotics", "Kite Labs"):
        assert L._quoted_from_kb(ws, a), a


@pytest.mark.parametrize("q", ["How many people work at Dune Robotics?", "What is the number of people born in Oslo?",
                               "Who is the oldest person at Kite Labs?", "Which company has more employees?",
                               "What is the total of the founding years?", "Who was born earliest?",
                               "Who joined their current employer earlier, Mina Garcia or Elena Olsen?"])
def test_aggregate_questions(q):
    assert L._asks_aggregate(q)


@pytest.mark.parametrize("q", ["Which college did Ann Lee graduate from?", "In which year was Bo Park born?",
                               "Where is the head office of the employer of Ann Lee?"])
def test_single_fact_questions(q):
    assert not L._asks_aggregate(q)


@pytest.mark.parametrize("fmt", ["- Born: {b}\n- Employer: {e}", "* Born: {b}\n* Employer: {e}",
                                 "**Born:** {b}\n**Employer:** {e}", "- **Born**: {b}\n- **Employer**: {e}",
                                 "__Born__: {b}\n__Employer__: {e}", "## Born: {b}\n## Employer: {e}",
                                 "Born:{b}\nEmployer:{e}"])
def test_field_pages_in_markdown_stay_on_the_kb_route(tmp_path, fmt):
    files = {f"kb/p{i}.md": f"# {t}\n\n" + fmt.format(b=b, e="Dune Robotics") + "\n"
             for i, (t, b) in enumerate([("Ann Lee", 1970), ("Bo Park", 1984), ("Cy Moss", 1991)])}
    task = _search(tmp_path, files)
    assert L._is_kb_task(task) and not L._is_prose_task(task)


def test_a_prose_title_line_is_not_a_field(tmp_path):
    files = {f"kb/p{i}.md": f"# Note: {t}\n\n{t} was born in {b} and works at Dune Robotics.\n"
             for i, (t, b) in enumerate([("Ann Lee", 1970), ("Bo Park", 1984), ("Cy Moss", 1991)])}
    assert L._kb_is_prose(_ws(tmp_path, files))


def test_a_cut_page_says_so_and_points_to_pages(tmp_path):
    long_bio = "# Ann Lee\n\n" + "Ann Lee travelled a lot in her youth. " * 60 + "Ann Lee graduated from Tartu College.\n"
    task = _search(tmp_path, {**BIO, "kb/ann_lee.md": long_bio})
    shown, _ = L._prose_context(task.workspace, Q)
    assert "Tartu College" not in shown
    assert "more characters of this page are cut here" in shown and "pages['Ann Lee']" in shown
    # a page that fits is shown whole, with no marker
    shown, _ = L._prose_context(_search(tmp_path / "b", BIO).workspace, Q)
    assert "cut here" not in shown

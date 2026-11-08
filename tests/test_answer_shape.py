"""Answer-shape validation (deterministic) and the one MODEL_ALT retry in routes 2a and 2b."""
import asyncio
import json
import shutil
from pathlib import Path

import pytest

from harness import config
from harness.json_route import solve_group
from harness.llm import ChatResult
from harness.llm_routes import solve_with_program
from harness.shape import INT, NAME, YEAR, expected_shape, mismatch
from harness.tasks import Task
from harness.templates import templatize

FIX = Path(__file__).parent / "fixtures" / "examples"
INTRO = "The directory kb/ contains a small knowledge base of markdown files. "
TAIL = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. "
        "When you are done, write ONLY the final answer to the file answer.txt in the current directory "
        "(for example with bash: echo 42 > answer.txt).")


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra": extra})
        return ChatResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost=6)


def _search(question: str, ws: Path = Path("."), tid: str = "s1") -> Task:
    return Task(tid, "search", INTRO + question + TAIL, 1.0, "exact_str", ws.parent, ws)


@pytest.mark.parametrize("question, shape", [
    ("In which year was Irina Smith born?", YEAR),
    ("Which calendar year did Ivan Novak come into the world?", YEAR),
    ("In what year did the employer of the mentor of Grace Kowalski-Bianchi come into being?", YEAR),
    ("Which year saw the foundation of the employer of Zoya Volkov?", YEAR),
    ("Please determine the birth year recorded for the mentor of Elena Haddad.", YEAR),
    ("When was Cobalt Works founded?", YEAR),
    ("Dig up the birth year of Hiro Moreau.", YEAR),
    ("Which firm employs Irina Smith work?", NAME),
    ("Which organisation has Elena Gomez on its payroll?", NAME),
    ("Tell me which person is mentoring the mentor of Sofia Smith.", NAME),
    ("In what city is the main office of the employer of Alice Garcia located?", NAME),
    ("Who, according to the knowledge base, mentors Hiro Smith?", NAME),
    ("Look up the city housing the headquarters of the employer of Hiro Sato.", NAME),
    ("Find the person mentoring Chen Sato.", NAME),
    ("Where is Cobalt Works headquartered?", NAME),
    # unknown field or mixed cues without a year/when question word: no check
    ("Frobnicate the alpha quietly?", None),
    ("Tell me the founding date of Ann Lee's employer.", None),
])
def test_search_shape_from_question_wording(question, shape):
    assert expected_shape(_search(question)) == shape


def test_shape_by_type_and_check():
    p = Path(".")
    assert expected_shape(Task("m", "math", "x", 1.0, "exact_int", p, p)) == INT
    assert expected_shape(Task("d", "data_analysis", "x", 1.0, "exact_int", p, p)) == INT
    assert expected_shape(Task("d", "data_analysis", "x", 1.0, "exact_str", p, p)) is None
    assert expected_shape(Task("c", "coding", "x", 1.0, "unittest", p, p)) is None


def test_mismatch():
    year = _search("In which year was Irina Smith born?")
    name = _search("Which firm employs Irina Smith?")
    p = Path(".")
    num = Task("m", "math", "x", 1.0, "exact_int", p, p)
    assert mismatch(year, "1994") is None and mismatch(year, " 1994\n") is None
    assert mismatch(year, "Bluefin Systems") and mismatch(year, "94")
    assert mismatch(name, "Bluefin Systems") is None and mismatch(name, "Rafael C. Jansen") is None
    assert mismatch(name, "1994") and mismatch(name, "12.5")
    assert mismatch(num, "12") is None and mismatch(num, "-3") is None
    assert mismatch(num, "12.5") and mismatch(num, "Porto")
    free = _search("Frobnicate the alpha quietly?")
    assert mismatch(free, "anything 1") is None


# ------------------------------------------------------------------ 2a

def _kb_ws(tmp_path):
    ws = tmp_path / "workspace"
    shutil.copytree(FIX / "search_0001" / "workspace", ws)
    from harness.solvers.search_solver import load_kb
    kb = load_kb(ws / "kb")
    name = next(n for n, f in kb.items() if "Born" in f and "Employer" in f)
    return ws, kb, name


def _group(task):
    template, slots = templatize(task.instruction)
    return template, [(task, slots)]


def test_2a_wrong_shape_retries_once_on_alt(tmp_path):
    ws, kb, name = _kb_ws(tmp_path)
    t = _search(f"Which firm employs {name}?", ws)
    template, members = _group(t)
    client = FakeClient('{"entity": "{E1}", "path": [], "field": "Born"}',
                        '{"entity": "{E1}", "path": [], "field": "Employer"}')
    out = asyncio.run(solve_group("search", template, members, client))
    assert [c["model"] for c in client.calls] == [config.MODEL_FAST, config.MODEL_ALT]
    assert out["s1"].status == "solved" and out["s1"].answer == kb[name]["Employer"]
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "name" in feedback and kb[name]["Born"] in feedback
    assert client.calls[1]["extra"] == {"response_format": {"type": "json_object"}}


def test_2a_wrong_shape_twice_is_unparsed(tmp_path):
    ws, kb, name = _kb_ws(tmp_path)
    t = _search(f"Which firm employs {name}?", ws)
    template, members = _group(t)
    client = FakeClient(*['{"entity": "{E1}", "path": [], "field": "Born"}'] * 2)
    out = asyncio.run(solve_group("search", template, members, client))
    assert len(client.calls) == 2
    assert out["s1"].status == "unparsed" and out["s1"].answer is None
    assert "shape" in out["s1"].note


def test_2a_right_shape_makes_one_call(tmp_path):
    ws, kb, name = _kb_ws(tmp_path)
    t = _search(f"In which year was {name} born?", ws)
    template, members = _group(t)
    client = FakeClient('{"entity": "{E1}", "path": [], "field": "Born"}')
    out = asyncio.run(solve_group("search", template, members, client))
    assert len(client.calls) == 1 and out["s1"].answer == kb[name]["Born"]


def test_2a_retry_keeps_good_members(tmp_path):
    """Only members with a wrong-shape answer take the retry result; the others keep theirs."""
    ws, kb, name = _kb_ws(tmp_path)
    t1 = _search(f"Which firm employs {name}?", ws, "s1")
    t2 = _search("Which firm employs Zzyzx Nobody?", ws, "s2")
    template, slots1 = templatize(t1.instruction)
    members = [(t1, slots1), (t2, templatize(t2.instruction)[1])]
    client = FakeClient('{"entity": "{E1}", "path": [], "field": "Born"}',
                        '{"entity": "{E1}", "path": [], "field": "Employer"}')
    out = asyncio.run(solve_group("search", template, members, client))
    assert len(client.calls) == 2
    assert out["s1"].answer == kb[name]["Employer"]
    assert out["s2"].status == "infeasible"  # never had an answer: untouched by the shape check


# ------------------------------------------------------------------ 2b

def _program_task(tmp_path, question):
    d = tmp_path / "t"
    (d / "workspace").mkdir(parents=True)
    return Task("t", "search", INTRO + question + TAIL, 1.0, "exact_str", d, d / "workspace")


def test_2b_wrong_shape_retries_once_on_alt(tmp_path):
    t = _program_task(tmp_path, "Which firm employs Ann Lee?")
    c = FakeClient("print(1994)", "print('Bluefin Systems')")
    r = asyncio.run(solve_with_program(t, c))
    assert r.status == "solved" and r.answer == "Bluefin Systems"
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_ALT]
    assert "1994" in c.calls[1]["messages"][-1]["content"]


def test_2b_wrong_shape_twice_is_unparsed(tmp_path):
    t = _program_task(tmp_path, "Which firm employs Ann Lee?")
    c = FakeClient("print(1994)", "print(1995)")
    r = asyncio.run(solve_with_program(t, c))
    assert r.status == "unparsed" and r.answer is None and "shape" in r.note
    assert len(c.calls) == 2


def test_2b_crash_then_wrong_shape_gets_one_alt_retry(tmp_path):
    t = _program_task(tmp_path, "In which year was Ann Lee born?")
    c = FakeClient("raise SystemExit(3)", "print('Porto')", "print(1983)")
    r = asyncio.run(solve_with_program(t, c))
    assert r.status == "solved" and r.answer == "1983"
    assert [x["model"] for x in c.calls] == [config.MODEL_FAST, config.MODEL_FAST, config.MODEL_ALT]


def test_2b_right_shape_one_call(tmp_path):
    t = _program_task(tmp_path, "In which year was Ann Lee born?")
    c = FakeClient("print(1983)")
    r = asyncio.run(solve_with_program(t, c))
    assert r.answer == "1983" and len(c.calls) == 1

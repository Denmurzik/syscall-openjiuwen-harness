import json
import shutil
from pathlib import Path

from harness.result import SolveResult
from harness.solvers.search_solver import execute, load_kb, parse, solve
from harness.tasks import Task

FIX = Path(__file__).parent / "fixtures" / "examples"
WS1 = FIX / "search_0001" / "workspace"
INTRO = "The directory kb/ contains a small knowledge base of markdown files. "
OUTRO = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory."
         " When you are done, write ONLY the final answer to the file answer.txt in the current directory.")


def _q(question: str) -> str:
    return INTRO + question + OUTRO


def _task(instruction: str, workspace: Path = WS1) -> Task:
    return Task("t", "search", instruction, 1.0, "exact_str", Path("."), workspace)


def _fixture(name: str) -> dict:
    return json.loads((FIX / name / "task.json").read_text(encoding="utf-8"))


def _kb(tmp_path: Path, files: dict[str, str]) -> Path:
    kb = tmp_path / "workspace" / "kb"
    kb.mkdir(parents=True)
    for name, text in files.items():
        (kb / name).write_text(text, encoding="utf-8")
    return tmp_path / "workspace"


def test_examples():
    for name in ("search_0001", "search_0005"):
        data = _fixture(name)
        r = solve(_task(data["instruction"], FIX / name / "workspace"))
        assert r == SolveResult("solved", data["expected"]), name


def test_unknown_person_is_infeasible():
    assert solve(_task(_q("In which year was Zed Nobody born?"))).status == "infeasible"


def test_two_hop_mentor_employer():
    kb = load_kb(WS1 / "kb")
    mentor = kb["Irina Smith"]["Mentor"]
    employer = kb[mentor]["Employer"]
    expected = kb[employer]["Headquarters"]
    r = solve(_task(_q("In which city is the headquarters of Irina Smith's mentor's employer?")))
    assert r == SolveResult("solved", expected)


def test_unknown_question_is_unparsed():
    assert solve(_task(_q("How tall is Irina Smith?"))).status == "unparsed"


def test_parse_gives_spec():
    data = _fixture("search_0005")
    assert parse(data["instruction"]) == {"entity": "Sergei Garcia", "path": ["Employer"], "field": "Headquarters"}


def test_parse_variants():
    cases = {
        "Where is Helix Motors headquartered?": ("Helix Motors", [], "Headquarters"),
        "In which year was Helix Motors founded?": ("Helix Motors", [], "Founded"),
        "Who is the mentor of Alice Novak?": ("Alice Novak", [], "Mentor"),
        "Who is Alice Novak\u2019s mentor?": ("Alice Novak", [], "Mentor"),
        "Which company does Alice Novak work for?": ("Alice Novak", [], "Employer"),
        "Who employs the mentor of Alice Novak?": ("Alice Novak", ["Mentor"], "Employer"),
        "Who is Alice Novak's employer?": ("Alice Novak", [], "Employer"),
        "Who is the employer of Alice Novak?": ("Alice Novak", [], "Employer"),
        "Where is the headquarters of Helix Motors?": ("Helix Motors", [], "Headquarters"),
        "In which city is Helix Motors headquartered?": ("Helix Motors", [], "Headquarters"),
        "What year was Alice Novak born?": ("Alice Novak", [], "Born"),
        "When was Helix Motors founded?": ("Helix Motors", [], "Founded"),
        "Who mentors Alice Novak?": ("Alice Novak", [], "Mentor"),
        "in  which year was  ALICE novak born.": ("ALICE novak", [], "Born"),
        "In which year was the employer of Alice Novak founded?": ("Alice Novak", ["Employer"], "Founded"),
    }
    for question, (entity, path, field) in cases.items():
        assert parse(_q(question)) == {"entity": entity, "path": path, "field": field}, question


def test_parse_rejects_noise():
    assert parse(_q("In which year was Alice Novak born? Also, where?")).status == "unparsed"
    assert parse(INTRO.strip()).status == "unparsed"
    assert parse(_q("In which year was  born?")).status == "unparsed"


def test_case_and_whitespace_insensitive_names():
    r = solve(_task(_q("In which year was  irina   SMITH born?")))
    assert r == SolveResult("solved", "1983")


def test_missing_workspace_or_kb_is_infeasible(tmp_path):
    spec = {"entity": "A", "path": [], "field": "Born"}
    assert execute(spec, tmp_path / "nope").status == "infeasible"
    assert execute(spec, tmp_path).status == "infeasible"


def test_execute_arbitrary_field_and_path(tmp_path):
    ws = _kb(tmp_path, {
        "a.md": "# Ann\n\nBorn: 1990\nBoss: Bob\nPet: Rex\n",
        "b.md": "# Bob\n\nBorn: 1970\nCity: Oslo\n",
    })
    assert execute({"entity": "Ann", "path": ["Boss"], "field": "City"}, ws) == SolveResult("solved", "Oslo")
    assert execute({"entity": "Ann", "path": [], "field": "pet"}, ws) == SolveResult("solved", "Rex")
    assert execute({"entity": "Ann", "path": ["Pet"], "field": "Born"}, ws).status == "infeasible"
    assert execute({"entity": "Ann", "path": ["Nope"], "field": "Born"}, ws).status == "infeasible"
    assert execute({"entity": "Ann", "path": [], "field": "Salary"}, ws).status == "infeasible"


def test_duplicate_name_first_file_wins(tmp_path):
    ws = _kb(tmp_path, {"b.md": "# Ann\nBorn: 2\n", "a.md": "# Ann\nBorn: 1\n"})
    assert execute({"entity": "Ann", "path": [], "field": "Born"}, ws) == SolveResult("solved", "1")


def test_execute_rejects_malformed_specs(tmp_path):
    ws = _kb(tmp_path, {"a.md": "# Ann\nBorn: 1\n"})
    bad = [
        None, [], {}, {"entity": "Ann"},
        {"entity": "Ann", "path": [], "field": "Born", "extra": 1},
        {"entity": 5, "path": [], "field": "Born"},
        {"entity": "", "path": [], "field": "Born"},
        {"entity": "Ann", "path": "Employer", "field": "Born"},
        {"entity": "Ann", "path": [1], "field": "Born"},
        {"entity": "Ann", "path": [], "field": ""},
        {"entity": "Ann", "path": [], "field": None},
        {"entity": "Ann", "path": ["X"] * 50, "field": "Born"},
    ]
    for spec in bad:
        assert execute(spec, ws).status == "unparsed", spec


def test_load_kb(tmp_path):
    ws = _kb(tmp_path, {"a.md": "# Ann Lee\n\nBorn: 1990\nNote: a: b\nbroken line\n", "x.txt": "# No\nBorn: 1\n",
                        "nohead.md": "Born: 5\n"})
    assert load_kb(ws / "kb") == {"Ann Lee": {"Born": "1990", "Note": "a: b"}}


def test_solve_never_raises(tmp_path):
    assert solve(_task(_q("In which year was Irina Smith born?"), tmp_path / "missing")).status == "infeasible"
    assert solve(_task(None)).status == "unparsed"  # type: ignore[arg-type]


def test_ambiguous_scope_gives_alternatives():
    spec = parse(_q("Who is the mentor of Alice Novak's employer?"))
    assert spec["alternatives"] == [
        {"entity": "Alice Novak", "path": ["Employer"], "field": "Mentor"},
        {"entity": "Alice Novak", "path": ["Mentor"], "field": "Employer"},
    ]
    assert execute(spec, WS1).status == "unparsed"  # execute stays single-spec strict


def test_ambiguous_scope_solve(tmp_path):
    ws = _kb(tmp_path, {
        "a.md": "# Ann\nEmployer: Acme\n",
        "c.md": "# Acme\nMentor: Dan\n",
    })
    # reading 1: mentor of (Ann's employer) = Dan; reading 2: (mentor of Ann)'s employer has no answer
    assert solve(_task(_q("Who is the mentor of Ann's employer?"), ws)) == SolveResult("solved", "Dan")
    assert solve(_task(_q("Who is the mentor of Nobody's employer?"), ws)).status == "infeasible"
    # both readings answer with different values -> unparsed
    ws2 = _kb(tmp_path / "x", {
        "a.md": "# Ann\nEmployer: Acme\nMentor: Bob\n",
        "b.md": "# Bob\nEmployer: Zed\n",
        "c.md": "# Acme\nMentor: Dan\n",
    })
    assert solve(_task(_q("Who is the mentor of Ann's employer?"), ws2)).status == "unparsed"


def test_ambiguous_readings_disagree_is_unparsed(tmp_path):
    ws = _kb(tmp_path, {
        "a.md": "# Ann\nEmployer: Acme\nMentor: Bob\n",
        "b.md": "# Bob\nEmployer: Zed\nMentor: Cy\n",
        "c.md": "# Acme\nEmployer: Q\nMentor: Dan\n",
        "d.md": "# Ann's employer\nMentor: Eve\n",
    })
    # reading 1: entity named "Ann's employer" -> Eve; reading 2: mentor of Acme -> Dan
    assert solve(_task(_q("Who is the mentor of Ann's employer?"), ws)).status == "unparsed"


def test_names_with_the_period_and_possessive(tmp_path):
    ws = _kb(tmp_path, {
        "a.md": "# The Foo Group\nFounded: 1999\nHeadquarters: Rome\n",
        "b.md": "# Acme Inc.\nFounded: 2001\n",
        "c.md": "# Macy's\nFounded: 1858\n",
    })
    assert solve(_task(_q("When was The Foo Group founded?"), ws)) == SolveResult("solved", "1999")
    assert solve(_task(_q("Where is The Foo Group headquartered?"), ws)) == SolveResult("solved", "Rome")
    assert solve(_task(_q("Who employs Acme Inc.?"), _kb(tmp_path / "w2", {"a.md": "# Bo" + chr(10) + "Employer: Acme Inc." + chr(10),
                                                                         "b.md": "# Acme Inc." + chr(10) + "Employer: X" + chr(10)}))) == SolveResult("solved", "X")
    assert solve(_task(_q("In which year was Macy's founded?"), ws)) == SolveResult("solved", "1858")
    assert solve(_task(_q("In which year was Acme Inc founded?"), ws)).status == "infeasible"

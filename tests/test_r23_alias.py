"""Expert 12, patch c (HARNESS_INITIAL_ALIAS): a start entity written with an initial that fits exactly one page of
the base is that page; several fitting pages send the task on to 2b; none leaves it infeasible as before."""
import pytest

from harness.solvers.search_solver import _initial_pages, execute


def _kb(tmp_path, people):
    kb = tmp_path / "kb"
    kb.mkdir()
    for name, employer in people.items():
        (kb / f"person_{name.lower().replace(' ', '_')}.md").write_text(
            f"# {name}\n\nEmployer: {employer}\nBorn: 1970\n", encoding="utf-8")
    return tmp_path


PEOPLE = {"Rhea Costa": "Fathom Marine", "Mira Costa": "Birch Analytics", "Mila Costa": "Glint Software",
          "Greta Duval": "Ember Optics"}


def _run(tmp_path, entity):
    return execute({"entity": entity, "path": [], "field": "Employer"}, _kb(tmp_path, PEOPLE))


@pytest.mark.parametrize("entity", ["R. Costa", "R Costa", "Rhea C."])
def test_an_initial_that_fits_one_page_is_that_page(tmp_path, monkeypatch, entity):
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    res = _run(tmp_path, entity)
    assert res.status == "solved" and res.answer == "Fathom Marine"


def test_an_initial_that_fits_several_pages_goes_on_to_2b(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    res = _run(tmp_path, "M. Costa")
    assert res.status == "unparsed" and "several pages" in res.note


@pytest.mark.parametrize("entity", ["Z. Costa", "R. Smith", "R. C.", "Rhea Smith"])
def test_no_fitting_page_stays_infeasible(tmp_path, monkeypatch, entity):
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    res = _run(tmp_path, entity)
    assert res.status == "infeasible" and res.note.startswith("start entity not found")


def test_exact_name_is_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    res = _run(tmp_path, "Greta Duval")
    assert res.status == "solved" and res.answer == "Ember Optics"


def test_switch_off_restores_round_22(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_INITIAL_ALIAS", "0")
    res = _run(tmp_path, "R. Costa")
    assert res.status == "infeasible" and res.note == "start entity not found: 'R. Costa'"


def test_initial_pages_needs_a_whole_word_and_the_same_number_of_words():
    index = {"rhea costa": {}, "rhea t. costa": {}, "costa": {}}
    assert _initial_pages("R. Costa", index) == ["rhea costa"]
    assert _initial_pages("Costa", index) == []
    assert _initial_pages("R. C.", index) == []


def _search_task(tmp_path, instruction):
    import json
    from harness.tasks import load_tasks
    d = tmp_path / "tasks" / "search_a"
    (d / "workspace").mkdir(parents=True)
    _kb(d / "workspace", PEOPLE)
    (d / "task.json").write_text(json.dumps({"task_id": "search_a", "task_type": "search", "reward": 2.0,
                                             "instruction": instruction, "check": "exact_str"}), encoding="utf-8")
    return load_tasks(tmp_path / "tasks")[0]


def test_prompt_names_the_one_page_an_initial_fits(tmp_path, monkeypatch):
    from harness.llm_routes import _alias_hint, _program_messages
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    task = _search_task(tmp_path, 'A letter is signed "R. Costa". Which company employs the sender?')
    hint = _alias_hint(task)
    assert "'R. Costa'" in hint and "'Rhea Costa'" in hint
    assert hint in _program_messages(task)[1]["content"]


def test_prompt_is_unchanged_without_a_single_fitting_page(tmp_path, monkeypatch):
    from harness.llm_routes import _alias_hint, _program_messages
    monkeypatch.delenv("HARNESS_INITIAL_ALIAS", raising=False)
    for i, q in enumerate(('A letter is signed "M. Costa". Which company employs the sender?',
                           'A letter is signed "Z. Costa". Which company employs the sender?',
                           "Which company employs Rhea Costa?")):
        task = _search_task(tmp_path / str(i), q)
        assert _alias_hint(task) == ""
        on = _program_messages(task)[1]["content"]
        monkeypatch.setenv("HARNESS_INITIAL_ALIAS", "0")
        assert _program_messages(task)[1]["content"] == on
        monkeypatch.delenv("HARNESS_INITIAL_ALIAS")


def test_switch_off_leaves_the_prompt_alone(tmp_path, monkeypatch):
    from harness.llm_routes import _alias_hint
    monkeypatch.setenv("HARNESS_INITIAL_ALIAS", "0")
    assert _alias_hint(_search_task(tmp_path, 'A letter is signed "R. Costa". Who employs the sender?')) == ""

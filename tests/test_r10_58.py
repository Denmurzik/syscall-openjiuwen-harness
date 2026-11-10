"""Round 10, task 58: a count question has no shape."""
import pytest

from harness.shape import _search_shape

INTRO = "The directory kb/ contains a small knowledge base of markdown files. "
OUTRO = " Use the file tools or bash (ls, grep, cat) to find it."


@pytest.mark.parametrize("q,shape", [
    ("How many employees of Bluefin Systems were born after 1975?", None),
    ("How many people have Dmitri Smith as their mentor?", None),
    ("How old was Irina Smith when she joined?", None),
    ("What is the number of employees born in 1980?", None),
    ("Answer yes or no: was Irina Smith born in 1980?", None),
    ("In which year was Irina Smith born?", "year"),
    ("In which city is the headquarters of Sergei Garcia's employer?", "name"),
    ("Who is the mentor of Irina Smith?", "name"),
])
def test_count_question_has_no_shape(q, shape):
    assert _search_shape(INTRO + q + OUTRO) == shape


@pytest.mark.parametrize("q,shape", [
    ("Which company has the largest number of employees?", "name"),
    ("Who has the largest number of mentees?", "name"),
    ("Who mentors the greatest number of people?", "name"),
    ("Where is the headquarters of the company with the highest number of employees?", "name"),
    ("In which city is the headquarters of the company that employs the largest number of people?", "name"),
])
def test_number_of_inside_a_name_question_keeps_shape(q, shape):
    assert _search_shape(INTRO + q + OUTRO) == shape


@pytest.mark.parametrize("q", [
    "Is Irina Smith older than Sergei Garcia?",
    "Does Bluefin Systems employ more than 5 people?",
    "Was Irina Smith born before 1980?",
    "What is the number of mentees of Irina Smith?",
    "How much older is Irina Smith than Sergei Garcia?",
])
def test_count_and_yes_no_questions_have_no_shape(q):
    assert _search_shape(INTRO + q + OUTRO) is None


def test_kb_prompt_says_answer_only_the_asked_field():
    from harness.llm_routes import SEARCH_SYSTEM
    assert "Answer only with the field the question asks for" in SEARCH_SYSTEM
    assert "print IMPOSSIBLE" in SEARCH_SYSTEM.split("Answer only with the field")[1].split("Never write")[0]

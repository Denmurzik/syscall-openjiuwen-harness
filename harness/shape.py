"""Deterministic check of a model-route answer's shape (no model call).

search: the asked field is read from the question wording (Born/Founded -> a 4-digit year;
Headquarters/Employer/Mentor -> a name, not a number); math/data with check exact_int -> an integer.
Unknown or conflicting wording -> no check.
"""
import re

from harness.solvers.common import strip_boilerplate

YEAR = "year"
NAME = "name"
INT = "int"

DESCRIBE = {YEAR: "a 4-digit year", NAME: "a name (text, not a number)", INT: "a whole number"}

_INTRO = re.compile(r"the directory kb/ contains a small knowledge base of markdown files\.", re.I)
_WORD = re.compile(r"[a-z]+")
# question words that name the asked kind outright
_YEAR_ASK = {"year", "when"}
_NAME_ASK = {"who", "whom", "where", "city"}
# with a bare "year" (no "when"), these make "year" possibly part of a clue, not the asked kind:
# "Which person was born in the same year as X?"
_NAME_NOUNS = {"person", "individual", "company", "firm", "organisation", "organization"}
# field words (Born/Founded vs Headquarters/Employer/Mentor and their paraphrases)
_YEAR_CUES = {"born", "birth", "birthday", "founded", "founding", "foundation", "established"}
_NAME_CUES = {"headquarters", "headquarter", "headquartered", "head", "office", "based", "located",
              "employer", "employs", "employ", "employed", "employing", "payroll", "company", "firm",
              "organisation", "organization", "mentor", "mentors", "mentored", "mentoring", "person",
              "individual"}

_IS_YEAR = re.compile(r"\d{4}")
_IS_INT = re.compile(r"[-+]?\d+")
_IS_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


# a count or a difference is a number whatever field words the question holds ("How many employees of X were
# born after 1975?", "How many people have X as their mentor?"): no shape is known for it. Only when the count
# phrase is the question's ask: the question opens with it (or with a yes/no verb), or it appears and no explicit
# who/which/where/when/what-city ask does ("Which company has the largest number of employees?" keeps its shape)
_COUNT_PHRASE = re.compile(r"\bhow (?:many|much|old)\b|\bnumber of\b|\byes or no\b", re.I)
_COUNT_OPENS = re.compile(
    r"^\W*(?:answer\s+(?:with\s+)?yes or no\W+)?(?:how (?:many|much|old)\b|what (?:is|was|are|were) the (?:total )?"
    r"number of\b|(?:is|are|does|did|was|were|do|has|have|can|will)\b)", re.I)
_EXPLICIT_ASK = {"who", "whom", "whose", "which", "where", "when"}
_ASKED_NOUN = {"city", "company", "firm", "person", "year", "organisation", "organization", "individual"}


def _count_ask(text: str) -> bool:
    if _COUNT_OPENS.match(text):
        return True
    if not _COUNT_PHRASE.search(text):
        return False
    words = _WORD.findall(text.casefold())
    if set(words) & _EXPLICIT_ASK:
        return False
    return not any(a == "what" and b in _ASKED_NOUN for a, b in zip(words, words[1:]))


def _search_shape(instruction: str) -> str | None:
    text = _INTRO.sub(" ", strip_boilerplate(instruction))
    if _count_ask(text):
        return None
    words = set(_WORD.findall(text.casefold()))
    year_ask, name_ask = bool(words & _YEAR_ASK), bool(words & _NAME_ASK)
    if year_ask and not name_ask and "when" not in words and words & _NAME_NOUNS:
        return None
    if year_ask != name_ask:
        return YEAR if year_ask else NAME
    if year_ask:
        return None  # both kinds asked
    year, name = bool(words & _YEAR_CUES), bool(words & _NAME_CUES)
    if year != name:
        return YEAR if year else NAME
    return None


def expected_shape(task) -> str | None:
    if task.task_type in ("math", "data_analysis"):
        return INT if task.check == "exact_int" else None
    if task.task_type == "search":
        return _search_shape(task.instruction or "")
    return None


def mismatch(task, answer: str) -> str | None:
    """None if the answer fits the expected shape (or none is known), else a short reason."""
    shape = expected_shape(task)
    if shape is None:
        return None
    a = (answer or "").strip()
    ok = (_IS_YEAR.fullmatch(a) if shape == YEAR
          else _IS_INT.fullmatch(a) if shape == INT
          else not _IS_NUMBER.fullmatch(a) and re.search(r"[^\W\d_]", a))
    return None if ok else f"expected {DESCRIBE[shape]}, got {a[:60]!r}"

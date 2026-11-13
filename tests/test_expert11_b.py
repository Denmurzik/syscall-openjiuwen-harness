"""Expert 11: HARNESS_FULL_SMALL (a), HARNESS_KB_EXTRAS (b), HARNESS_REVIEW_RETRY (c), HARNESS_REVIEW_LONG (d)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import llm_routes as L
from harness import review as r
from harness.tasks import Task

UT = "import unittest\n"


def _kb_task(tmp_path, question):
    kb = tmp_path / "workspace" / "kb"
    kb.mkdir(parents=True)
    (kb / "company_acme.md").write_text("# Acme\n\n- Headquarters: Graz\n- Founded: 1990\n", encoding="utf-8")
    (kb / "person_ada.md").write_text("# Ada Abara\n\n- Born: 1970\n- Employer: Acme\n", encoding="utf-8")
    (kb / "person_bo.md").write_text("# Bo Brandt\n\n- Born: 1980\n- Employer: Acme\n", encoding="utf-8")
    return Task("search_1", "search", "The directory kb/ contains a small knowledge base of markdown files. " + question,
                1, "exact_str", tmp_path, tmp_path / "workspace")


def test_plain_base_gets_the_same_prompt(tmp_path, monkeypatch):
    t = _kb_task(tmp_path, "Where is the headquarters of Acme?")
    on = L._program_messages(t)
    monkeypatch.setenv("HARNESS_KB_EXTRAS", "0")
    assert on == L._program_messages(t)


def test_page_without_fields_is_shown(tmp_path, monkeypatch):
    t = _kb_task(tmp_path, "Where is the headquarters of Acme today?")
    (t.workspace / "kb" / "news.md").write_text("# News 2023\n\nAcme moved its headquarters from Graz to Faro.\n",
                                                encoding="utf-8")
    assert L._is_kb_task(t) and not L._is_prose_task(t)
    user = L._program_messages(t)[1]["content"]
    assert "pages['News 2023']" in user and "from Graz to Faro" in user
    monkeypatch.setenv("HARNESS_KB_EXTRAS", "0")
    assert "Faro" not in L._program_messages(t)[1]["content"]


def test_file_outside_kb_is_shown(tmp_path):
    t = _kb_task(tmp_path, "The file hr/pay.csv lists salaries. What is the total salary of the employees of Acme?")
    (t.workspace / "hr").mkdir()
    (t.workspace / "hr" / "pay.csv").write_text("name,salary\nAda Abara,10\nBo Brandt,20\n", encoding="utf-8")
    user = L._program_messages(t)[1]["content"]
    assert "--- hr/pay.csv (first lines) ---" in user and "Ada Abara,10" in user and "open()" in user

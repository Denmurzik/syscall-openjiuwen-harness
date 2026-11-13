"""Expert 11: HARNESS_FULL_SMALL (a), HARNESS_KB_EXTRAS (b), HARNESS_REVIEW_RETRY (c), HARNESS_REVIEW_LONG (d)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import llm_routes as L
from harness import review as r
from harness.tasks import Task

UT = "import unittest\n"


NOTES = "Meeting notes\nAttendees: A, B\n\nDiscussion\n- budget\n\nAction items\n- Ada to book the venue\n"


def test_small_document_is_shown_whole(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES, encoding="utf-8")
    out = L.describe_workspace(tmp_path)
    assert "--- notes.txt (whole file) ---" in out and "Ada to book the venue" in out


def test_switch_off_is_the_old_preview(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text(NOTES, encoding="utf-8")
    monkeypatch.setenv("HARNESS_FULL_SMALL", "0")
    out = L.describe_workspace(tmp_path)
    assert "--- notes.txt (first lines) ---" in out and "Ada" not in out


def test_tables_logs_code_and_large_files_keep_five_lines(tmp_path, monkeypatch):
    rows = "".join(f"{i},x{i}\n" for i in range(40))
    (tmp_path / "t.csv").write_text("id,v\n" + rows, encoding="utf-8")
    (tmp_path / "app.log").write_text(rows, encoding="utf-8")
    (tmp_path / "m.py").write_text("\n".join(f"a{i} = {i}" for i in range(40)), encoding="utf-8")
    (tmp_path / "big.md").write_text("# Big\n" + "line of text\n" * 200, encoding="utf-8")
    on = L.describe_workspace(tmp_path)
    monkeypatch.setenv("HARNESS_FULL_SMALL", "0")
    assert on == L.describe_workspace(tmp_path) and "(whole file)" not in on


def test_a_document_of_five_lines_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\n\n- Born: 1970\n", encoding="utf-8")
    on = L.describe_workspace(tmp_path)
    monkeypatch.setenv("HARNESS_FULL_SMALL", "0")
    assert on == L.describe_workspace(tmp_path)


def test_whole_documents_stay_within_the_extra_room(tmp_path, monkeypatch):
    for i in range(8):
        (tmp_path / f"d{i}.txt").write_text("".join(f"line {j} of document {i}\n" for j in range(50)), encoding="utf-8")
    on = L.describe_workspace(tmp_path)
    monkeypatch.setenv("HARNESS_FULL_SMALL", "0")
    off = L.describe_workspace(tmp_path)
    assert 0 < len(on.encode()) - len(off.encode()) <= L.FULL_EXTRA
    assert on.count("(whole file)") < 8 and "(first lines)" in on

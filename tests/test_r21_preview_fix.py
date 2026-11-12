"""Round 21 fixes of patch d (HARNESS_RICH_PREVIEW): a bounded cost per call and per task."""
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from harness import llm_routes as L


def test_a_large_json_is_not_read_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RICH_JSON_MAX", 200)
    p = tmp_path / "events.json"
    p.write_text(json.dumps([{"id": i, "kind": "click"} for i in range(100)], indent=1), encoding="utf-8")
    reads = []
    real = Path.read_text

    def spy(self, *a, **k):
        reads.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    out = L.describe_workspace(tmp_path)
    assert "--- events.json (first lines) ---" in out and "objects with keys" not in out and reads == []


def test_the_json_line_uses_the_preview_already_read(tmp_path, monkeypatch):
    (tmp_path / "rows.json").write_text(json.dumps([{"sku": i} for i in range(50)], indent=1), encoding="utf-8")
    reads = []
    real = Path.read_text

    def spy(self, *a, **k):
        reads.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    assert "(a list of 50 objects with keys 'sku')" in L.describe_workspace(tmp_path)
    assert reads == ["rows.json"]   # once, by the structure line itself


def test_a_large_workbook_shows_its_sheet_names_without_openpyxl(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Sales & Co"
    wb.active.append(["id", "amount"])
    wb.create_sheet("Returns")
    wb.save(tmp_path / "sales.xlsx")
    monkeypatch.setattr(L, "RICH_XLSX_MAX", 10)
    monkeypatch.setattr(openpyxl, "load_workbook", lambda *a, **k: pytest.fail("opened"))
    out = L.describe_workspace(tmp_path)
    assert "(Excel workbook, too large to show rows here; sheets: 'Sales & Co', 'Returns')" in out


def test_sqlite_views_show_their_schema_only_and_long_statements_are_cut(tmp_path, monkeypatch):
    con = sqlite3.connect(tmp_path / "a.db")
    con.execute("create table t(x integer)")
    con.executemany("insert into t values (?)", ((i,) for i in range(300)))
    con.execute("create view v as select a.x + b.x as s from t a, t b order by s desc")
    con.commit()
    con.close()
    out = L.describe_workspace(tmp_path)
    assert "CREATE VIEW v as" in out and "(598,)" not in out and "(0,)" in out
    monkeypatch.setattr(L, "RICH_SQL_STEPS", 1)
    L._RICH_CACHE.clear()
    out = L.describe_workspace(tmp_path)      # every statement interrupted: the file is only named
    assert "a.db" in out


def test_the_block_is_built_once_per_unchanged_file(tmp_path, monkeypatch):
    with zipfile.ZipFile(tmp_path / "kb.zip", "w") as z:
        z.writestr("a.md", "# A\n")
    calls = []
    real = L._binary_preview
    monkeypatch.setattr(L, "_binary_preview", lambda p, raw: calls.append(p.name) or real(p, raw))
    first = L.describe_workspace(tmp_path)
    assert L.describe_workspace(tmp_path) == first and calls == ["kb.zip"]
    with zipfile.ZipFile(tmp_path / "kb.zip", "w") as z:
        z.writestr("a.md", "# A\n")
        z.writestr("b.md", "# B\n")
    assert "(zip archive, 2 members)" in L.describe_workspace(tmp_path) and calls == ["kb.zip", "kb.zip"]


def test_an_archive_with_many_members_is_not_opened(tmp_path, monkeypatch):
    with zipfile.ZipFile(tmp_path / "kb.zip", "w") as z:
        for i in range(30):
            z.writestr(f"p{i}.md", "x")
    assert L._zip_members(tmp_path / "kb.zip") == 30
    monkeypatch.setattr(L, "RICH_ZIP_MEMBERS", 20)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **k: pytest.fail("opened"))
    out = L.describe_workspace(tmp_path)
    assert "--- kb.zip ---\n(zip archive, 30 members)" in out


def test_a_database_name_with_a_hash_or_a_space_creates_no_file(tmp_path):
    for name in ("c#d.db", "my data.db", "q%41.db"):
        con = sqlite3.connect(tmp_path / name)
        con.execute("create table t(x)")
        con.execute("insert into t values (7)")
        con.commit()
        con.close()
    before = sorted(p.name for p in tmp_path.iterdir())
    out = L.describe_workspace(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert out.count("CREATE TABLE t(x)") == 3 and out.count("(7,)") == 3

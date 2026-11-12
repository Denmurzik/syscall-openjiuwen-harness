"""Expert 10, patch d (HARNESS_RICH_PREVIEW): the 2b prompt shows what is inside SQLite, xlsx, zip and gzip files and
the top-level structure of a JSON file longer than its preview; what it adds comes after every text preview."""
import gzip
import json
import sqlite3
import zipfile

import pytest

from harness import llm_routes as L


def _db(path):
    con = sqlite3.connect(path)
    con.execute("create table customer(id integer primary key, full_name text)")
    con.execute("insert into customer values (1, 'Wanda Castillo')")
    con.execute('create table "odd name"(x)')
    con.commit()
    con.close()


def test_sqlite_schema_and_rows(tmp_path):
    _db(tmp_path / "store.db")
    before = sorted(p.name for p in tmp_path.iterdir())
    out = L.describe_workspace(tmp_path)
    assert "CREATE TABLE customer(id integer primary key, full_name text)" in out
    assert "(1, 'Wanda Castillo')" in out and "odd name" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == before      # no journal, nothing written


def test_xlsx_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Orders"
    wb.active.append(["order_id", "amount"])
    wb.active.append([7, 1.5])
    wb.create_sheet("Customers").append(["customer_id", "name"])
    wb.save(tmp_path / "shop.xlsx")
    out = L.describe_workspace(tmp_path)
    assert "sheet 'Orders':" in out and "('order_id', 'amount')" in out and "sheet 'Customers':" in out


def test_zip_and_gzip(tmp_path):
    with zipfile.ZipFile(tmp_path / "kb.zip", "w") as z:
        z.writestr("kb/a.md", "# Ada\n\n- Born: 1815\n")
        z.writestr("kb/b.md", "# Bob\n")
    with gzip.open(tmp_path / "log.txt.gz", "wb") as f:
        f.write(b"line one\nline two\n")
    out = L.describe_workspace(tmp_path)
    assert "(zip archive, 2 members)" in out and "kb/a.md" in out and "- Born: 1815" in out
    assert "(gzip file, first lines after decompression)" in out and "line two" in out


def test_json_structure_line_only_when_the_preview_is_cut(tmp_path):
    (tmp_path / "small.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "regions.json").write_text(json.dumps({"towns": {f"t{i}": "r" for i in range(9)},
                                                       "capitals": {"r": "c"}}, indent=1), encoding="utf-8")
    (tmp_path / "rows.json").write_text(json.dumps([{"sku": i, "qty": i} for i in range(50)], indent=1),
                                        encoding="utf-8")
    out = L.describe_workspace(tmp_path)
    assert "'capitals': object with 1 keys such as 'r'" in out
    assert "(a list of 50 objects with keys 'sku', 'qty')" in out
    assert "--- small.json ---" not in out


def test_additions_follow_every_text_preview_and_the_switch_restores_the_old_text(tmp_path, monkeypatch):
    _db(tmp_path / "a.db")
    for i in range(6):
        (tmp_path / f"note{i}.md").write_text(f"# Note {i}\n\nbody {i}\n", encoding="utf-8")
    (tmp_path / "z.json").write_text(json.dumps({"k": list(range(40)), "m": {"x": 1}}, indent=1), encoding="utf-8")
    new = L.describe_workspace(tmp_path)
    monkeypatch.setenv("HARNESS_RICH_PREVIEW", "0")
    old = L.describe_workspace(tmp_path)
    assert new.startswith(old) and new != old
    assert "SQLite" not in old and "top-level keys" not in old


def test_a_damaged_file_is_only_named(tmp_path):
    (tmp_path / "broken.db").write_bytes(b"SQLite format 3\x00" + b"\xff" * 64)
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 20)
    (tmp_path / "bad.json").write_text("{\n" + '"a": [1,\n' * 30, encoding="utf-8")
    out = L.describe_workspace(tmp_path)
    assert "broken.db" in out and "--- broken.db ---" not in out and "--- broken.zip ---" not in out
    assert "--- bad.json ---" not in out


def test_the_total_stays_within_the_limit(tmp_path):
    for i in range(30):
        _db(tmp_path / f"d{i:02d}.db")
    assert len(L.describe_workspace(tmp_path).encode("utf-8")) <= L.PREVIEW_TOTAL

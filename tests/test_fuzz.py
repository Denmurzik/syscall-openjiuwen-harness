"""Smoke test for tools/fuzz.py: a tiny run per difficulty, report files, no WRONG on the generator's own phrasings."""
import json
from pathlib import Path

import pytest

from tools import fuzz


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_build_and_run_chunk(tmp_path, difficulty):
    rows = fuzz.run_chunk(tmp_path, 12, 4242, difficulty, 0.5, 0.25, workers=2)
    assert len(rows) == 12
    assert {r["type"] for r in rows} == {"math", "data_analysis", "search"}
    assert all(r["family"] and r["difficulty"] == difficulty for r in rows)
    assert not [r for r in rows if r["outcome"] == "wrong"], [r["instruction"] for r in rows if r["outcome"] == "wrong"]
    assert list(tmp_path.iterdir()) == []  # the chunk directory is removed


def test_reports(tmp_path):
    rows = fuzz.run_chunk(tmp_path, 9, 7, "easy", 0.2, 0.2, workers=2)
    out = tmp_path / "out"
    summary = fuzz.write_reports(out, {"n": len(rows)}, rows)
    assert summary["overall"]["total"] == 9
    data = json.loads((out / "fuzz.json").read_text(encoding="utf-8"))
    assert data["summary"]["by_type"].keys() == {"math", "data_analysis", "search"}
    assert "WRONG" in (out / "fuzz.md").read_text(encoding="utf-8")

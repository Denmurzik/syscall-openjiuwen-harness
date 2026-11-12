"""Expert 10, patch c (HARNESS_CONFTEST): the pytest stand-in loads a conftest.py of fixtures and helpers."""
from harness.solvers.testrun import run_in_dir

NO_PYTEST = "raise ImportError('no pytest in this image')\n"
BAD = "def area(w, h):\n    return w + h\n"
GOOD = "def area(w, h):\n    return w * h\n"
CONF = ("import pytest\n\n\n@pytest.fixture\ndef square():\n    return (3, 3)\n\n\n"
        "@pytest.fixture\ndef shapes(square):\n    return [square, (2, 5)]\n")
TESTS = ("import pytest\nfrom mylib import area\n\n\ndef test_square(square):\n    assert area(*square) == 9\n\n\n"
         "def test_all(shapes):\n    assert [area(*s) for s in shapes] == [9, 10]\n\n\n"
         "@pytest.mark.parametrize('w,h,e', [(1, 1, 1), (2, 3, 6)])\ndef test_p(w, h, e):\n    assert area(w, h) == e\n")
PLAIN = "from mylib import area\n\n\ndef test_square(square):\n    assert area(*square) == 9\n"


def _ws(tmp_path, files):
    ws = tmp_path / "ws"
    for name, body in {"pytest.py": NO_PYTEST, **files}.items():
        p = ws / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return ws


def test_fixtures_of_a_conftest_are_used(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": BAD, "conftest.py": CONF, "test_mylib.py": TESTS})
    rep = run_in_dir(ws, "test_mylib.py")
    assert not rep.ok and "test_square" in rep.failed and "<unknown>" not in rep.failed
    (ws / "mylib.py").write_text(GOOD, encoding="utf-8")
    assert run_in_dir(ws, "test_mylib.py").ok is True


def test_a_test_file_that_does_not_import_pytest_itself(tmp_path):
    ws = _ws(tmp_path, {"mylib.py": GOOD, "conftest.py": CONF, "test_mylib.py": PLAIN})
    assert run_in_dir(ws, "test_mylib.py").ok is True
    (ws / "mylib.py").write_text(BAD, encoding="utf-8")
    assert run_in_dir(ws, "test_mylib.py").ok is False


def test_the_test_files_own_fixture_overrides_the_conftest(tmp_path):
    tests = ("import pytest\nfrom mylib import area\n\n\n@pytest.fixture\ndef square():\n    return (2, 2)\n\n\n"
             "def test_square(square):\n    assert area(*square) == 4\n")
    ws = _ws(tmp_path, {"mylib.py": GOOD, "conftest.py": CONF, "test_mylib.py": tests})
    assert run_in_dir(ws, "test_mylib.py").ok is True


def test_conftests_from_the_root_down_to_the_test_file(tmp_path):
    inner = "import pytest\n\n\n@pytest.fixture\ndef shapes(square):\n    return [square, (4, 5)]\n"
    tests = "from mylib import area\n\n\ndef test_all(shapes):\n    assert [area(*s) for s in shapes] == [9, 20]\n"
    ws = _ws(tmp_path, {"mylib.py": GOOD, "conftest.py": CONF, "tests/conftest.py": inner, "tests/test_mylib.py": tests})
    assert run_in_dir(ws, "tests/test_mylib.py").ok is True


def test_hooks_plugins_and_a_broken_conftest_stay_red(tmp_path):
    for i, conf in enumerate((CONF + "\n\ndef pytest_collection_modifyitems(items):\n    items[:] = []\n",
                              "pytest_plugins = ['x']\n" + CONF, CONF + "\nraise RuntimeError('boom')\n",
                              "import pytest\n\n\n@pytest.fixture(params=[1, 2])\ndef square(request):\n    return (3, 3)\n")):
        ws = _ws(tmp_path / str(i), {"mylib.py": GOOD, "conftest.py": conf, "test_mylib.py": TESTS})
        assert run_in_dir(ws, "test_mylib.py").ok is False, conf


def test_the_switch_restores_the_red_run(tmp_path, monkeypatch):
    ws = _ws(tmp_path, {"mylib.py": GOOD, "conftest.py": CONF, "test_mylib.py": TESTS})
    monkeypatch.setenv("HARNESS_CONFTEST", "0")
    assert run_in_dir(ws, "test_mylib.py").ok is False


def test_unittest_files_beside_a_conftest_run_as_before(tmp_path):
    tests = ("import unittest\nfrom mylib import area\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n"
             "        self.assertEqual(area(2, 3), 6)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
    ws = _ws(tmp_path, {"mylib.py": GOOD, "conftest.py": "X = 1\n", "test_mylib.py": tests})
    assert run_in_dir(ws, "test_mylib.py").ok is True

"""Round 21 fix of patch c (HARNESS_CONFTEST): a conftest.py decides the stand-in only for a test file that takes a
fixture; a unittest file or one whose tests take no parameter runs as before the patch."""
from harness.solvers.testrun import run_in_dir

NO_PYTEST = "raise ImportError('no pytest in this image')\n"
GOOD = "def area(w, h):\n    return w * h\n"
BAD = "def area(w, h):\n    return w + h\n"
UNIT = ("import unittest\nfrom mylib import area\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n"
        "        self.assertEqual(area(2, 3), 6)\n")
PLAIN = "from mylib import area\n\n\ndef test_a():\n    assert area(2, 3) == 6\n"
CONFS = ("import pytest\n\n\ndef pytest_configure(config):\n    config.addinivalue_line('markers', 'slow: slow')\n",
         "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef env(monkeypatch):\n    monkeypatch.setenv('X', '1')\n",
         "import pytest\nimport numpy_missing_here\n",
         "import pytest\n\npytest.register_assert_rewrite('helpers')\n")


def _ws(tmp_path, files):
    ws = tmp_path / "ws"
    for name, body in {"pytest.py": NO_PYTEST, **files}.items():
        (ws / name).parent.mkdir(parents=True, exist_ok=True)
        (ws / name).write_text(body, encoding="utf-8")
    return ws


def test_unittest_and_plain_files_beside_a_pytest_conftest_run_as_before(tmp_path):
    for i, conf in enumerate(CONFS):
        for tests in (UNIT, PLAIN):
            ws = _ws(tmp_path / f"{i}{len(tests)}", {"mylib.py": GOOD, "conftest.py": conf, "test_mylib.py": tests})
            assert run_in_dir(ws, "test_mylib.py").ok is True, (conf, tests)
            (ws / "mylib.py").write_text(BAD, encoding="utf-8")
            assert run_in_dir(ws, "test_mylib.py").ok is False, (conf, tests)


def test_a_test_taking_a_fixture_still_gets_the_conftest(tmp_path):
    conf = "import pytest\n\n\n@pytest.fixture\ndef dims():\n    return (2, 3)\n"
    for tests in ("from mylib import area\n\n\ndef test_a(dims):\n    assert area(*dims) == 6\n",
                  "from mylib import area\n\n\nclass TestArea:\n    def test_a(self, dims):\n        assert area(*dims) == 6\n"):
        ws = _ws(tmp_path / str(len(tests)), {"mylib.py": GOOD, "conftest.py": conf, "test_mylib.py": tests})
        assert run_in_dir(ws, "test_mylib.py").ok is True
        (ws / "mylib.py").write_text(BAD, encoding="utf-8")
        assert run_in_dir(ws, "test_mylib.py").ok is False


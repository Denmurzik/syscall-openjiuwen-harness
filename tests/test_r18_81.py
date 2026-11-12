"""A test file that imports pytest runs in an image without pytest (expert 9, idea b): the runner's stand-in."""
from harness.solvers.testrun import run_tests

# makes `import pytest` fail inside the run, as in the image: the workspace is first on sys.path
NO_PYTEST = "raise ImportError('no pytest in this image')\n"
GOOD = ("def div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n\n\n"
        "def double(x):\n    return x * 2\n")
BAD = GOOD.replace("x * 2", "x + 2")
TESTS = '''import pytest
from mylib import div, double


def test_raises():
    with pytest.raises(ValueError, match="zero"):
        div(1, 0)


def test_raises_call_form():
    pytest.raises(ValueError, div, 1, 0)


def test_approx():
    assert div(1, 3) == pytest.approx(0.3333333, rel=1e-4)
    assert [div(1, 3), 1.0] == pytest.approx([0.33333333333, 1.0])


@pytest.mark.parametrize("x, want", [(1, 2), (2, 4), pytest.param(5, 10, id="five")])
def test_double(x, want):
    assert double(x) == want


@pytest.mark.parametrize("x", [0, 3])
def test_double_one_name(x):
    assert double(x) == 2 * x


@pytest.mark.slow
def test_marked():
    assert double(0) == 0


@pytest.mark.skip(reason="not now")
def test_skipped():
    assert False


@pytest.fixture
def numbers():
    return [1, 2, 3]


@pytest.fixture()
def doubled(numbers):
    yield [double(n) for n in numbers]


def test_fixture(doubled, numbers):
    assert doubled == [2, 4, 6] and numbers == [1, 2, 3]


class TestClass:
    @pytest.mark.parametrize("x", [1, 4])
    def test_method(self, x):
        assert double(x) == 2 * x

    def test_method_fixture(self, numbers):
        assert double(numbers[0]) == 2
'''


def _ws(tmp_path, lib, tests=TESTS, shim=True):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    if shim:
        (ws / "pytest.py").write_text(NO_PYTEST, encoding="utf-8")
    return ws


def test_pytest_file_green_without_pytest(tmp_path):
    r = run_tests(_ws(tmp_path, GOOD))
    assert r.ok, r.output


def test_pytest_file_red_names_the_failing_tests(tmp_path):
    r = run_tests(_ws(tmp_path, BAD))
    assert not r.ok
    assert {"test_double", "test_double_one_name", "test_fixture", "test_method", "test_method_fixture"} <= set(r.failed)
    assert "test_raises" not in r.failed and "test_approx" not in r.failed and "test_skipped" not in r.failed


def test_did_not_raise_is_red(tmp_path):
    lib = GOOD.replace("        raise ValueError('division by zero')\n", "        return 0\n")
    r = run_tests(_ws(tmp_path, lib))
    assert not r.ok and "test_raises" in r.failed and "test_raises_call_form" in r.failed


def test_unknown_fixture_stays_unrunnable(tmp_path):
    tests = "import pytest\nfrom mylib import double\n\n\ndef test_x(tmp_path):\n    assert double(1) == 2\n"
    assert not run_tests(_ws(tmp_path, GOOD, tests)).ok


def test_testcase_file_with_pytest_raises(tmp_path):
    tests = ("import unittest\nimport pytest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
             "    def test_a(self):\n        with pytest.raises(ValueError):\n            mylib.div(1, 0)\n"
             "        self.assertEqual(mylib.double(3), 6)\n")
    assert run_tests(_ws(tmp_path, GOOD, tests)).ok
    r = run_tests(_ws(tmp_path / "b", BAD, tests))
    assert not r.ok and r.failed == ["test_a"]


def test_plain_files_run_as_before(tmp_path):
    tests = "from mylib import double\n\n\ndef test_x():\n    assert double(3) == 6\n"
    assert run_tests(_ws(tmp_path, GOOD, tests, shim=False)).ok
    assert run_tests(_ws(tmp_path / "b", BAD, tests, shim=False)).failed == ["test_x"]

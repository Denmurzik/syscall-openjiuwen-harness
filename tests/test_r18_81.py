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


# round 18 fix round: where the stand-in cannot follow pytest the file is red, never green
LIB = GOOD + "\n\ndef big():\n    return 1000000.0\n"
BADLIB = BAD + "\n\ndef big():\n    return 1000000.5\n"


def _both(tmp_path, tests, shim=True):
    """(report on the right mylib, report on the broken one)"""
    return (run_tests(_ws(tmp_path / "good", LIB, tests, shim)), run_tests(_ws(tmp_path / "bad", BADLIB, tests, shim)))


def _green_red(tmp_path, tests, shim=True):
    good, bad = _both(tmp_path, tests, shim)
    return good.ok and not bad.ok


def test_approx_with_abs_alone_uses_abs_alone(tmp_path):
    tests = "import pytest\nfrom mylib import big\n\n\ndef test_big():\n    assert big() == pytest.approx(1e6, abs=0.01)\n"
    assert _green_red(tmp_path, tests)
    tests = ("import pytest\n\n\ndef test_inf():\n    assert float('inf') == pytest.approx(float('inf'))\n"
             "    assert 1.0 + 1e-7 == pytest.approx(1.0)\n    assert 1.0 + 1e-9 != pytest.approx(1.0, abs=1e-12)\n")
    assert run_tests(_ws(tmp_path / "inf", LIB, tests)).ok


def test_parametrize_on_a_class_runs_its_methods(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\ndef test_ident():\n    assert True\n\n\n"
             "@pytest.mark.parametrize('x', [1, 2])\nclass TestDouble:\n    def test_d(self, x):\n"
             "        assert double(x) == 2 * x\n")
    good, bad = _both(tmp_path, tests)
    assert good.ok and not bad.ok and bad.failed == ["test_d"]


def test_skip_and_xfail_on_a_class_apply_to_each_method(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\ndef test_ok():\n    assert double(3) == 6\n\n\n"
             "@pytest.mark.skip(reason='later')\nclass TestLater:\n    def test_a(self):\n        assert False\n\n\n"
             "@pytest.mark.xfail(strict=True)\nclass TestKnown:\n    def test_b(self):\n        assert double(3) == 7\n")
    assert run_tests(_ws(tmp_path, LIB, tests)).ok
    xpass = tests.replace("double(3) == 7", "double(3) == 6")
    assert run_tests(_ws(tmp_path / "x", LIB, xpass)).failed == ["test_b"]


def test_xfail_with_a_false_condition_runs_the_test(tmp_path):
    tests = ("import sys, pytest\nfrom mylib import double\n\n\n"
             "@pytest.mark.xfail(sys.platform == 'no-such-os', reason='only there')\n"
             "def test_double():\n    assert double(3) == 6\n")
    assert _green_red(tmp_path, tests)


def test_xfail_strict_and_not_strict(tmp_path):
    head = "import pytest\nfrom mylib import double\n\n\n"
    strict_xpass = head + "@pytest.mark.xfail(strict=True)\ndef test_k():\n    assert double(3) == 6\n"
    assert not run_tests(_ws(tmp_path / "a", LIB, strict_xpass)).ok
    loose_xpass = head + "@pytest.mark.xfail(reason='flaky')\ndef test_k():\n    assert double(3) == 6\n"
    assert run_tests(_ws(tmp_path / "b", LIB, loose_xpass)).ok
    loose_fail = head + "@pytest.mark.xfail\ndef test_k():\n    assert double(3) == 7\n"
    assert run_tests(_ws(tmp_path / "c", LIB, loose_fail)).ok
    wrong_raise = head + "@pytest.mark.xfail(raises=KeyError)\ndef test_k():\n    assert double(3) == 7\n"
    assert not run_tests(_ws(tmp_path / "d", LIB, wrong_raise)).ok
    ws = _ws(tmp_path / "e", LIB, loose_xpass)
    (ws / "pytest.ini").write_text("[pytest]\nxfail_strict = true\n", encoding="utf-8")
    assert not run_tests(ws).ok


def test_skipif_with_a_string_condition(tmp_path):
    head = "import pytest\nfrom mylib import double\n\n\n"
    assert _green_red(tmp_path / "a", head + "@pytest.mark.skipif('sys.version_info < (3, 0)', reason='py2')\n"
                      "def test_d():\n    assert double(3) == 6\n")
    assert run_tests(_ws(tmp_path / "b", BADLIB, head + "@pytest.mark.skipif('sys.version_info > (3, 0)', reason='x')\n"
                         "def test_d():\n    assert double(3) == 6\n")).ok
    assert not run_tests(_ws(tmp_path / "c", LIB, head + "@pytest.mark.skipif('no_such_name', reason='x')\n"
                             "def test_d():\n    assert double(3) == 6\n")).ok


def test_yield_fixture_wraps_the_test(tmp_path):
    tests = ("import os, pytest, tempfile\nfrom mylib import double\nEV = []\n\n\n@pytest.fixture\ndef d():\n"
             "    with tempfile.TemporaryDirectory() as t:\n        EV.append('setup')\n        try:\n            yield t\n"
             "        finally:\n            EV.append('teardown')\n\n\n"
             "def test_a(d):\n    assert os.path.isdir(d) and EV == ['setup']\n    assert double(3) == 6\n\n\n"
             "def test_b():\n    assert EV == ['setup', 'teardown']\n")
    assert _green_red(tmp_path, tests)


def test_error_in_fixture_teardown_fails_the_test(tmp_path):
    tests = ("import pytest\n\n\n@pytest.fixture\ndef r():\n    yield 1\n    raise RuntimeError('teardown')\n\n\n"
             "def test_r(r):\n    assert r == 1\n")
    assert run_tests(_ws(tmp_path, LIB, tests)).failed == ["test_r"]


def test_autouse_and_usefixtures_run(tmp_path):
    autouse = ("import pytest\nfrom mylib import double\nSEEN = []\n\n\n@pytest.fixture(autouse=True)\ndef check():\n"
               "    yield\n    assert SEEN and all(double(x) == 2 * x for x in SEEN)\n\n\n"
               "def test_a():\n    SEEN.append(3)\n")
    assert _green_red(tmp_path / "a", autouse)
    use = ("import pytest\nfrom mylib import double\nSTATE = {}\n\n\n@pytest.fixture\ndef ready():\n"
           "    STATE['x'] = double(3)\n\n\n@pytest.mark.usefixtures('ready')\ndef test_u():\n    assert STATE['x'] == 6\n")
    assert _green_red(tmp_path / "b", use)


def test_fixture_is_one_instance_per_test_and_module_scope_is_shared(tmp_path):
    tests = ("import pytest\n\n\n@pytest.fixture\ndef store():\n    return []\n\n\n@pytest.fixture\n"
             "def filled(store):\n    store.append(1)\n    return store\n\n\n"
             "def test_same(filled, store):\n    assert filled is store and store == [1]\n\n\n"
             "N = []\n\n\n@pytest.fixture(scope='module')\ndef once():\n    N.append(1)\n    return len(N)\n\n\n"
             "def test_1(once):\n    assert once == 1\n\n\ndef test_2(once):\n    assert once == 1 and N == [1]\n")
    assert run_tests(_ws(tmp_path, LIB, tests)).ok


def test_fixture_with_parametrize(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\n@pytest.fixture\ndef k():\n    return 2\n\n\n"
             "@pytest.mark.parametrize('x', [1, 3])\ndef test_double(x, k):\n    assert double(x) == k * x\n")
    assert _green_red(tmp_path, tests)


def test_param_marks_skip_and_xfail(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\n@pytest.mark.parametrize('x, want', [(1, 2), "
             "pytest.param(3, 7, marks=pytest.mark.xfail), pytest.param(4, 0, marks=[pytest.mark.skip])])\n"
             "def test_d(x, want):\n    assert double(x) == want\n")
    assert _green_red(tmp_path, tests)


def test_pytest_fail_is_not_swallowed_by_except_exception(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\ndef test_f():\n    try:\n        if double(3) != 6:\n"
             "            pytest.fail('wrong')\n    except Exception:\n        pytest.skip('no')\n")
    assert _green_red(tmp_path, tests)


def test_static_and_class_method_tests(tmp_path):
    tests = ("from mylib import double\n\n\nclass TestS:\n    @staticmethod\n    def test_s():\n"
             "        assert double(3) == 6\n\n    def test_m(self):\n        assert double(1) == 2\n")
    assert _green_red(tmp_path / "plain", tests, shim=False)   # no pytest: as before the stand-in
    with_pytest = "import pytest\n" + tests + "\n    @classmethod\n    def test_c(cls):\n        assert double(3) == 6\n"
    assert _green_red(tmp_path / "pt", with_pytest)


def test_guarded_import_gets_no_stand_in(tmp_path):
    tests = ("import unittest\ntry:\n    import pytest\nexcept ImportError:\n    pytest = None\nimport mylib\n\n\n"
             "class T(unittest.TestCase):\n    @unittest.skipIf(pytest is None, 'no pytest')\n    def test_w(self):\n"
             "        with pytest.warns(UserWarning):\n            pass\n\n    def test_d(self):\n"
             "        self.assertEqual(mylib.double(3), 6)\n")
    assert _green_red(tmp_path, tests)


def test_what_the_stand_in_cannot_follow_is_red(tmp_path):
    head = "import pytest, unittest\nfrom mylib import double\n\n\n"
    cases = {
        "indirect": head + "@pytest.fixture\ndef v():\n    return 1\n\n\n"
                    "@pytest.mark.parametrize('v', [1], indirect=True)\ndef test_v(v):\n    assert v\n",
        "params": head + "@pytest.fixture(params=[1, 2])\ndef v(request):\n    return request.param\n\n\n"
                  "def test_v(v):\n    assert v\n",
        "cycle": head + "@pytest.fixture\ndef a(b):\n    return 1\n\n\n@pytest.fixture\ndef b(a):\n    return 1\n\n\n"
                 "def test_x(a):\n    assert a\n",
        "testcase_mark": head + "class T(unittest.TestCase):\n    @pytest.mark.skip(reason='x')\n    def test_a(self):\n"
                         "        self.assertTrue(True)\n",
        "setup_module": head + "def setup_module():\n    pass\n\n\ndef test_a():\n    assert double(3) == 6\n",
    }
    for name, tests in cases.items():
        assert not run_tests(_ws(tmp_path / name, LIB, tests)).ok, name
    ws = _ws(tmp_path / "conftest", LIB, head + "def test_a():\n    assert double(3) == 6\n")
    # expert 10: a conftest.py of fixtures is followed, one with collection hooks is not
    (ws / "conftest.py").write_text("def pytest_collection_modifyitems(items):\n    pass\n", encoding="utf-8")
    assert not run_tests(ws).ok


def test_module_pytestmark_applies(tmp_path):
    tests = ("import pytest\nfrom mylib import double\npytestmark = [pytest.mark.slow, pytest.mark.skipif(True, "
             "reason='x')]\n\n\ndef test_d():\n    assert double(2) == 5\n")
    assert run_tests(_ws(tmp_path, LIB, tests)).ok   # every test skipped, as pytest


def test_nested_test_class_makes_the_file_red(tmp_path):
    tests = ("import pytest\nfrom mylib import double\n\n\nclass TestOuter:\n    def test_a(self):\n"
             "        assert double(0) == 0 or True\n\n    class TestInner:\n        def test_b(self):\n"
             "            assert double(3) == 6\n")
    good, bad = _both(tmp_path, tests)
    assert not good.ok and not bad.ok   # pytest would run test_b too: not followed, never green

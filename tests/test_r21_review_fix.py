"""Round 21 fixes of patch a (HARNESS_REVIEW_PROBES): what counts as a hang of the file, a named exception, a
default left out, and probe values not tuned to one set."""
import os

import pytest

from harness import review as r

UT = "import unittest\n"


def accepted(src, tests, name, new, other=None):
    return bool(r._verdict(src, tests, name, new, other or new)[0])


TIME_LEFT = '''def time_left(elapsed, limit):
    """Seconds left before `limit`. Raises TimeoutError once `elapsed` is past `limit`."""
    if elapsed > limit:
        raise TimeoutError("deadline passed")
    return limit - elapsed
'''
TIME_LEFT_TESTS = UT + '''from mylib import time_left


class T(unittest.TestCase):
    def test_left(self):
        self.assertEqual(time_left(3, 10), 7)
'''


def test_a_timeout_error_the_function_raises_is_no_hang():
    new = TIME_LEFT.replace('    if elapsed > limit:\n        raise TimeoutError("deadline passed")\n    return limit - elapsed',
                            "    return max(0, limit - elapsed)")
    assert not accepted(TIME_LEFT, TIME_LEFT_TESTS, "time_left", new)


def test_the_review_child_shows_its_own_alarm_apart_from_a_raise():
    src = "def f(x):\n    if x == 1:\n        raise TimeoutError('mine')\n    try:\n        while x == 2:\n            pass\n" \
          "    except Exception:\n        return 'swallowed'\n    return x\n"
    out = r._expr_behaviour(src, "f", "", ["(1,)", "(3,)"], timeout=20)
    assert out == ["<exc:TimeoutError>", "3"]
    if os.name != "nt":   # the alarm needs SIGALRM; the function's `except Exception` does not swallow it
        assert r._expr_behaviour(src, "f", "", ["(2,)"], timeout=20, alarm=0.3) == ["<exc:ProbeAlarm>"]

"""Expert 10, patch a (HARNESS_REVIEW_PROBES): the review's trial inputs reach string formats, Decimal/date
arguments, parameters with defaults, hangs and a named exception raised in place of another one."""
import os

import pytest

from harness import review as r

UT = "import unittest\n"


def accepted(src, tests, name, new):
    return bool(r._verdict(src, tests, name, new, new)[0])


TO_UTC = '''from datetime import datetime, timedelta


def to_utc(stamp):
    """Parse 'YYYY-MM-DDTHH:MM+HH:MM' (or with -HH:MM) and return a naive datetime in UTC."""
    body, sign, off = stamp[:16], stamp[16], stamp[17:]
    delta = timedelta(hours=int(off[:2]), minutes=int(off[3:]))
    local = datetime.strptime(body, "%Y-%m-%dT%H:%M")
    return local - delta
'''
TO_UTC_NEW = '''def to_utc(stamp):
    """Parse 'YYYY-MM-DDTHH:MM+HH:MM' (or with -HH:MM) and return a naive datetime in UTC."""
    body, sign, off = stamp[:16], stamp[16], stamp[17:]
    delta = timedelta(hours=int(off[:2]), minutes=int(off[3:]))
    local = datetime.strptime(body, "%Y-%m-%dT%H:%M")
    return local - delta if sign == "+" else local + delta
'''
TO_UTC_TESTS = UT + '''from datetime import datetime
from mylib import to_utc


class T(unittest.TestCase):
    def test_plus(self):
        self.assertEqual(to_utc("2024-03-10T12:00+02:00"), datetime(2024, 3, 10, 10, 0))
'''

AGE = '''def parse_age(text):
    """Returns an int between 0 and 150 inclusive. Raises ValueError when the text is not an integer or the
    value is outside that range."""
    return int(text)
'''
AGE_NEW = '''def parse_age(text):
    """Returns an int between 0 and 150 inclusive. Raises ValueError when the text is not an integer or the
    value is outside that range."""
    value = int(text)
    if not 0 <= value <= 150:
        raise ValueError(text)
    return value
'''
AGE_TESTS = UT + '''from mylib import parse_age


class T(unittest.TestCase):
    def test_age(self):
        self.assertEqual(parse_age(" 42 "), 42)
'''

TAG = '''def add_tag(tag, tags=[]):
    """Return a list made of `tags` followed by `tag`; add_tag('x') without a list always returns ['x']."""
    tags.append(tag)
    return tags
'''
TAG_NEW = '''def add_tag(tag, tags=None):
    """Return a list made of `tags` followed by `tag`; add_tag('x') without a list always returns ['x']."""
    return list(tags or []) + [tag]
'''
TAG_TESTS = UT + '''from mylib import add_tag


class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add_tag("b", ["a"]), ["a", "b"])
'''

TAX = '''from decimal import Decimal


def add_tax(net, rate_percent):
    """Gross amount, a Decimal with two places, rounded half up (0.125 becomes 0.13)."""
    net = Decimal(net)
    return round(net + net * Decimal(rate_percent) / 100, 2)
'''
TAX_NEW = '''def add_tax(net, rate_percent):
    """Gross amount, a Decimal with two places, rounded half up (0.125 becomes 0.13)."""
    net = Decimal(net)
    return (net + net * Decimal(rate_percent) / 100).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
'''
TAX_TESTS = UT + '''from decimal import Decimal

from mylib import add_tax


class T(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(add_tax(Decimal("10.00"), Decimal("20")), Decimal("12.00"))
'''

RATIO = '''def parse_ratio(text):
    """'3:4' -> 0.75. Raises ValueError when the second number is zero or the text has no ':'."""
    a, b = text.split(":")
    return int(a) / int(b)
'''
RATIO_NEW = '''def parse_ratio(text):
    """'3:4' -> 0.75. Raises ValueError when the second number is zero or the text has no ':'."""
    a, b = text.split(":")
    if int(b) == 0:
        raise ValueError(text)
    return int(a) / int(b)
'''
RATIO_TESTS = UT + '''from mylib import parse_ratio


class T(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(parse_ratio("3:4"), 0.75)
'''

CASES = [(TO_UTC, TO_UTC_TESTS, "to_utc", TO_UTC_NEW), (AGE, AGE_TESTS, "parse_age", AGE_NEW),
         (TAG, TAG_TESTS, "add_tag", TAG_NEW), (TAX, TAX_TESTS, "add_tax", TAX_NEW),
         (RATIO, RATIO_TESTS, "parse_ratio", RATIO_NEW)]


@pytest.mark.parametrize("src,tests,name,new", CASES, ids=[c[2] for c in CASES])
def test_right_rewrite_is_accepted(src, tests, name, new):
    assert accepted(src, tests, name, new)


@pytest.mark.parametrize("src,tests,name,new", CASES, ids=[c[2] for c in CASES])
def test_switch_off_keeps_the_file(src, tests, name, new, monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_PROBES", "0")
    assert not accepted(src, tests, name, new)


def test_string_variants():
    v = r._str_variants("2024-03-10T12:00+02:00")
    assert "2024-03-10T12:00-02:00" in v and len(v) <= 40
    v = r._str_variants(" 42 ")
    assert {" 151 ", " -1 ", "42", "999"} <= set(v)
    assert r._str_variants("hello") == [] and r._str_variants("") == []


def test_a_rewrite_equal_in_behaviour_is_no_evidence():
    same = TO_UTC.split("\n\n\n", 1)[1].replace("return local - delta", "result = local - delta\n    return result")
    assert not accepted(TO_UTC, TO_UTC_TESTS, "to_utc", same)


def test_two_rewrites_that_differ_are_kept():
    other = TO_UTC_NEW.replace('local + delta', 'local + delta + delta')
    got, disagree = r._verdict(TO_UTC, TO_UTC_TESTS, "to_utc", TO_UTC_NEW, other)
    assert got == [] and disagree


def test_an_exception_the_docstring_does_not_name_is_no_evidence():
    src = RATIO.replace("Raises ValueError when the second number is zero or the text has no ':'.", "Ratio as a float.")
    new = RATIO_NEW.replace("Raises ValueError when the second number is zero or the text has no ':'.", "Ratio as a float.")
    assert not accepted(src, RATIO_TESTS, "parse_ratio", new)


def test_a_swap_without_an_explicit_raise_is_not_stored():
    new = RATIO_NEW.replace("        raise ValueError(text)", "        return int('x')")
    assert not accepted(RATIO, RATIO_TESTS, "parse_ratio", new)


def test_a_default_that_is_no_bug_is_kept():
    src = 'def join(a, sep=", "):\n    """Join the items."""\n    return sep.join(a)\n'
    new = 'def join(a, sep=", "):\n    """Join the items."""\n    return sep.join(list(a))\n'
    tests = UT + 'from mylib import join\n\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(join(["a", "b"], "-"), "a-b")\n'
    assert not accepted(src, tests, "join", new)


HANG = '''def bin_search(xs, target):
    """Index of target in the sorted list xs, or -1 when absent."""
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1
'''
HANG_TESTS = UT + '''from mylib import bin_search


class T(unittest.TestCase):
    def test_found(self):
        self.assertEqual(bin_search([1, 3, 5, 7], 5), 2)
'''


@pytest.mark.skipif(os.name == "nt", reason="the probe's per-input alarm needs SIGALRM")
def test_a_hang_of_the_file_is_evidence():
    new = HANG.replace("hi = mid\n", "hi = mid - 1\n")
    assert accepted(HANG, HANG_TESTS, "bin_search", new)
    assert not accepted(HANG, HANG_TESTS, "bin_search", HANG.replace("lo, hi = 0", "lo, hi = 0 + 0"))


def test_expression_battery_is_capped_and_keeps_the_visible_call():
    head, args = r._visible_exprs(TAX_TESTS, "add_tax")
    battery = r._expr_battery(args)
    assert head == "from decimal import Decimal" and battery[0] == '(Decimal("10.00"), Decimal("20"),)'
    assert len(battery) <= r.BATTERY_CAP
    assert r._visible_exprs(AGE_TESTS, "parse_age") is None      # literal arguments: the usual battery

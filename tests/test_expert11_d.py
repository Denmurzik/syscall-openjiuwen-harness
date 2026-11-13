"""Expert 11: HARNESS_FULL_SMALL (a), HARNESS_KB_EXTRAS (b), HARNESS_REVIEW_RETRY (c), HARNESS_REVIEW_LONG (d)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import llm_routes as L
from harness import review as r
from harness.tasks import Task

UT = "import unittest\n"


CACHE = '''_cache = {}
PRICES = {"pen": 120, "book": 950}


def line_total(item, qty, discount_pct=0):
    """Price in cents of qty items after a percentage discount (integer cents, rounded down). Cached."""
    key = (item, qty)
    if key not in _cache:
        _cache[key] = PRICES[item] * qty * (100 - discount_pct) // 100
    return _cache[key]
'''
CACHE_NEW = '''def line_total(item, qty, discount_pct=0):
    """Price in cents of qty items after a percentage discount (integer cents, rounded down). Cached."""
    key = (item, qty, discount_pct)
    if key not in _cache:
        _cache[key] = PRICES[item] * qty * (100 - discount_pct) // 100
    return _cache[key]
'''
CACHE_T = UT + ("from mylib import line_total\n\n\nclass T(unittest.TestCase):\n    def test(self):\n"
                "        self.assertEqual(line_total('pen', 3), 360)\n")


def test_long_inputs_follow_the_visible_arguments():
    out = r._long_inputs(CACHE, "line_total", ("pen", 3))
    assert out and all(t[:2] == ("pen", 3) and len(t) == 3 and t[2] != 0 for t in out)
    assert r._long_inputs(CACHE, "line_total", ("pen", 3, 5)) == []          # nothing is left at its default
    assert r._long_inputs("def f(a, *rest, k=1):\n    return a\n", "f", (1,)) == []
    assert r._long_inputs("def f(a, b=None, c=[]):\n    return a\n", "f", (1,)) == []
    two = r._long_inputs("def f(a, b=1, c=True):\n    return a\n", "f", (1,))
    assert (1, 2) in two and (1, 1, False) in two


def test_wrong_cache_key_is_evidence():
    assert r._verdict(CACHE, CACHE_T, "line_total", CACHE_NEW, CACHE_NEW)[0]


def test_long_switch_off(monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_LONG", "0")
    assert not r._verdict(CACHE, CACHE_T, "line_total", CACHE_NEW, CACHE_NEW)[0]


def test_a_rewrite_that_only_restyles_is_still_no_evidence():
    same = CACHE_NEW.replace("key = (item, qty, discount_pct)", "key = (item, qty)")
    assert not r._verdict(CACHE, CACHE_T, "line_total", same, same)[0]


ROUND = '''def price(amount, digits=2):
    """Round amount to `digits` decimal places."""
    return round(amount, digits)
'''
ROUND_T = UT + ("from mylib import price\n\n\nclass T(unittest.TestCase):\n    def test(self):\n"
                "        self.assertEqual(price(2.5), 2.5)\n")


def test_a_right_function_with_a_default_is_kept():
    other = ROUND.replace("return round(amount, digits)", "return float(round(amount, digits))")
    assert not r._verdict(ROUND, ROUND_T, "price", other, other)[0]

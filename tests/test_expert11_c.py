"""Expert 11: HARNESS_FULL_SMALL (a), HARNESS_KB_EXTRAS (b), HARNESS_REVIEW_RETRY (c), HARNESS_REVIEW_LONG (d)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness import llm_routes as L
from harness import review as r
from harness.tasks import Task

UT = "import unittest\n"


class _Client:
    def __init__(self, replies):
        self.replies, self.limits = list(replies), []

    async def chat(self, model, msgs, max_tokens, **kw):
        self.limits.append(max_tokens)
        text, n = self.replies.pop(0)
        return SimpleNamespace(text=text, completion_tokens=n, prompt_tokens=10, truncated=False)


def test_cut_reply_is_asked_again_with_more_room():
    c = _Client([("FUNC f\ndef f(", r.REVIEW_MAX_TOKENS), ("NONE", 3)])
    reply, cut = asyncio.run(r._ask(c, "m", []))
    assert reply.text == "NONE" and not cut and c.limits == [r.REVIEW_MAX_TOKENS, r.REVIEW_RETRY_TOKENS]


def test_second_cut_reply_ends_the_review():
    c = _Client([("x", r.REVIEW_MAX_TOKENS), ("y", r.REVIEW_RETRY_TOKENS)])
    assert asyncio.run(r._ask(c, "m", []))[1] is True and len(c.limits) == 2


def test_retry_switch_off(monkeypatch):
    monkeypatch.setenv("HARNESS_REVIEW_RETRY", "0")
    c = _Client([("x", r.REVIEW_MAX_TOKENS), ("NONE", 3)])
    assert asyncio.run(r._ask(c, "m", []))[1] is True and c.limits == [r.REVIEW_MAX_TOKENS]


def test_whole_reply_is_one_call():
    c = _Client([("NONE", 3)])
    assert asyncio.run(r._ask(c, "m", []))[1] is False and len(c.limits) == 1

"""The review of a package module below the root (round 19, task 85, HARNESS_DEEP_TESTS)."""
import asyncio

from harness import review
from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import ChatResult
from harness.tasks import Task

from tests.test_r19_85 import DISC, DISC_FIX, DISC_TESTS, _ws


class ReviewFake:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def chat(self, model, messages, max_tokens, **extra):
        self.calls.append(model)
        return ChatResult(text=self.reply, prompt_tokens=10, completion_tokens=1, cost=2)


def test_review_repairs_a_silent_bug_in_a_package_module(tmp_path):
    t = _ws(tmp_path, {"shop/__init__.py": "", "shop/pricing.py": DISC, "tests/unit/test_p.py": DISC_TESTS})
    (t.dir / "answer.txt").write_text("DONE", encoding="utf-8")
    c = ReviewFake("FUNC bulk_price\n" + DISC_FIX)
    changed, note = asyncio.run(review.review(t, c))
    assert changed == ["bulk_price"] and c.calls == [MODEL_FAST, MODEL_ALT], note
    assert "qty >= 10" in (t.workspace / "shop" / "pricing.py").read_text(encoding="utf-8")

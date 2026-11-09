import asyncio
import os

import pytest

from harness.budget import TokenLedger
from harness.llm import LLMClient

pytestmark = pytest.mark.live


@pytest.mark.skipif(not (os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL")),
                    reason="LLM_API_KEY / LLM_BASE_URL not set")
def test_live_noreason_usage():
    c = LLMClient(os.environ["LLM_BASE_URL"],
                  os.environ["LLM_API_KEY"], TokenLedger(100000, 1000))
    r = asyncio.run(c.chat("qwen3.6-35b-a3b-noreason", [{"role": "user", "content": "Say hi"}], 32))
    assert r.prompt_tokens > 0 and r.completion_tokens > 0
    assert c.ledger.spent == r.cost

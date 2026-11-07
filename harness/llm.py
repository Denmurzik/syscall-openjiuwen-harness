"""LLM client on top of openjiuwen: budget accounting, retries, concurrency."""
import asyncio
import logging
import re
from dataclasses import dataclass

from harness.budget import TokenLedger
from harness.config import LLM_CONCURRENCY, LLM_TIMEOUT, TIMEOUT_CHARGE

logging.getLogger("openjiuwen").setLevel(logging.WARNING)

RETRY_DELAYS = (2, 4, 8)
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


class BudgetExhausted(Exception):
    pass


class LLMUnavailable(Exception):
    pass


def _chain(exc: BaseException):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


_STATUS_RE = re.compile(r"(?:Error code:|error|status(?: code)?:?)\s*(\d{3})", re.I)


def _status_of(exc: BaseException) -> int | None:
    """HTTP status of a (possibly openjiuwen-wrapped) exception, if any.

    openjiuwen re-raises provider errors as ModelError `from e`, so the openai
    exception (with .status_code) sits on the __cause__ chain.
    """
    for e in _chain(exc):
        for attr in ("status_code", "status"):
            v = getattr(e, attr, None)
            if isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599:
                return v
        resp = getattr(e, "response", None)
        v = getattr(resp, "status_code", None)
        if isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599:
            return v
    for e in _chain(exc):
        m = _STATUS_RE.search(str(e))
        if m:
            return int(m.group(1))
    return None


def _is_timeout(exc: BaseException) -> bool:
    return any(
        isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in type(e).__name__
        for e in _chain(exc)
    )


def _is_connect_error(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in _chain(exc)}
    return bool(names & {"ConnectError", "ConnectTimeout", "APIConnectionError"}) and not _is_read_timeout(exc)


def _is_read_timeout(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in _chain(exc)}
    return bool(names & {"ReadTimeout", "APITimeoutError", "TimeoutError"}) and "ConnectError" not in names \
        and "ConnectTimeout" not in names


class LLMClient:
    def __init__(self, base_url: str, api_key: str, ledger: TokenLedger,
                 max_concurrency: int = LLM_CONCURRENCY, timeout: float = LLM_TIMEOUT):
        self.base_url = base_url
        self.api_key = api_key
        self.ledger = ledger
        self.timeout = timeout
        self._max_concurrency = max_concurrency
        self._sem: asyncio.Semaphore | None = None
        self._models: dict = {}
        self._failures = 0

    def _model(self, name: str):
        m = self._models.get(name)
        if m is None:
            from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
            cc = ModelClientConfig(client_provider="OpenAI", api_key=self.api_key,
                                   api_base=self.base_url, verify_ssl=True)
            mc = ModelRequestConfig(model=name, temperature=0.0)
            m = self._models[name] = Model(cc, mc)
        return m

    def _fail(self) -> None:
        self._failures += 1
        if self._failures >= MAX_CONSECUTIVE_FAILURES:
            self.ledger.mark_exhausted()
            raise BudgetExhausted(f"{self._failures} consecutive failed LLM calls")

    async def chat(self, model: str, messages: list[dict], max_tokens: int, **extra) -> ChatResult:
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrency)
        async with self._sem:
            return await self._chat(model, messages, max_tokens, extra)

    async def _chat(self, model, messages, max_tokens, extra) -> ChatResult:
        m = self._model(model)
        attempt = 0
        while True:
            if self.ledger.exhausted:
                raise BudgetExhausted("ledger exhausted")
            try:
                r = await asyncio.wait_for(
                    m.invoke(messages, max_tokens=max_tokens, **extra), self.timeout)
            except Exception as exc:
                status = _status_of(exc)
                if status in (402, 403):
                    self.ledger.mark_exhausted()
                    raise BudgetExhausted(f"HTTP {status}") from exc
                if status is None and _is_timeout(exc) and not _is_connect_error(exc):
                    # request may have been served and billed: charge, no retry
                    self.ledger.charge(0, TIMEOUT_CHARGE)
                    self._fail()
                    raise LLMUnavailable("timeout") from exc
                retryable = status == 429 or (status is None and _is_connect_error(exc))
                if retryable and attempt < len(RETRY_DELAYS):
                    await asyncio.sleep(RETRY_DELAYS[attempt])
                    attempt += 1
                    continue
                if status is not None and status >= 500:
                    self.ledger.charge(0, TIMEOUT_CHARGE)
                self._fail()
                raise LLMUnavailable(f"{type(exc).__name__}: {exc}"[:300]) from exc
            self._failures = 0
            u = r.usage_metadata
            p = int(getattr(u, "input_tokens", 0) or 0)
            c = int(getattr(u, "output_tokens", 0) or 0)
            cost = self.ledger.charge(p, c)
            return ChatResult(text=r.content or "", prompt_tokens=p, completion_tokens=c, cost=cost)

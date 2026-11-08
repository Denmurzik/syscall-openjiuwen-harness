"""LLM client on top of openjiuwen: budget accounting, retries, concurrency."""
import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass

from harness.budget import TokenLedger
from harness import config
from harness.config import LLM_CONCURRENCY, TIMEOUT_CHARGE

logger = logging.getLogger(__name__)

def _quiet_log_config() -> dict:
    base = os.environ.get("HARNESS_LOG_DIR") or os.path.join(tempfile.gettempdir(), "harness")
    return {
        "level": "WARNING",
        "log_path": os.path.join(base, "openjiuwen-logs"),
        "output": ["file"],
        "interface_output": ["file"],
        "performance_output": ["file"],
    }


def _quiet_openjiuwen_logs() -> None:
    """openjiuwen logs full prompts at INFO to stdout; keep only WARNING+ in files.

    Never raises: a logging misconfiguration must not break the run.
    """
    try:
        from openjiuwen.core.common.logging.log_config import configure_log_config
        configure_log_config(_quiet_log_config())
    except Exception as exc:
        logger.warning("could not configure openjiuwen logging: %s", exc)
    logging.getLogger("openjiuwen").setLevel(logging.WARNING)


_logs_quieted = False

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


_STATUS_RE = re.compile(r"(?:Error code:|error|status(?: code)?:?)\s*\b(\d{3})\b", re.I)


def _valid_status(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599


def _status_of(exc: BaseException) -> int | None:
    """HTTP status of a (possibly openjiuwen-wrapped) exception, if any.

    openjiuwen re-raises provider errors as ModelError `from e`, so the openai
    exception (with .status_code) sits on the __cause__ chain.
    """
    for e in _chain(exc):
        for attr in ("status_code", "status"):
            v = getattr(e, attr, None)
            if _valid_status(v):
                return v
        v = getattr(getattr(e, "response", None), "status_code", None)
        if _valid_status(v):
            return v
    for e in _chain(exc):
        m = _STATUS_RE.search(str(e))
        if m:
            return int(m.group(1))
    return None


def _names(exc: BaseException) -> set[str]:
    return {type(e).__name__ for e in _chain(exc)}


def _is_connect_error(exc: BaseException) -> bool:
    """Request provably never reached the server (safe to retry)."""
    return bool(_names(exc) & {"ConnectError", "ConnectTimeout"})


def _is_transport_failure(exc: BaseException) -> bool:
    """Timeout or connection broken after send: the request may have been billed."""
    if any(isinstance(e, (asyncio.TimeoutError, TimeoutError)) for e in _chain(exc)):
        return True
    return any(n.endswith("Timeout") or n.endswith("TimeoutError")
               or n in {"APIConnectionError", "RemoteProtocolError", "ReadError", "WriteError",
                        "NetworkError", "TransportError"}
               for n in _names(exc))


class LLMClient:
    def __init__(self, base_url: str, api_key: str, ledger: TokenLedger,
                 max_concurrency: int = LLM_CONCURRENCY, timeout: float | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.ledger = ledger
        self.timeout = config.llm_timeout() if timeout is None else timeout
        self._max_concurrency = max_concurrency
        self._sem: asyncio.Semaphore | None = None
        self._models: dict = {}
        self._failures = 0
        self.structured_ok = True

    def _model(self, name: str):
        m = self._models.get(name)
        if m is None:
            global _logs_quieted
            if not _logs_quieted:
                _logs_quieted = True
                _quiet_openjiuwen_logs()
            from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
            cc = ModelClientConfig(client_provider="OpenAI", api_key=self.api_key,
                                   api_base=self.base_url, verify_ssl=True)
            mc = ModelRequestConfig(model=name, temperature=0.0)
            m = self._models[name] = Model(cc, mc)
        return m

    def _count_failure(self) -> None:
        """Count a failure that suggests the proxy is gone; 3 in a row exhaust the run."""
        self._failures += 1
        if self._failures >= MAX_CONSECUTIVE_FAILURES:
            self.ledger.mark_exhausted()
            raise BudgetExhausted(f"{self._failures} consecutive failed LLM calls")

    async def chat(self, model: str, messages: list[dict], max_tokens: int, **extra) -> ChatResult:
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        if self._sem is None:
            # bound to the running loop on first use; one loop per client in production
            self._sem = asyncio.Semaphore(self._max_concurrency)
        async with self._sem:
            return await self._chat(model, messages, max_tokens, extra)

    async def _chat(self, model, messages, max_tokens, extra) -> ChatResult:
        m = self._model(model)
        attempt = 0
        extra = dict(extra)
        stripped = False  # structured params dropped for this call as a trial
        while True:
            if not self.structured_ok or stripped:
                extra.pop("response_format", None)
                extra.pop("extra_body", None)
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
                if (status == 400 and self.structured_ok
                        and ("response_format" in extra or "extra_body" in extra)):
                    # proxy rejects structured output: retry once without it, remember
                    # the flag flips only if the retry succeeds (proves the 400 was about them)
                    logger.warning("HTTP 400 with structured output; retrying without it")
                    stripped = True
                    continue
                connect = status is None and _is_connect_error(exc)
                delays = config.retry_delays()
                if (status == 429 or connect) and attempt < len(delays):
                    await asyncio.sleep(delays[attempt])
                    attempt += 1
                    continue
                msg = f"{type(exc).__name__}: {exc}"[:300]
                if connect:
                    self._count_failure()
                elif status is not None and status >= 500 or (status is None and _is_transport_failure(exc)):
                    # may have been served and billed: charge a flat amount, never retry
                    self.ledger.charge(0, TIMEOUT_CHARGE)
                    self._count_failure()
                # other 4xx (400/404/422, exhausted 429): deterministic, not a dead proxy
                raise LLMUnavailable(msg) from exc
            self._failures = 0
            if stripped:
                self.structured_ok = False
            u = getattr(r, "usage_metadata", None)
            if u is None:
                p = len(str(messages)) // 4
                c = max_tokens
                logger.warning("no usage_metadata from %s; charging estimate %s/%s", model, p, c)
            else:
                p = int(getattr(u, "input_tokens", 0) or 0)
                c = int(getattr(u, "output_tokens", 0) or 0)
            cost = self.ledger.charge(p, c)
            return ChatResult(text=r.content or "", prompt_tokens=p, completion_tokens=c, cost=cost)

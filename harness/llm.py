"""LLM client on top of openjiuwen: budget accounting, retries, concurrency."""
import asyncio
import logging
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass

from harness.budget import TokenLedger
from harness import config
from harness.config import LLM_CONCURRENCY, TIMEOUT_CHARGE
from harness.trajectory import record_call

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

# a request or a model that fails this often while something else answers is about itself, not about the proxy
MODEL_FAILING_AFTER = 2
_PING = [{"role": "user", "content": "ping"}]
_MODEL_MISSING_RE = re.compile(
    r"model_not_found|(no such|unknown) model|no endpoints found for model"
    r"|model\b.{0,60}\b(not found|does not exist|doesn't exist)|(not found|does not exist).{0,30}\bmodel\b",
    re.I | re.S)
MODEL_LISTING_TRIES = 2
MODEL_LISTING_CAP = 5000


def _clean_msg(msg: str) -> str:
    """The framework's own words 'model call failed' are not the proxy saying a model is missing."""
    return msg.replace("model call failed", "call failed")


USAGE_GUESS_MARGIN = 1.5  # length-based estimate of a reply with no usage: len / 3 chars per token, times this
_MODELS = (config.MODEL_FAST, config.MODEL_ALT, config.MODEL_REASON)


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    truncated: bool = False  # stopped by a limit the budget cut: not the whole reply the caller asked for


class BudgetExhausted(Exception):
    pass


class LLMUnavailable(Exception):
    pass


class ModelFailing(LLMUnavailable):
    """This model keeps failing while the proxy answers: another model can help."""


class ProxyDown(LLMUnavailable):
    """A call that cannot be repeated in place ended by a timeout, and no model shows the proxy alive."""


@dataclass
class _ModelHealth:
    """One model since its last answer. It is out only while this says so: when the pause ends, it is tried again."""
    pause_until: float = float("-inf")
    pause_set_at: float = float("-inf")
    step: int = 0
    out: bool = False  # failed while the proxy answers: calls go elsewhere during the pause, one tries it after


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


def _billed_tokens(u) -> tuple[int, int]:
    """(prompt, completion) to charge. Reasoning counts on top of completion_tokens only when the proxy
    reports it outside them: the total shows it, or it is larger than the completion_tokens it should be in."""
    p = int(getattr(u, "input_tokens", 0) or 0)
    c = int(getattr(u, "output_tokens", 0) or 0)
    reasoning = int(getattr(u, "reasoning_tokens", 0) or 0)
    if reasoning > 0 and (int(getattr(u, "total_tokens", 0) or 0) >= p + c + reasoning or reasoning > c):
        c += reasoning
    return p, c


async def chat_or_alt(client, model: str, messages: list[dict], max_tokens: int, **extra) -> ChatResult:
    """For a route with no ladder of its own: `model`, then the others in a fixed order. Only when every model is
    failing it waits and asks again, until the proxy is declared dead or the unit's deadline comes."""
    models = [model] + [m for m in _MODELS if m != model]
    while True:
        for m in models:
            limit, params = max_tokens, extra
            if m == config.MODEL_REASON:  # with the routes' limits it would reason to the end and reply with nothing
                limit, params = max(max_tokens, config.REASON_FALLBACK_TOKENS), {**extra, "reasoning_effort": "low"}
            try:
                return await client.chat(m, messages, limit, **params)
            except (ModelFailing, ProxyDown):
                pass
        await client.wait_for_model(models)


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
        self._model_ids: dict[str, str] = {}  # our model name -> the id this proxy knows it by
        self._base_tried = False  # the other spelling of the base URL has been tried (once per run)
        self._base_orig: str | None = None  # set while the other spelling is on trial: the one to go back to
        self._listing_done = False  # GET {base}/models gave a listing (one per run)
        self._listing_tries = 0
        self._fix_gen = 0  # changes whenever the base URL or a model id changes
        self._fix_lock: asyncio.Lock | None = None
        self.structured_ok = True
        self._now = time.monotonic
        # the proxy as a whole: dead only when no model answers
        self._served = 0  # successful replies so far
        self._down_since: float | None = None
        self.probe_models = _MODELS
        self._health: defaultdict[str, _ModelHealth] = defaultdict(_ModelHealth)

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
            mc = ModelRequestConfig(model=self._model_ids.get(name, name), temperature=0.0)
            m = self._models[name] = Model(cc, mc)
        return m

    async def _fix_endpoint(self, model: str, status: int, msg: str, gen: int) -> bool:
        """A 404 on the address, or a model id the proxy does not know: the jury's proxy may spell either differently
        from ours. True when something changed (now or while this call was failing) and the call is worth repeating.
        The address is changed only while the original has never answered, and goes back if the other fails too."""
        if self._fix_lock is None:
            self._fix_lock = asyncio.Lock()
        async with self._fix_lock:
            if gen != self._fix_gen:  # another call fixed something while this one was failing
                return True
            if _MODEL_MISSING_RE.search(_clean_msg(msg)):
                return await self._map_model_ids()
            if status != 404:
                return False
            if self._base_orig is not None:  # the other spelling is on trial and failed as well
                self.base_url, self._base_orig = self._base_orig, None
                self._models.clear()
                self._fix_gen += 1
                return False
            if not self._base_tried and self._served == 0:
                self._base_tried = True
                self._base_orig = self.base_url
                base = self.base_url.rstrip("/")
                self.base_url = base[:-3] if base.endswith("/v1") else base + "/v1"
                self._models.clear()
                self._fix_gen += 1
                logger.warning("HTTP 404 before any answer; trying the other spelling of the base URL")
                return True
        return False

    async def _map_model_ids(self) -> bool:
        """GET {base}/models costs no tokens: ids that end with our names take their place. One listing per run
        (a failed attempt may be repeated once). Called under the fix lock: the others wait for it."""
        if self._listing_done or self._listing_tries >= MODEL_LISTING_TRIES:
            return False
        self._listing_tries += 1
        try:
            import httpx
            async with httpx.AsyncClient(timeout=min(self.timeout, 30.0)) as http:
                resp = await http.get(self.base_url.rstrip("/") + "/models",
                                      headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            data = resp.json().get("data", [])
            ids = [str(d["id"]) for d in data[:MODEL_LISTING_CAP] if isinstance(d, dict) and d.get("id")]
        except Exception as exc:
            logger.warning("could not read the model listing: %s", type(exc).__name__)
            return False
        self._listing_done = True
        low = {i.lower(): i for i in ids}
        changed = False
        for name in _MODELS:
            if name in ids:
                continue
            n = name.lower()
            found = [i for i in ids if i.lower().endswith("/" + n)] or ([low[n]] if n in low else [])
            if found:
                self._model_ids[name] = sorted(found, key=len)[0]
                changed = True
        if changed:
            self._models.clear()
            self._fix_gen += 1
            logger.warning("model ids taken from the proxy's listing: %s", self._model_ids)
        return changed

    def _proxy_ok(self, model: str, base: str | None = None) -> None:
        """A real answer from `base`. A ping never gets here: passing one says nothing of real work."""
        self._served += 1
        if self._base_orig is not None and base is not None:  # the other spelling is on trial
            if base == self._base_orig:  # the original answers: the trial was a misfire
                self.base_url, self._base_orig = base, None
                self._models.clear()
                self._fix_gen += 1
            elif base == self.base_url:
                self._base_orig = None  # the trial address answers: it is the right one
        self._down_since = None
        self._health[model] = _ModelHealth()

    def _proxy_failed(self, model: str, sent: float) -> None:
        """One failure that suggests an outage. The proxy is declared dead only after HARNESS_PROXY_DEAD_S
        without a success from any model; until then the pause of this model grows, once per drop:
        calls already in flight when it was set fail for the same reason and do not count."""
        now = self._now()
        if self._down_since is None:
            self._down_since = sent
        down = now - self._down_since
        if down >= config.proxy_dead_s():
            self.ledger.mark_exhausted()
            raise BudgetExhausted(f"no successful LLM response for {down:.0f} s")
        h = self._health[model]
        if sent > h.pause_set_at or h.out:  # a model that is out has one call at a time on it: always its own failure
            pause = config.proxy_pause(h.step)
            h.step += 1
            h.pause_set_at, h.pause_until = now, now + pause
            logger.warning("LLM %s failing, no answer from any model for %.0f s, pausing it for %s s",
                           model, down, pause)

    def _check_running(self) -> None:
        """For every waiting loop. On 3.11 wait_for can lose a cancel when a request ends in the same iteration of the
        loop, and only the unit's deadline cancels these waits: the request for it stays visible on the task."""
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()

    async def _ping(self, model: str) -> bool:
        """Does another model answer? Tells a failing model from a proxy outage without buying a dearer rung. It moves
        only this call on: the health of the pinged model and the dead clock stay, a ping is no proof of real work."""
        for other in self.probe_models:
            if other == model or self._now() < self._health[other].pause_until:
                continue
            sent = self._now()
            try:
                r = await asyncio.wait_for(self._model(other).invoke(_PING, max_tokens=1), self.timeout)
            except Exception as exc:
                status = _status_of(exc)
                if status in (402, 403):
                    self.ledger.mark_exhausted()
                    raise BudgetExhausted(f"HTTP {status}") from exc
                if status is None or status >= 500:
                    self._proxy_failed(other, sent)
                continue
            self._account(other, _PING, 1, r, False)
            return True
        return False

    async def _wait_pause(self, model: str) -> None:
        """Out the pause as it stands now: one renewed meanwhile by a failing call must not starve this one,
        or a single bad request would keep the others from proving the proxy alive. In short slices, so a
        success anywhere or the end of the run is noticed at once."""
        until, served = self._health[model].pause_until, self._served
        while True:
            self._check_running()
            left = min(until, self._health[model].pause_until) - self._now()
            if left <= 0 or self._served > served:
                return
            await asyncio.sleep(min(left, 1.0))

    async def wait_for_model(self, models: list[str]) -> None:
        """A moment before the caller asks `models` again: up to the end of the first pause among those that have one,
        in short slices, as a probe can end sooner than the pause it set. Always yields to the other workers."""
        self._check_running()
        pauses = [self._health[m].pause_until for m in models if self._health[m].step]
        await asyncio.sleep(min(max(min(pauses, default=0.0) - self._now(), 0.05), 1.0))

    async def _hold(self, prompt_cost: float, max_tokens: int, min_tokens: int) -> int:
        """The output limit this call may use, with its cost set aside in the ledger. Waits only while a call
        really in flight holds the budget: each of those ends within the LLM timeout."""
        while True:
            self._check_running()
            limit = self.ledger.hold(prompt_cost, max_tokens, min_tokens)
            if limit is None:
                await asyncio.sleep(0.05)
            elif limit > 0:
                return limit
            else:
                raise LLMUnavailable("call does not fit the remaining budget")

    async def chat(self, model: str, messages: list[dict], max_tokens: int, **extra) -> ChatResult:
        """An answer, or: ModelFailing when `model` keeps failing while another model answers (after its pause it
        gets a new try); ProxyDown after a timeout with no model answering; LLMUnavailable for the rest. 5xx and
        refused connections while no model answers wait here, until the proxy is back or is declared dead."""
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        if self._sem is None:
            # bound to the running loop on first use; one loop per client in production
            self._sem = asyncio.Semaphore(self._max_concurrency)
        async with self._sem:
            return await self._chat(model, messages, max_tokens, extra)

    async def _chat(self, model, messages, max_tokens, extra) -> ChatResult:
        try:
            m = self._model(model)
            m_base = self.base_url
        except Exception as exc:  # not the unit's fault: the task must stay open for the routes and the retry pass
            msg = f"{type(exc).__name__}: {exc}"[:300]
            record_call(model, 0, 0, 0.0, error=msg[:100])
            raise LLMUnavailable(msg) from exc
        attempt = 0
        tries = 0  # failed tries of this call that a pause and a retry can help
        served_before = self._served
        alive = False  # the proxy answers something, so one more failure of this model is about the model
        extra = dict(extra)
        stripped = False  # structured params dropped for this call as a trial
        recovered = 0  # times the base URL or the model id was changed for this call
        gen = self._fix_gen
        prompt_tokens = len(str(messages)) // 4
        prompt_cost = self.ledger.cost(prompt_tokens, 0)
        min_tokens = config.MIN_REASONING_CALL_TOKENS if "reasoning_effort" in extra else config.MIN_CALL_TOKENS
        while True:
            if not self.structured_ok or stripped:
                extra.pop("response_format", None)
                extra.pop("extra_body", None)
            if tries == 0 and self._health[model].out and self._now() < self._health[model].pause_until:
                raise ModelFailing(f"{model} keeps failing")
            if not alive:
                await self._wait_pause(model)
            limit = await self._hold(prompt_cost, max_tokens, min_tokens)
            sent, wait = self._now(), 0.0
            if self._health[model].out:  # this call is the probe: the others keep away until it ends
                self._health[model].pause_until = sent + self.timeout
            try:
                r = await asyncio.wait_for(m.invoke(messages, max_tokens=limit, **extra), self.timeout)
            except Exception as exc:
                status = _status_of(exc)
                msg = f"{type(exc).__name__}: {exc}"[:300]
                connect = status is None and _is_connect_error(exc)
                transport = status is None and not connect and _is_transport_failure(exc)
                if not connect and not transport:
                    # the request went out: the proxy bills its input whatever the answer (a connect error never
                    # left, a transport failure has its own flat charge below). Inside this call's hold, so the
                    # ledger never goes past what hold() allowed.
                    record_call(model, prompt_tokens, 0, self.ledger.charge(prompt_tokens, 0), error=msg[:100])
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
                delays = config.retry_delays()
                if status == 429 and attempt < len(delays):
                    wait = delays[attempt]
                    attempt += 1
                elif connect or (status is not None and status >= 500):
                    # no generation behind it (a 5xx was charged its input above): safe to repeat after the pause
                    if connect:
                        record_call(model, 0, 0, 0.0, error=msg[:100])
                    self._proxy_failed(model, sent)
                    tries += 1
                    if tries >= MODEL_FAILING_AFTER or self._health[model].out:
                        if alive:
                            self._health[model].out = True
                            raise ModelFailing(msg) from exc
                        alive = self._served > served_before or await self._ping(model)
                elif transport:
                    # may have been served and billed: a flat amount, never more than the call can cost; never repeat in place
                    charge = min(TIMEOUT_CHARGE, math.ceil(prompt_cost + limit))
                    record_call(model, 0, charge, self.ledger.charge(0, charge), error=msg[:100])
                    self._proxy_failed(model, sent)
                    if self._served > served_before or await self._ping(model):
                        self._health[model].out = True
                        raise ModelFailing(msg) from exc
                    raise ProxyDown(msg) from exc
                else:
                    # other 4xx (400/404/422, exhausted 429): deterministic, not a dead proxy; charged above
                    if status in (400, 404) and recovered < 2 and await self._fix_endpoint(model, status, msg, gen):
                        recovered += 1
                        m = self._model(model)
                        m_base = self.base_url
                        gen = self._fix_gen
                        continue
                    raise LLMUnavailable(msg) from exc
            else:
                self._proxy_ok(model, m_base)
                if stripped:
                    self.structured_ok = False
                return self._account(model, messages, limit, r, limit < max_tokens)
            finally:
                self.ledger.release(prompt_cost + limit)
            if wait:
                await asyncio.sleep(wait)

    def _account(self, model, messages, limit, r, cut: bool) -> ChatResult:
        u = getattr(r, "usage_metadata", None)
        p, c = (0, 0) if u is None else _billed_tokens(u)
        if u is None or not (p or c or not r.content):  # a reply was generated: it cost something
            p, c = len(str(messages)) // 4, limit
            if model.endswith("-noreason"):  # no hidden reasoning: the reply is all it cost
                c = min(limit, max(1, math.ceil(len((r.content or "").encode("utf-8")) / 3 * USAGE_GUESS_MARGIN)))
            logger.warning("no usage from %s; charging estimate %s/%s", model, p, c)
        cost = self.ledger.charge(p, c)
        record_call(model, p, c, cost)
        return ChatResult(text=r.content or "", prompt_tokens=p, completion_tokens=c, cost=cost,
                          truncated=cut and getattr(r, "finish_reason", None) == "length")

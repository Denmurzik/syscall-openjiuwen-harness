"""All thresholds and timeouts in one place."""
import logging
import os

# model names: the single place (chosen by measurement, see README)
MODEL_FAST = "qwen3.8-27b-noreason"
MODEL_ALT = "qwen3.6-35b-a3b-noreason"
MODEL_REASON = "qwen3.8-27b"
# 2c arbitration: both are asked independently, the phase-1 choice changes only if they agree
# (round 3: one model alone broke 4 of 42 correct choices on hard coding)
ARBITRATE_MODELS = (MODEL_FAST, MODEL_ALT)

PHASE1_WORKERS = 8
CODING_TIME_LIMIT = 85
PHASE1_TIMEOUT = {"coding": 100, "default": 15}


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return v if v >= 1 else default


# 3 is the dev provider's account ceiling (HTTP 429 above it); the final proxy allows up to 5
LLM_CONCURRENCY = _int_env("HARNESS_LLM_CONCURRENCY", 3)
LLM_TIMEOUT = 180
DEADLINE_S = 3300
STOP_NEW_BEFORE_DEADLINE_S = 150
RESERVE_FRACTION = 0.05
TIMEOUT_CHARGE = 1500  # charged for an LLM call that ended by timeout
MIN_CALL_TOKENS = 100  # a call cut to fit the budget is not made with a smaller output limit
MIN_REASONING_CALL_TOKENS = 1000  # the same for a call with reasoning: below this it reasons and never answers
REASON_FALLBACK_TOKENS = 1500  # output limit of a route's call that fell through to MODEL_REASON, at least
PROGRAM_MAX_TOKENS = 900  # output limit of a 2b program: search programs that read the KB were cut at 400
PROXY_PAUSE_MAX_S = 60

RETRY_DELAYS = (2, 4, 8, 16, 30)  # the shared account ceiling resets within ~40 s

_log = logging.getLogger("harness")
_warned: set[str] = set()


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
        if v != v or v < minimum or v in (float("inf"), float("-inf")):
            raise ValueError(raw)
        return v
    except ValueError:
        if name not in _warned:
            _warned.add(name)
            _log.warning("invalid %s=%r, using default %r", name, raw, default)
        return default


BATCH_2A = True  # default of HARNESS_BATCH_2A (round 6, task 32): measured, no accuracy loss, see the task report


# default of HARNESS_VERIFY_CODING. Round 7, task 38: 0 broken of 221 correct fixes on all sets; 0 fixed on the
# reviewer's set, seed 74 and LIB (none has a wrong green single fix), 2 fixed on a held-out set. On by the lead's
# ruling: hidden tests are probably stronger than the visible ones, see the report
VERIFY_CODING = True
VERIFY_MAX_TOKENS = 400  # output limit of the model writing its own fix, as the first rung of fix_coding


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if name not in _warned:
        _warned.add(name)
        _log.warning("invalid %s=%r, using default %r", name, raw, default)
    return default


def batch_2a() -> bool:
    """Several small 2a groups of one type in one call. Read at call time: HARNESS_BATCH_2A=1/0."""
    return _env_bool("HARNESS_BATCH_2A", BATCH_2A)


def verify_coding() -> bool:
    """A model checks green single coding fixes on its own. Read at call time: HARNESS_VERIFY_CODING=1/0."""
    return _env_bool("HARNESS_VERIFY_CODING", VERIFY_CODING)


# default of HARNESS_CONFIRM_2A. Round 8, task 40: a 2a answer is checked by an independent MODEL_FAST program; on a
# difference a MODEL_ALT program is the third vote and two votes against 2a replace it. See the task report
CONFIRM_2A = True
# search is left out: live, a search vote cost ~1240 a task (programs that read the KB) against ~150 for math and
# data, and it changed no answer on either gate set
CONFIRM_TYPES = ("math", "data_analysis")
CONFIRM_SPEND_LIMIT = 0.5  # no new vote once this share of the budget is spent (as the final REASON pass)
CONFIRM_GROUP_SAMPLE = 2   # a 2a group larger than this is accepted when its best-paid members' FAST votes all agree
VERIFY_UNIT_S = 60         # time set aside per 2d unit (per worker) when the confirmation stage runs before 2d


def confirm_2a() -> bool:
    """Majority vote of independent programs on 2a answers. Read at call time: HARNESS_CONFIRM_2A=1/0."""
    return _env_bool("HARNESS_CONFIRM_2A", CONFIRM_2A)


def llm_timeout() -> float:
    return _env_float("HARNESS_LLM_TIMEOUT", LLM_TIMEOUT, 0.001)


def deadline_s() -> float:
    return _env_float("HARNESS_DEADLINE_S", DEADLINE_S, 0.0)


def proxy_dead_s() -> float:
    """No successful reply for this long, with failures all along: the proxy is gone. The run deadline by default: at
    the end of the budget the proxy does nothing and the run is stopped, so there is nothing to detect early, while
    giving up on an outage would lose the rest of the hour."""
    return _env_float("HARNESS_PROXY_DEAD_S", deadline_s(), 0.0)


def stop_new_before_deadline_s() -> float:
    return _env_float("HARNESS_STOP_NEW_BEFORE_DEADLINE_S", STOP_NEW_BEFORE_DEADLINE_S, 0.0)


def phase1_timeout(task_type: str) -> float:
    if task_type == "coding":
        return _env_float("HARNESS_PHASE1_TIMEOUT_CODING", PHASE1_TIMEOUT["coding"], 0.001)
    default = _env_float("HARNESS_PHASE1_TIMEOUT_DEFAULT", PHASE1_TIMEOUT["default"], 0.001)
    return PHASE1_TIMEOUT.get(task_type, default) if task_type in PHASE1_TIMEOUT else default


def coding_time_limit() -> float:
    """The coding solver's own limit inside the phase-1 child: 15 s under the kill timeout (85 at the
    default 100), so it stops by itself (tentative answer, arbitration hint) before the parent kills it."""
    t = phase1_timeout("coding")
    return t - 15 if t > 30 else t * 0.5


def retry_delays() -> tuple:
    raw = os.environ.get("HARNESS_RETRY_DELAYS")
    if raw is None or raw.strip() == "":
        return RETRY_DELAYS
    try:
        vals = tuple(float(x) for x in raw.split(","))
        if any(v != v or v < 0 or v == float("inf") for v in vals):
            raise ValueError(raw)
        return vals
    except ValueError:
        if "HARNESS_RETRY_DELAYS" not in _warned:
            _warned.add("HARNESS_RETRY_DELAYS")
            _log.warning("invalid HARNESS_RETRY_DELAYS=%r, using default %r", raw, RETRY_DELAYS)
        return RETRY_DELAYS


def proxy_pause(step: int) -> float:
    """Length of the shared pause after `step` earlier ones of an outage: the retry delays, then doubling."""
    delays = retry_delays()
    if step < len(delays):
        return min(delays[step], PROXY_PAUSE_MAX_S)
    return min(delays[-1] * 2 ** min(step - len(delays) + 1, 20), PROXY_PAUSE_MAX_S)

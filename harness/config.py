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
RESERVE_FRACTION = 0.01
TIMEOUT_CHARGE = 1500  # charged for an LLM call that ended by timeout

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


def llm_timeout() -> float:
    return _env_float("HARNESS_LLM_TIMEOUT", LLM_TIMEOUT, 0.001)


def deadline_s() -> float:
    return _env_float("HARNESS_DEADLINE_S", DEADLINE_S, 0.0)


def stop_new_before_deadline_s() -> float:
    return _env_float("HARNESS_STOP_NEW_BEFORE_DEADLINE_S", STOP_NEW_BEFORE_DEADLINE_S, 0.0)


def phase1_timeout(task_type: str) -> float:
    if task_type == "coding":
        return _env_float("HARNESS_PHASE1_TIMEOUT_CODING", PHASE1_TIMEOUT["coding"], 0.001)
    default = _env_float("HARNESS_PHASE1_TIMEOUT_DEFAULT", PHASE1_TIMEOUT["default"], 0.001)
    return PHASE1_TIMEOUT.get(task_type, default) if task_type in PHASE1_TIMEOUT else default


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

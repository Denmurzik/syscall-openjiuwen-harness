"""All thresholds and timeouts in one place."""
import os

# model names: the single place (chosen by measurement, see README)
MODEL_FAST = "qwen3.8-27b-noreason"
MODEL_ALT = "qwen3.6-35b-a3b-noreason"
MODEL_REASON = "qwen3.8-27b"
MODEL_ARBITRATE = MODEL_FAST  # re-measured round 2 final: 146/146 hidden-correct, 0 regressions, same as MODEL_ALT

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

"""All thresholds and timeouts in one place."""
import os

PHASE1_WORKERS = 8
CODING_TIME_LIMIT = 90
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
